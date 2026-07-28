import logging
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import pandas as pd

from .config import NORMALIZATION
from .utils import (
    _create_comparison_pattern,
    _get_total_subjects_from_repartition,
    _normalize_disease_acronym,
    _split_and_clean_terms,
    normalize_boolean_like,
    normalize_method_label,
    normalize_metric_name,
    parse_probast_assessment,
    normalize_architecture,
    normalize_dataset,
    normalize_modality,
)

logger = logging.getLogger(__name__)

_FTD_SUBTYPE_TERMS = set(NORMALIZATION.get("constants", {}).get("ftd_subtypes", []))
_OOD_RIGOR_KEYWORDS = NORMALIZATION.get("constants", {}).get("ood_rigor_keywords", {})

_PERFORMANCE_LINE_PATTERN = re.compile(
    r"^\[([^\]]+)\]\s+(.+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?))?\s*$"
)

_PARENTHETICAL_SUFFIX_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")


def _expand_compound_term(term: str) -> list[str]:
    """Split one token into sub-terms when authors list multiple items inline."""
    if not term or not isinstance(term, str):
        return []

    parts = re.split(r"\s*(?:\+|;|\||&|\band\b)\s*", term, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part and part.strip()]


def _normalize_disease_for_differential(
    disease: str,
    context_diagnoses: list[str],
    preserve_ftd_subtypes: bool,
) -> Optional[str]:
    """Normalize disease terms for differential figures with optional FTD subtype bucket."""
    if not disease or not isinstance(disease, str):
        return None

    if disease.strip().upper() in _FTD_SUBTYPE_TERMS and preserve_ftd_subtypes:
        return "FTD subtypes"

    return _normalize_disease_acronym(disease, context_diagnoses=context_diagnoses)


def _strip_parenthetical_suffix(text: str) -> str:
    """Remove trailing parenthetical precision notes from a token."""
    if not text or not isinstance(text, str):
        return ""
    return _PARENTHETICAL_SUFFIX_PATTERN.sub("", text).strip()


def _split_metric_domain_token(token: str) -> tuple[str, str]:
    """Parse [domain|metric] or [metric|domain] token into (metric, domain)."""
    if not token:
        return "", "ID"

    parts = [part.strip() for part in token.split("|", 1)]
    if len(parts) == 1:
        return normalize_metric_name(parts[0]), "ID"

    left, right = parts
    left_upper = left.upper()
    right_upper = right.upper()

    if left_upper in {"ID", "OOD"}:
        return normalize_metric_name(right), left_upper
    if right_upper in {"ID", "OOD"}:
        return normalize_metric_name(left), right_upper

    return normalize_metric_name(token), "ID"


def _canonicalize_comparison_label(label: str) -> str:
    """Canonicalize comparison labels to avoid duplicate variants in outputs."""
    if not label or not isinstance(label, str):
        return ""

    normalized = re.sub(r"\s+", " ", label.strip())
    normalized = re.sub(r"\bSubtypes\b", "subtypes", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace(" vs ", " vs. ")
    return normalized


def _categorize_ood_rigor(ood_flag: Optional[str], evidence: Any, notes: Any) -> str:
    """Categorize OOD strategy from OOD flag and free-text evidence/notes."""
    if ood_flag != "Yes":
        return "No OOD"

    combined = f"{evidence or ''} {notes or ''}".strip().lower()
    if not combined:
        return "OOD reported (unspecified)"

    for category, keywords in _OOD_RIGOR_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            return category

    return "OOD reported (unspecified)"


def _filter_counter_by_min_frequency(counts: Counter, min_frequency: int) -> Counter:
    """Return only entries whose count is at least min_frequency."""
    return Counter(
        {key: value for key, value in counts.items() if value >= min_frequency}
    )


def _get_normalized_non_control_diseases(disease_text: Any) -> set:
    """Normalize diseases from one row and exclude controls."""
    if pd.isna(disease_text):
        return set()

    raw_diseases = [d.strip() for d in str(disease_text).split(",") if d.strip()]
    normalized_diseases = set()

    for disease in raw_diseases:
        normalized = _normalize_disease_acronym(disease, context_diagnoses=raw_diseases)
        if normalized and normalized != "CN":
            if normalized.startswith("MCI"):
                normalized = "MCI"
            normalized_diseases.add(normalized)

    return normalized_diseases


def _get_normalized_diseases_including_controls(disease_text: Any) -> set:
    """Normalize diseases from one row, preserving CN for task complexity counts."""
    if pd.isna(disease_text):
        return set()

    raw_diseases = [d.strip() for d in str(disease_text).split(",") if d.strip()]
    normalized_diseases = set()

    for disease in raw_diseases:
        normalized = _normalize_disease_acronym(disease, context_diagnoses=raw_diseases)
        if normalized:
            if normalized.startswith("MCI"):
                normalized = "MCI"
            normalized_diseases.add(normalized)

    return normalized_diseases


def analyze_year_distribution(data: list[dict[str, Any]]) -> Counter:
    """Counts the number of publications per year."""
    return Counter(
        row["Year"] for row in data if "Year" in row and pd.notna(row["Year"])
    )


def analyze_ml_dl_methods(data: list[dict[str, Any]]) -> Counter:
    """Counts the usage of ML, DL, or Hybrid methods."""
    ml_dl_counts = Counter()
    for row in data:
        if "ML / DL" in row and pd.notna(row["ML / DL"]):
            method = normalize_method_label(row["ML / DL"])
            ml_dl_counts.update([method])
    return ml_dl_counts


def analyze_architectures(data: list[dict[str, Any]]) -> Counter:
    """
    Counts architecture usage with article-level deduplication.
    """
    architecture_counts = Counter()

    for row in data:
        if "Architecture(s) Used" in row and pd.notna(row["Architecture(s) Used"]):
            architectures = _split_and_clean_terms(str(row["Architecture(s) Used"]))
            normalized_architectures_in_row = set()

            for arch in architectures:
                if not arch:
                    continue

                for arch_part in _expand_compound_term(arch):
                    normalized_arch = normalize_architecture(arch_part)

                    if normalized_arch != "Other":
                        normalized_architectures_in_row.add(normalized_arch)

            architecture_counts.update(normalized_architectures_in_row)

    return architecture_counts


def analyze_modalities(data: list[dict[str, Any]]) -> Counter:
    """
    Counts modality usage after cleaning and normalization.
    Deduplicates normalized modalities within each row to avoid double-counting.
    """
    modality_counts = Counter()

    for row in data:
        if "Modalities" in row and pd.notna(row["Modalities"]):
            modalities = _split_and_clean_terms(str(row["Modalities"]))
            normalized_modalities_in_row = set()

            for mod in modalities:
                if not mod:
                    continue

                normalized_mod = normalize_modality(mod)

                if normalized_mod != "Other":
                    normalized_modalities_in_row.add(normalized_mod)

            modality_counts.update(normalized_modalities_in_row)

    return modality_counts


def analyze_modality_overlap_sets(
    data: list[dict[str, Any]], top_n: int = 6
) -> dict[str, Any]:
    """Return study sets for the top-N modalities to feed an overlap diagram."""
    modality_counts = analyze_modalities(data)
    top_modalities = [mod for mod, _ in modality_counts.most_common(top_n)]
    modality_study_sets = {mod: set() for mod in top_modalities}

    for study_index, row in enumerate(data):
        if "Modalities" not in row or pd.isna(row["Modalities"]):
            continue

        modalities = _split_and_clean_terms(str(row["Modalities"]))
        normalized_modalities_in_row = set()
        for mod in modalities:
            if not mod:
                continue

            normalized_mod = normalize_modality(mod)
            if normalized_mod != "Other":
                normalized_modalities_in_row.add(normalized_mod)

        for modality in top_modalities:
            if modality in normalized_modalities_in_row:
                modality_study_sets[modality].add(study_index)

    return {
        "top_modalities": top_modalities,
        "study_sets": [modality_study_sets[mod] for mod in top_modalities],
        "counts": modality_counts,
    }


def analyze_diseases(data: list[dict[str, Any]]) -> Counter:
    """
    Counts disease usage after normalization with article-level deduplication.
    Each normalized disease is counted at most once per article.
    Controls (CN) are excluded and MCI-stage variants are grouped under MCI.
    """
    disease_counts = Counter()

    for row in data:
        if "Diseases" in row:
            disease_counts.update(
                _get_normalized_non_control_diseases(row.get("Diseases"))
            )

    return disease_counts


def analyze_code_availability(data: list[dict[str, Any]]) -> dict[str, int]:
    """Determines the number of studies with and without available code."""
    code_availability = Counter()
    for row in data:
        availability_flag = normalize_boolean_like(row.get("Code Available"))
        code_value = row.get("Code")
        has_code_entry = pd.notna(code_value) and str(code_value).strip() not in [
            "",
            "/",
        ]

        if availability_flag == "Yes" or has_code_entry:
            code_availability["Yes"] += 1
        else:
            code_availability["No"] += 1
    return dict(code_availability)


def analyze_multi_center(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes whether studies are single-center or multi-center."""
    counts = Counter()
    for row in data:
        normalized = normalize_boolean_like(row.get("Multi-Center"))
        if normalized is None:
            counts["Unknown"] += 1
        elif normalized == "Yes":
            counts["Multi-center"] += 1
        else:
            counts["Single-center"] += 1
    return dict(counts)


def analyze_multi_center_trends(
    data: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    """Analyzes multi-center/single-center trends by year."""
    by_year = defaultdict(Counter)
    for row in data:
        year_raw = row.get("Year")
        if pd.isna(year_raw):
            continue

        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue

        normalized = normalize_boolean_like(row.get("Multi-Center"))
        if normalized == "Yes":
            by_year[year]["Multi-center"] += 1
        elif normalized == "No":
            by_year[year]["Single-center"] += 1
        else:
            by_year[year]["Unknown"] += 1

    return dict(sorted(by_year.items()))


def analyze_ci_reporting_status(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes confidence interval reporting status from explicit column values."""
    counts = Counter()
    for row in data:
        normalized = normalize_boolean_like(
            row.get("Confidence Intervals Reported"),
            allow_partial=True,
        )
        counts[normalized or "Unknown"] += 1
    return dict(counts)


def analyze_probast_risk(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Parses PROBAST assessments into overall/domain-level risk counts."""
    overall_counts = Counter()
    domain_counts = {
        "P1": Counter(),
        "P2": Counter(),
        "P3": Counter(),
        "P4": Counter(),
    }

    for row in data:
        parsed = parse_probast_assessment(row.get("PROBAST"))
        overall_counts[parsed.get("overall") or "Unknown"] += 1
        for domain in ("P1", "P2", "P3", "P4"):
            domain_counts[domain][parsed.get(domain) or "Unknown"] += 1

    return {
        "overall": dict(overall_counts),
        "domains": {domain: dict(counts) for domain, counts in domain_counts.items()},
    }


def analyze_ood_rigor(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyzes OOD rigor categories and their yearly trends."""
    category_counts = Counter()
    yearly = defaultdict(Counter)

    for row in data:
        ood_flag = normalize_boolean_like(row.get("OOD"))
        category = _categorize_ood_rigor(
            ood_flag,
            evidence=row.get("OOD_evidence"),
            notes=row.get("OOD_notes"),
        )
        category_counts[category] += 1

        year_raw = row.get("Year")
        if pd.isna(year_raw):
            continue
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        yearly[year][category] += 1

    return {
        "categories": dict(category_counts),
        "yearly": dict(sorted(yearly.items())),
    }


def analyze_reproducibility_signals(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes reproducibility signals for code, data, and pretrained weights."""
    counts = Counter()

    for row in data:
        code_flag = normalize_boolean_like(row.get("Code Available"))
        code_value = row.get("Code")
        has_code = code_flag == "Yes" or (
            pd.notna(code_value) and str(code_value).strip() not in ["", "/"]
        )

        data_flag = normalize_boolean_like(row.get("Data Availability"))
        has_data = data_flag == "Yes"

        weights_flag = normalize_boolean_like(row.get("Pretrained Weights Available"))
        has_weights = weights_flag == "Yes"

        if has_code:
            counts["Code available"] += 1
        if has_data:
            counts["Data available"] += 1
        if has_weights:
            counts["Pretrained weights"] += 1
        if has_code and has_data and has_weights:
            counts["All three"] += 1

    return dict(counts)


def _categorize_validation_rigor(row: dict[str, Any]) -> str:
    """Assign a validation-rigor category from OOD, multi-center, and CV strategy."""
    ood_flag = normalize_boolean_like(row.get("OOD")) == "Yes"
    multicenter_flag = normalize_boolean_like(row.get("Multi-Center")) == "Yes"
    cv_text = str(row.get("Cross-Validation Strategy") or "").strip().lower()

    if ood_flag and multicenter_flag:
        return "External OOD + multi-center"

    independent_tokens = [
        "independent test",
        "independent cohort",
        "external test",
        "external cohort",
        "hold-out",
        "holdout",
        "train/test",
    ]
    robust_cv_tokens = [
        "nested",
        "repeated",
        "bootstrap",
        "monte carlo",
    ]
    standard_cv_tokens = [
        "cross-validation",
        "cv",
        "k-fold",
        "loocv",
        "leave-one-out",
    ]

    if any(token in cv_text for token in independent_tokens):
        return "Independent hold-out/test"
    if any(token in cv_text for token in robust_cv_tokens):
        return "Nested/repeated CV"
    if any(token in cv_text for token in standard_cv_tokens):
        return "Standard CV"

    return "Unclear"


def analyze_validation_rigor_by_method(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute validation-rigor profile proportions for each method family (ML/DL/Hybrid)."""
    rigor_categories = [
        "External OOD + multi-center",
        "Independent hold-out/test",
        "Nested/repeated CV",
        "Standard CV",
    ]

    method_totals = Counter()
    method_category_counts = defaultdict(lambda: defaultdict(int))

    for row in data:
        method = normalize_method_label(row.get("ML / DL"))
        if not method:
            continue
        rigor_category = _categorize_validation_rigor(row)
        method_totals[method] += 1
        method_category_counts[method][rigor_category] += 1

    profiles = {}
    for method, total in method_totals.items():
        profiles[method] = {
            category: method_category_counts[method].get(category, 0) / total
            if total > 0
            else 0.0
            for category in rigor_categories
        }

    return {
        "axes": rigor_categories,
        "profiles": profiles,
        "counts": dict(method_totals),
    }


def _normalize_data_availability_bucket(value: Any) -> Optional[str]:
    """Normalize data availability text to compact categories for matrix plots."""
    raw = str(value or "").strip().lower()
    if not raw or raw in {"/", "na", "n/a", "none"}:
        return None
    if "public" in raw:
        return "Public"
    if "mixed" in raw:
        return "Mixed"
    if "request" in raw:
        return "On request"
    if "proprietary" in raw or "private" in raw:
        return "Proprietary"
    return None


def analyze_reproducibility_matrix(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Code x Data-Availability matrix with pretrained-weights overlays."""
    row_labels = ["Code: Yes", "Code: No"]
    col_labels = ["Public", "Mixed", "On request", "Proprietary"]

    counts = defaultdict(Counter)
    weights_yes = defaultdict(Counter)
    invalid_rows = []

    for row_index, row in enumerate(data, start=1):
        code_value = row.get("Code")
        has_code_entry = pd.notna(code_value) and str(code_value).strip() not in {
            "",
            "/",
        }
        has_code = (
            normalize_boolean_like(row.get("Code Available")) == "Yes" or has_code_entry
        )
        code_bucket = "Code: Yes" if has_code else "Code: No"

        data_bucket = _normalize_data_availability_bucket(row.get("Data Availability"))
        if data_bucket is None:
            invalid_rows.append(
                {
                    "row": row_index,
                    "value": str(row.get("Data Availability") or "").strip(),
                }
            )
            continue

        counts[code_bucket][data_bucket] += 1

        has_weights = (
            normalize_boolean_like(row.get("Pretrained Weights Available")) == "Yes"
        )
        if has_weights:
            weights_yes[code_bucket][data_bucket] += 1

    if invalid_rows:
        sample = ", ".join(
            [
                f"row={entry['row']} value='{entry['value']}'"
                for entry in invalid_rows[:10]
            ]
        )
        extra = "" if len(invalid_rows) <= 10 else f" (+{len(invalid_rows) - 10} more)"
        raise ValueError(
            "Reproducibility matrix failed: 'Data Availability' must map to one of "
            "{Public, Mixed, On request, Proprietary}. Invalid entries: "
            f"{sample}{extra}"
        )

    matrix_counts = [
        [counts[row_label].get(col_label, 0) for col_label in col_labels]
        for row_label in row_labels
    ]
    matrix_weights = [
        [weights_yes[row_label].get(col_label, 0) for col_label in col_labels]
        for row_label in row_labels
    ]

    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "counts": matrix_counts,
        "weights_yes": matrix_weights,
    }


def _categorize_gold_standard_strength(value: Any) -> str:
    """Map free-text gold-standard description to strength buckets."""
    text = str(value or "").strip().lower()
    if not text or text in {"/", "na", "n/a", "none"}:
        return "Unclear / missing"

    if any(token in text for token in ["neuropath", "autopsy", "histopath"]):
        return "Neuropathological"

    biomarker_tokens = [
        "biomarker",
        "amyloid",
        "tau",
        "pet",
        "csf",
        "datscan",
        "spect",
    ]
    if any(token in text for token in biomarker_tokens):
        return "Clinical + biomarker"

    if "clinical" in text or "consensus" in text or "criteria" in text:
        return "Clinical only"

    return "Unclear / missing"


def analyze_gold_standard_performance(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build study-level performance points grouped by gold-standard strength and domain."""
    performance_records = _parse_normalized_performances(data)
    by_row_and_domain = defaultdict(list)

    metric_priority = [
        "ACC",
        "BACC",
        "AUC",
        "F1",
        "MCC",
        "PRECISION",
        "SENSITIVITY",
        "SPECIFICITY",
        "NPV",
    ]

    for record in performance_records:
        by_row_and_domain[
            (record["source_row_index"], record.get("domain", "ID"))
        ].append(record)

    points = []
    for row_index, row in enumerate(data):
        strength = _categorize_gold_standard_strength(row.get("Gold Standard"))

        for domain in ["ID", "OOD"]:
            records = by_row_and_domain.get((row_index, domain), [])
            if not records:
                continue

            selected = []
            selected_metric = None
            for metric in metric_priority:
                metric_records = [r for r in records if r.get("metric") == metric]
                if metric_records:
                    selected = metric_records
                    selected_metric = metric
                    break
            if not selected:
                continue

            mean_perf = sum(r["performance"] for r in selected) / len(selected)
            points.append(
                {
                    "strength": strength,
                    "domain": domain,
                    "performance": round(mean_perf, 3),
                    "metric": selected_metric,
                    "method": normalize_method_label(row.get("ML / DL")),
                    "source_row_index": row_index,
                }
            )

    return points


def analyze_interpretability_usage(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes interpretability/xAI methods usage."""
    counts = Counter()

    feature_importance_keywords = [
        "feature importance",
        "importance ranking",
        "permutation importance",
        "gini importance",
        "xgboost feature importance",
        "feature selection",
        "coefficien",
        "weight vector",
        "weighting factor",
        "relieff",
        "rfe",
        "wilks",
        "discriminant function",
    ]

    map_based_keywords = [
        "map",
        "mapping",
        "atlas",
        "roi",
        "voxel",
        "topograph",
        "brainpainter",
        "disease coordinate",
        "spatial distribution",
        "grading",
        "displacement",
        "variance diagram",
        "weight image",
        "bsage",
    ]

    visualization_keywords = [
        "visualization",
        "t-sne",
        "umap",
        "latent space",
        "chord diagram",
        "graph",
        "tree",
        "eigenbrain",
        "neurosynth decoding",
    ]

    decision_keywords = [
        "normative modeling",
        "mahalanobis",
        "calibration curve",
        "decision curve",
        "nomogram",
        "rad-score",
        "class membership score",
        "volumetric report",
        "percentile table",
        "probability indices",
        "disease state index",
    ]

    for row in data:
        raw_value = row.get("Interpretability Method")
        if pd.isna(raw_value):
            counts["Not reported"] += 1
            continue

        text = str(raw_value).strip().lower()
        if text in {"", "/", "nan", "none", "not reported", "nr", "n/a"}:
            counts["Not reported"] += 1
            continue

        if "shap" in text:
            counts["SHAP"] += 1
        elif "lime" in text:
            counts["LIME"] += 1
        elif "integrated gradients" in text or re.search(r"\big\b", text):
            counts["Integrated Gradients"] += 1
        elif (
            "grad-cam" in text
            or "class activation map" in text
            or re.search(r"\bcam\b", text)
        ):
            counts["CAM / Grad-CAM"] += 1
        elif "attention" in text:
            counts["Attention maps"] += 1
        elif "saliency" in text or "guided backprop" in text:
            counts["Saliency / gradient maps"] += 1
        elif any(keyword in text for keyword in feature_importance_keywords):
            counts["Feature importance / coefficients"] += 1
        elif any(keyword in text for keyword in decision_keywords):
            counts["Clinical score / decision analysis"] += 1
        elif any(keyword in text for keyword in map_based_keywords):
            counts["Statistical / structural maps"] += 1
        elif any(keyword in text for keyword in visualization_keywords):
            counts["Latent-space / graph visualization"] += 1
        else:
            counts["Other reported"] += 1

    return dict(counts)


def analyze_method_radar_profile(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute study-level readiness profiles for ML / DL / Hybrid radar plotting."""
    axes = [
        "Public dataset",
        "In-house dataset",
        "Multi-center",
        "OOD validation",
        "Code available",
        "Confidence intervals",
        "Balanced metric",
        "Biomarker/pathology labels",
        "Interpretability / xAI",
    ]

    balanced_metric_rows = set()
    for row_index, row in enumerate(data):
        metrics_raw = row.get("Metrics Used")
        if pd.isna(metrics_raw):
            continue
        metrics = {
            normalize_metric_name(metric)
            for metric in str(metrics_raw).split(",")
            if metric and str(metric).strip()
        }
        if {"BACC", "MCC"}.intersection(metrics):
            balanced_metric_rows.add(row_index)

    performance_records = _parse_normalized_performances(data)
    for record in performance_records:
        if record.get("metric") in {"BACC", "MCC"}:
            balanced_metric_rows.add(record["source_row_index"])

    method_totals = Counter()
    method_axis_counts = defaultdict(Counter)

    for row_index, row in enumerate(data):
        method = normalize_method_label(row.get("ML / DL"))
        if not method:
            continue

        method_totals[method] += 1

        public_dataset = False
        inhouse_dataset = False
        datasets_raw = row.get("Datasets")
        if pd.notna(datasets_raw):
            for dataset_term in _split_and_clean_terms(str(datasets_raw)):
                for dataset_part in _expand_compound_term(dataset_term):
                    normalized_dataset = normalize_dataset(dataset_part)
                    if normalized_dataset == "In-house":
                        inhouse_dataset = True
                    elif normalized_dataset != "Other":
                        public_dataset = True

        biomarker_labels = False
        neuropathological = normalize_boolean_like(
            row.get("Neuropathological"), allow_partial=True
        )
        if neuropathological == "Yes":
            biomarker_labels = True
        else:
            gold_standard = str(row.get("Gold Standard") or "").strip().lower()
            biomarker_keywords = [
                "biomarker",
                "neuropath",
                "patholog",
                "autopsy",
                "histopath",
                "biological",
            ]
            biomarker_labels = any(
                keyword in gold_standard for keyword in biomarker_keywords
            )

        code_value = row.get("Code")
        has_code_entry = pd.notna(code_value) and str(code_value).strip() not in {
            "",
            "/",
        }
        has_code = (
            normalize_boolean_like(row.get("Code Available")) == "Yes" or has_code_entry
        )

        interpretability_raw = row.get("Interpretability Method")
        has_interpretability = pd.notna(interpretability_raw) and str(
            interpretability_raw
        ).strip() not in {
            "",
            "/",
            "not reported",
        }

        flags = {
            "Public dataset": public_dataset,
            "In-house dataset": inhouse_dataset,
            "Multi-center": normalize_boolean_like(row.get("Multi-Center")) == "Yes",
            "OOD validation": normalize_boolean_like(row.get("OOD")) == "Yes",
            "Code available": has_code,
            "Confidence intervals": normalize_boolean_like(
                row.get("Confidence Intervals Reported")
            )
            == "Yes",
            "Balanced metric": row_index in balanced_metric_rows,
            "Biomarker/pathology labels": biomarker_labels,
            "Interpretability / xAI": has_interpretability,
        }

        for axis, enabled in flags.items():
            if enabled:
                method_axis_counts[method][axis] += 1

    profiles = {}
    for method, total in method_totals.items():
        profiles[method] = {
            axis: method_axis_counts[method].get(axis, 0) / total if total else 0.0
            for axis in axes
        }

    return {
        "axes": axes,
        "profiles": profiles,
        "counts": dict(method_totals),
    }


def analyze_accuracy_vs_complexity(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build study-level points for reported performance vs. diagnostic complexity."""
    performance_records = _parse_normalized_performances(data)
    records_by_row = defaultdict(list)
    for record in performance_records:
        records_by_row[record["source_row_index"]].append(record)

    metric_priority = [
        "ACC",
        "BACC",
        "AUC",
        "F1",
        "MCC",
        "PRECISION",
        "SENSITIVITY",
        "SPECIFICITY",
        "NPV",
    ]

    points: list[dict[str, Any]] = []

    for row_index, row in enumerate(data):
        class_count = len(
            _get_normalized_diseases_including_controls(row.get("Diseases"))
        )
        if class_count < 2:
            continue

        row_records = records_by_row.get(row_index, [])
        if not row_records:
            continue

        preferred_domains = (
            ["OOD", "ID"]
            if any(record.get("domain") == "OOD" for record in row_records)
            else ["ID", "OOD"]
        )

        selected_records = []
        selected_domain = None
        selected_metric = None

        for domain in preferred_domains:
            domain_records = [
                record for record in row_records if record.get("domain") == domain
            ]
            if not domain_records:
                continue

            for metric in metric_priority:
                metric_records = [
                    record
                    for record in domain_records
                    if record.get("metric") == metric
                ]
                if metric_records:
                    selected_records = metric_records
                    selected_domain = domain
                    selected_metric = metric
                    break

            if selected_records:
                break

        if not selected_records:
            continue

        mean_performance = sum(
            record["performance"] for record in selected_records
        ) / len(selected_records)

        points.append(
            {
                "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
                "classes": class_count,
                "performance": round(mean_performance, 3),
                "domain": selected_domain or "ID",
                "metric": selected_metric or selected_records[0].get("metric", ""),
                "method": normalize_method_label(row.get("ML / DL")),
                "authors": row.get("Authors", ""),
                "source_row_index": row_index,
            }
        )

    return points


def analyze_metrics(data: list[dict[str, Any]]) -> Counter:
    """Counts the usage of different evaluation metrics."""
    metrics_counts = Counter()
    for row in data:
        if "Metrics Used" in row and pd.notna(row["Metrics Used"]):
            metrics = [m.strip() for m in str(row["Metrics Used"]).split(",")]
            normalized = [normalize_metric_name(m) for m in metrics if m]
            metrics_counts.update([m for m in normalized if m])
    return metrics_counts


def analyze_ml_dl_trends(data: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    """Analyzes the trend of ML/DL method usage over the years."""
    ml_dl_by_year = defaultdict(Counter)
    for row in data:
        if (
            "Year" in row
            and "ML / DL" in row
            and pd.notna(row["Year"])
            and pd.notna(row["ML / DL"])
        ):
            year = int(row["Year"])
            method = normalize_method_label(row["ML / DL"])
            ml_dl_by_year[year].update([method])
    return dict(sorted(ml_dl_by_year.items()))


def analyze_gpu_usage(data: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    """Analyzes GPU usage specifically within DL and Hybrid studies."""
    gpu_availability = Counter()
    dl_studies_count = 0
    for row in data:
        method = str(row.get("ML / DL", "")).strip()
        if "DL" in method:  # Captures 'DL' and 'ML, DL'
            dl_studies_count += 1
            if (
                "GPUs" in row
                and pd.notna(row["GPUs"])
                and str(row["GPUs"]).strip() not in ["", "/"]
            ):
                gpu_availability["Yes"] += 1
            else:
                gpu_availability["No"] += 1
    return dict(gpu_availability), dl_studies_count


def analyze_datasets(data: list[dict[str, Any]], min_frequency: int = 2) -> Counter:
    """
    Counts usage of datasets after robust normalization with deduplication.
    Only includes neurological datasets (excludes "Other" category and non-neuro datasets).
    Filters out datasets that appear fewer than min_frequency times.
    """
    dataset_counts = Counter()
    neuro_datasets = set(NORMALIZATION["datasets"]["neuro"].keys())

    for row in data:
        if "Datasets" in row and pd.notna(row["Datasets"]):
            datasets = _split_and_clean_terms(str(row["Datasets"]))
            normalized_datasets_in_row = set()

            for d_name in datasets:
                if not d_name:
                    continue

                for dataset_part in _expand_compound_term(d_name):
                    normalized_dataset = normalize_dataset(dataset_part)

                    if normalized_dataset == "Other":
                        continue
                    if normalized_dataset in neuro_datasets:
                        normalized_datasets_in_row.add(normalized_dataset)

            dataset_counts.update(normalized_datasets_in_row)

    return _filter_counter_by_min_frequency(dataset_counts, min_frequency)


def analyze_diseases_vs_modalities(data: list[dict[str, Any]]) -> Counter:
    """Computes frequency of (raw disease term count, raw modality term count)."""
    frequencies = Counter()

    for row in data:
        diseases_raw = row.get("Diseases")
        modalities_raw = row.get("Modalities")

        diseases = (
            _split_and_clean_terms(str(diseases_raw)) if pd.notna(diseases_raw) else []
        )
        modalities = (
            _split_and_clean_terms(str(modalities_raw))
            if pd.notna(modalities_raw)
            else []
        )

        num_diseases = len(diseases)
        num_modalities = len(modalities)

        if num_diseases > 0 and num_modalities > 0:
            frequencies[(num_diseases, num_modalities)] += 1

    return frequencies


def analyze_global_disease_repartition(
    data: list[dict[str, Any]], min_frequency: int = 2
) -> Counter:
    """
    Aggregates the total number of subjects for each disease across all studies
    by parsing the 'Repartition' column.
    Filters out diseases that appear in fewer than min_frequency studies.
    """
    disease_counts = Counter()
    disease_study_counts = Counter()  # Track how many studies each disease appears in

    for row in data:
        repartition_text = row.get("Repartition")
        if pd.notna(repartition_text):
            diseases_in_study = set()
            pairs = re.findall(r"(\d+)\s*([A-Za-z\-]+)", str(repartition_text))
            for count_str, disease_acronym in pairs:
                normalized_disease = _normalize_disease_acronym(disease_acronym)
                if normalized_disease and normalized_disease != "CN":
                    disease_counts[normalized_disease] += int(count_str)
                    diseases_in_study.add(normalized_disease)
            for disease in diseases_in_study:
                disease_study_counts[disease] += 1
    return Counter(
        {
            disease: count
            for disease, count in disease_counts.items()
            if disease_study_counts[disease] >= min_frequency
        }
    )


def analyze_dataset_size_by_method(data: list[dict[str, Any]]) -> dict[str, list[int]]:
    """
    Analyzes the distribution of total dataset sizes (number of subjects)
    for each ML/DL method.
    """
    sizes_by_method = defaultdict(list)
    years_by_method = defaultdict(list)
    for row in data:
        method = row.get("ML / DL")
        repartition = row.get("Repartition")

        if pd.notna(method) and pd.notna(repartition):
            method_str = normalize_method_label(method)

            total_subjects = _get_total_subjects_from_repartition(str(repartition))

            if total_subjects > 0:
                sizes_by_method[method_str].append(total_subjects)
                years_by_method[method_str].append(row.get("Year"))

    return dict(sizes_by_method), dict(years_by_method)


def _get_frequent_diseases(data: list[dict[str, Any]], min_frequency: int = 2) -> set:
    """
    Get the set of diseases that appear in at least min_frequency studies.
    This ensures consistent filtering across all analyses.
    """
    diagnosis_counts = Counter()
    for row in data:
        diagnosis_counts.update(
            _get_normalized_non_control_diseases(row.get("Diseases"))
        )

    return {
        disease for disease, count in diagnosis_counts.items() if count >= min_frequency
    }


def analyze_disease_dataset_architecture_flow(
    data: list[dict[str, Any]], min_disease_frequency: int = 2
) -> list[dict[str, Any]]:
    """
    Analyzes the flow between diseases, datasets, and architectures for Sankey diagram.
    Returns a list of flow records with normalized categories.
    Filters out diseases and datasets that appear in fewer than min_disease_frequency studies.
    """
    neuro_datasets = set()
    for dataset_name in NORMALIZATION["datasets"]["neuro"].keys():
        neuro_datasets.add(dataset_name)

    frequent_diseases = _get_frequent_diseases(data, min_disease_frequency)

    frequent_datasets = _get_frequent_datasets(data, min_disease_frequency)

    flow_data = []

    for row in data:
        diseases = set()
        if "Diseases" in row and pd.notna(row["Diseases"]):
            raw_diseases = [d.strip() for d in str(row["Diseases"]).split(",")]
            for disease in raw_diseases:
                normalized = _normalize_disease_acronym(
                    disease, context_diagnoses=raw_diseases
                )
                if (
                    normalized
                    and normalized != "CN"
                    and normalized in frequent_diseases
                ):
                    diseases.add(normalized)

        datasets = set()
        if "Datasets" in row and pd.notna(row["Datasets"]):
            raw_datasets = _split_and_clean_terms(str(row["Datasets"]))
            for dataset in raw_datasets:
                if dataset:
                    for dataset_part in _expand_compound_term(dataset):
                        normalized = normalize_dataset(dataset_part)
                        if (
                            normalized != "Other"
                            and normalized in neuro_datasets
                            and normalized in frequent_datasets
                        ):
                            datasets.add(normalized)

        architectures = set()
        if "Architecture(s) Used" in row and pd.notna(row["Architecture(s) Used"]):
            raw_architectures = _split_and_clean_terms(str(row["Architecture(s) Used"]))
            for arch in raw_architectures:
                if arch:
                    for arch_part in _expand_compound_term(arch):
                        normalized = normalize_architecture(arch_part)
                        if normalized != "Other":  # Exclude "Other" category
                            architectures.add(normalized)

        for disease in diseases:
            for dataset in datasets:
                for architecture in architectures:
                    flow_data.append(
                        {
                            "disease": disease,
                            "dataset": dataset,
                            "architecture": architecture,
                        }
                    )

    return flow_data


def _get_frequent_datasets(data: list[dict[str, Any]], min_frequency: int = 2) -> set:
    """
    Get the set of datasets that appear in at least min_frequency studies.
    This ensures consistent filtering across all analyses.
    """
    neuro_datasets = set(NORMALIZATION["datasets"]["neuro"].keys())

    dataset_counts = Counter()
    for row in data:
        if "Datasets" in row and pd.notna(row["Datasets"]):
            datasets = _split_and_clean_terms(str(row["Datasets"]))
            normalized_datasets_in_row = set()

            for d_name in datasets:
                if not d_name:
                    continue

                for dataset_part in _expand_compound_term(d_name):
                    normalized_dataset = normalize_dataset(dataset_part)

                    if (
                        normalized_dataset != "Other"
                        and normalized_dataset in neuro_datasets
                    ):
                        normalized_datasets_in_row.add(normalized_dataset)

            dataset_counts.update(normalized_datasets_in_row)

    frequent_datasets = {
        dataset for dataset, count in dataset_counts.items() if count >= min_frequency
    }

    return frequent_datasets


def analyze_differential_diagnosis_patterns(
    data: list[dict[str, Any]],
    min_disease_frequency: int = 2,
    preserve_ftd_subtypes: bool = False,
) -> dict[str, Any]:
    """
    Analyzes differential diagnosis patterns from the 'Normalized Performances' column.
    Returns pairwise comparisons, comparison types, and performance data over time.
    """
    frequent_diseases = _get_frequent_diseases(data, min_disease_frequency)

    performance_records = _parse_normalized_performances(
        data,
        preserve_ftd_subtypes=preserve_ftd_subtypes,
    )

    if preserve_ftd_subtypes and "FTD" in frequent_diseases:
        frequent_diseases = set(frequent_diseases)
        frequent_diseases.add("FTD subtypes")

    pairwise_comparisons = Counter()
    comparison_types = Counter()
    performance_over_time = []

    study_comparisons = {}
    study_pairwise = {}

    for record in performance_records:
        if not record.get("comparison"):
            continue

        study_id = record["source_row_index"]

        all_diseases_frequent = all(
            disease in frequent_diseases for disease in record["normalized_diseases"]
        )

        if all_diseases_frequent:
            if study_id not in study_comparisons:
                study_comparisons[study_id] = set()
            study_comparisons[study_id].add(record["comparison"])

            if record["year"] is not None:
                performance_over_time.append(
                    {
                        "year": record["year"],
                        "metric": record["metric"],
                        "comparison": record["comparison"],
                        "domain": record["domain"],
                        "performance": record["performance"],
                        "ci": record.get("ci"),
                        "method": record["method"],
                        "dataset_size": record["dataset_size"],
                    }
                )

        if study_id not in study_pairwise:
            study_pairwise[study_id] = set()

        frequent_unique_diseases = [
            d
            for d in list(set(record["normalized_diseases"]))
            if d in frequent_diseases
        ]

        if len(frequent_unique_diseases) >= 2:
            for i in range(len(frequent_unique_diseases)):
                for j in range(i + 1, len(frequent_unique_diseases)):
                    pair = tuple(
                        sorted(
                            [frequent_unique_diseases[i], frequent_unique_diseases[j]]
                        )
                    )
                    study_pairwise[study_id].add(pair)

    for comparisons in study_comparisons.values():
        comparison_types.update(comparisons)

    for pairwise in study_pairwise.values():
        pairwise_comparisons.update(pairwise)

    return {
        "pairwise_comparisons": pairwise_comparisons,
        "comparison_types": comparison_types,
        "performance_over_time": performance_over_time,
    }


def _parse_normalized_performances(
    data: list[dict[str, Any]],
    preserve_ftd_subtypes: bool = False,
) -> list[dict[str, Any]]:
    """
    Common function to parse the 'Normalized Performances' column.
    Returns a list of parsed performance records with all necessary information.
    """
    performance_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(data):
        normalized_performances = row.get("Normalized Performances", "")
        year = row.get("Year", None)
        ml_dl = row.get("ML / DL", "")
        repartition = row.get("Repartition", "")
        authors = row.get("Authors", "")

        # Extract context diseases for disambiguation (e.g., MD → Mild Dementia vs. Major Depression)
        context_diagnoses = []
        if "Diseases" in row and pd.notna(row["Diseases"]):
            context_diagnoses = [d.strip() for d in str(row["Diseases"]).split(",")]

        if not normalized_performances or pd.isna(normalized_performances):
            continue

        dataset_size = 1
        if repartition and not pd.isna(repartition):
            total_subjects = _get_total_subjects_from_repartition(str(repartition))
            if total_subjects > 0:
                dataset_size = total_subjects

        lines = str(normalized_performances).strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = _PERFORMANCE_LINE_PATTERN.match(line)
            if not match:
                continue

            metric_token = match.group(1).strip()
            metric, metric_domain = _split_metric_domain_token(metric_token)
            lhs_expression = match.group(2).strip()
            performance_value = float(match.group(3))
            ci_value = float(match.group(4)) if match.group(4) else None

            has_parenthetical_precision = bool(
                _PARENTHETICAL_SUFFIX_PATTERN.search(lhs_expression)
            )
            lhs_clean = _strip_parenthetical_suffix(lhs_expression)

            diseases = [d.strip() for d in re.split(r" vs\.? ", lhs_clean) if d.strip()]
            is_single_class_metric = len(diseases) == 1

            normalized_diseases = []
            for disease in diseases:
                disease_clean = _strip_parenthetical_suffix(disease)
                normalized = _normalize_disease_for_differential(
                    disease_clean,
                    context_diagnoses=context_diagnoses,
                    preserve_ftd_subtypes=preserve_ftd_subtypes,
                )
                if normalized:
                    if normalized.startswith("MCI"):
                        normalized = "MCI"
                    normalized_diseases.append(normalized)

            if not normalized_diseases:
                continue

            comparison_pattern = None
            if len(normalized_diseases) >= 2:
                comparison_pattern = _create_comparison_pattern(normalized_diseases)
                if not comparison_pattern:
                    continue

            method = str(ml_dl).strip()
            method = normalize_method_label(method)

            performance_records.append(
                {
                    "source_row_index": row_index,
                    "year": int(year) if year and not pd.isna(year) else None,
                    "metric": metric.strip(),
                    "domain": metric_domain,
                    "comparison": comparison_pattern,
                    "performance": performance_value,
                    "ci": ci_value,
                    "has_ci": ci_value is not None,
                    "raw_lhs": lhs_expression,
                    "has_parenthetical_precision": has_parenthetical_precision,
                    "is_single_class_metric": is_single_class_metric,
                    "target_disease": normalized_diseases[0]
                    if is_single_class_metric and normalized_diseases
                    else None,
                    "method": method,
                    "dataset_size": dataset_size,
                    "normalized_diseases": normalized_diseases,
                    "authors": authors,
                }
            )

    return performance_records


def analyze_performance_over_time(
    data: list[dict[str, Any]],
    preserve_ftd_subtypes: bool = False,
) -> list[dict[str, Any]]:
    """
    Analyzes performance metrics over time for differential diagnosis patterns.
    Returns data suitable for scatter plotting with performance trends.
    """
    performance_records = _parse_normalized_performances(
        data,
        preserve_ftd_subtypes=preserve_ftd_subtypes,
    )

    performance_data = []
    for record in performance_records:
        if record["year"] is not None:
            comparison_label = record.get("comparison")
            if not comparison_label and record.get("target_disease"):
                comparison_label = f"{record['target_disease']} (single-class)"
            if not comparison_label:
                comparison_label = (
                    str(record.get("raw_lhs", "Unspecified")).strip() or "Unspecified"
                )

            comparison_label = _canonicalize_comparison_label(comparison_label)

            performance_data.append(
                {
                    "source_row_index": record["source_row_index"],
                    "year": record["year"],
                    "metric": record["metric"],
                    "domain": record["domain"],
                    "comparison": comparison_label,
                    "performance": record["performance"],
                    "ci": record.get("ci"),
                    "method": record["method"],
                    "dataset_size": record["dataset_size"],
                    "authors": record["authors"],
                }
            )

    return performance_data


def analyze_ci_consistency(data: list[dict[str, Any]]) -> dict[str, int]:
    """Compares CI-reported column against actual ± occurrences in performance lines."""
    performance_records = _parse_normalized_performances(data)
    has_ci_by_row = defaultdict(bool)
    for record in performance_records:
        if record.get("has_ci"):
            has_ci_by_row[record["source_row_index"]] = True

    confusion = Counter()
    for row_index, row in enumerate(data):
        reported = (
            normalize_boolean_like(
                row.get("Confidence Intervals Reported"),
                allow_partial=True,
            )
            or "Unknown"
        )
        has_ci = "Yes" if has_ci_by_row[row_index] else "No"
        confusion[f"Reported={reported} | InlineCI={has_ci}"] += 1

    return dict(confusion)


def analyze_ci_width_by_metric(
    data: list[dict[str, Any]],
    min_frequency: int = 3,
) -> dict[str, list[float]]:
    """Collects CI widths by domain/metric, filtered by minimum observations."""
    performance_records = _parse_normalized_performances(data)
    widths_by_metric = defaultdict(list)

    for record in performance_records:
        if record.get("ci") is None:
            continue
        key = f"{record['domain']} | {record['metric']}"
        widths_by_metric[key].append(float(record["ci"]))

    return {
        key: values
        for key, values in widths_by_metric.items()
        if len(values) >= min_frequency
    }


def analyze_neuropathological_data(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes the presence of neuropathological data in studies."""
    neuropath_counts = Counter()
    for row in data:
        normalized = normalize_boolean_like(
            row.get("Neuropathological"), allow_partial=True
        )
        if normalized == "Yes":
            neuropath_counts["Yes"] += 1
        else:
            neuropath_counts["No"] += 1
    return dict(neuropath_counts)


def analyze_ood_validation(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes the presence of out-of-distribution (OOD) validation in studies."""
    ood_counts = Counter()
    for row in data:
        normalized = normalize_boolean_like(row.get("OOD"), allow_partial=True)
        if normalized == "Yes":
            ood_counts["Yes"] += 1
        else:
            ood_counts["No"] += 1
    return dict(ood_counts)


def analyze_inhouse_dataset_usage(data: list[dict[str, Any]]) -> dict[str, int]:
    """Analyzes the usage of in-house datasets by counting unique appearances."""
    inhouse_studies = set()
    total_studies = 0

    for i, row in enumerate(data):
        total_studies += 1
        if "Datasets" in row and pd.notna(row["Datasets"]):
            datasets = _split_and_clean_terms(str(row["Datasets"]))
            for dataset in datasets:
                normalized_dataset = normalize_dataset(dataset)
                if normalized_dataset == "In-house":
                    inhouse_studies.add(i)  # Use row index as unique identifier
                    break  # Only count once per study

    return {"Yes": len(inhouse_studies), "No": total_studies - len(inhouse_studies)}


def analyze_studies_with_many_diseases(
    data: list[dict[str, Any]], min_diseases: int = 7
) -> int:
    """Counts studies with at least min_diseases normalized disease classes."""
    if min_diseases <= 0:
        return len(data)

    studies_count = 0
    for row in data:
        diseases_raw = row.get("Diseases")
        if pd.isna(diseases_raw):
            continue

        raw_terms = [d.strip() for d in str(diseases_raw).split(",") if d.strip()]
        if not raw_terms:
            continue

        normalized_diseases = set()
        for disease in raw_terms:
            normalized = _normalize_disease_acronym(
                disease, context_diagnoses=raw_terms
            )
            if normalized:
                if normalized.startswith("MCI"):
                    normalized = "MCI"
                normalized_diseases.add(normalized)

        if len(normalized_diseases) >= min_diseases:
            studies_count += 1

    return studies_count


def _normalize_target_comparison_terms(terms: list[str]) -> Optional[str]:
    """Builds a normalized comparison pattern from CLI terms."""
    cleaned_terms = [str(t).strip() for t in terms if str(t).strip()]
    if len(cleaned_terms) < 2:
        return None

    normalized_terms = []
    for term in cleaned_terms:
        normalized = _normalize_disease_acronym(term, context_diagnoses=cleaned_terms)
        if normalized:
            if normalized.startswith("MCI"):
                normalized = "MCI"
            normalized_terms.append(normalized)

    if len(normalized_terms) < 2:
        return None

    return _create_comparison_pattern(normalized_terms)


def analyze_target_comparison_counts(
    data: list[dict[str, Any]], target_comparisons: list[list[str]]
) -> dict[str, int]:
    """
    Counts studies matching each target comparison in 'Normalized Performances'.

    Matching is order-invariant and counted at most once per study and per target.
    """
    normalized_targets = []
    for target in target_comparisons:
        normalized_target = _normalize_target_comparison_terms(target)
        if normalized_target:
            normalized_targets.append(normalized_target)

    if not normalized_targets:
        return {}

    ordered_unique_targets = list(dict.fromkeys(normalized_targets))
    counts = Counter({target: 0 for target in ordered_unique_targets})

    performance_records = _parse_normalized_performances(data)
    comparisons_by_study = defaultdict(set)
    for record in performance_records:
        comparison = record.get("comparison")
        if comparison:
            comparisons_by_study[record["source_row_index"]].add(comparison)

    for comparisons_in_study in comparisons_by_study.values():
        for target in ordered_unique_targets:
            if target in comparisons_in_study:
                counts[target] += 1

    return dict(counts)
