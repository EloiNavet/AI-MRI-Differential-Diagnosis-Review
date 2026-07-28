import logging
import re
from typing import Any, Optional, Sequence, Union

import pandas as pd
from matplotlib.colors import to_rgb

from .config import NORMALIZATION

logger = logging.getLogger(__name__)

_DEFAULT_METHOD_LABEL_ALIASES = {
    "ml, dl": "Hybrid (ML+DL)",
    "hybrid": "Hybrid (ML+DL)",
    "hybrid (ml+dl)": "Hybrid (ML+DL)",
}

_DEFAULT_ARCHITECTURE_PRIORITY = [
    "CNN",
    "Advanced DL",
    "Transformer",
    "Logistic Regression",
    "Random Forest",
    "Tree-Based Methods",
    "Linear Methods",
    "SVM",
    "MLP/NN",
    "Feature/Statistical Methods",
    "Other Classical ML",
    "Specialized Methods",
]

_DEFAULT_DATASET_OTHER_VALUES = ["none", "na", "n/a", "", "/"]
_DEFAULT_DATASET_INHOUSE_KEYWORDS = [
    "private",
    "in-house",
    "hospital",
    "clinic",
    "center",
    "centre",
]

_DEFAULT_BOOL_TRUE_VALUES = {"true", "yes", "y", "1"}
_DEFAULT_BOOL_FALSE_VALUES = {"false", "no", "n", "0"}
_DEFAULT_BOOL_PARTIAL_VALUES = {"partial", "mixed"}
_STRICT_NORMALIZATION_IGNORED_TERMS = {
    "rejected paper",
    "not applicable - rejected",
}


def _signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    value_clean = str(value).strip().lower()
    null_like_values = {
        str(item).strip().lower()
        for item in NORMALIZATION.get(
            "null_like_values", ["", "/", "na", "n/a", "none", "nan"]
        )
    }
    return value_clean in null_like_values


def _split_and_clean_terms(text: str) -> list[str]:
    """Split comma-separated terms while respecting nested (), [], {} groups."""
    if not isinstance(text, str):
        return []

    terms: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            current.append(char)
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
            continue
        if char == "[":
            bracket_depth += 1
            current.append(char)
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
            continue
        if char == "{":
            brace_depth += 1
            current.append(char)
            continue
        if char == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
            continue

        if char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            token = "".join(current).strip()
            if token:
                terms.append(token)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        terms.append(tail)

    return terms


def _is_explicit_alias_match(value: str, alias: str) -> bool:
    """Return True when value matches alias explicitly (exact/signature/word-boundary)."""
    if not value or not alias:
        return False

    value_clean = str(value).strip()
    alias_clean = str(alias).strip()
    if not value_clean or not alias_clean:
        return False

    if value_clean.lower() == alias_clean.lower():
        return True

    if _signature(value_clean) == _signature(alias_clean):
        return True

    return bool(
        re.search(r"\b" + re.escape(alias_clean) + r"\b", value_clean, re.IGNORECASE)
    )


def normalize_architecture(raw_name: str) -> str:
    """Map an architecture token to a YAML category, with config priority for overlaps."""
    if not raw_name or not isinstance(raw_name, str):
        return "Other"

    def _matches(category_name: str) -> bool:
        for pattern in NORMALIZATION["architectures"].get(category_name, []):
            if _is_explicit_alias_match(raw_name, pattern):
                return True
        return False

    priority_order = NORMALIZATION.get("architecture_priority") or list(
        _DEFAULT_ARCHITECTURE_PRIORITY
    )
    for canonical_name in priority_order:
        if canonical_name in NORMALIZATION["architectures"] and _matches(
            canonical_name
        ):
            return canonical_name

    for canonical_name in NORMALIZATION["architectures"].keys():
        if canonical_name in priority_order:
            continue
        if _matches(canonical_name):
            return canonical_name

    return "Other"


def normalize_dataset(dataset_name: str) -> str:
    """
    Normalizes dataset names to standard forms using the NORMALIZATION config.
    Now handles the nested structure with 'neuro' and 'other' categories.
    """
    dataset_name = dataset_name.strip()

    other_values = {
        v.lower()
        for v in (
            NORMALIZATION.get("dataset_other_values") or _DEFAULT_DATASET_OTHER_VALUES
        )
    }
    if dataset_name.lower() in other_values:
        return "Other"

    inhouse_keywords = (
        NORMALIZATION.get("dataset_inhouse_keywords")
        or _DEFAULT_DATASET_INHOUSE_KEYWORDS
    )
    if any(keyword in dataset_name.lower() for keyword in inhouse_keywords):
        return "In-house"

    for _category_name, category_datasets in NORMALIZATION["datasets"].items():
        for standard_name, variations in category_datasets.items():
            for variation in variations:
                if _is_explicit_alias_match(dataset_name, variation):
                    return standard_name

    return "Other"


def _has_explicit_dataset_mapping(dataset_name: str) -> bool:
    """Return True only when a dataset term matches an explicit canonical/alias entry."""
    if not dataset_name or not isinstance(dataset_name, str):
        return False

    dataset_clean = dataset_name.strip()
    dataset_lower = dataset_clean.lower()

    other_values = {
        v.lower()
        for v in (
            NORMALIZATION.get("dataset_other_values") or _DEFAULT_DATASET_OTHER_VALUES
        )
    }
    if dataset_lower in other_values:
        return True

    inhouse_keywords = (
        NORMALIZATION.get("dataset_inhouse_keywords")
        or _DEFAULT_DATASET_INHOUSE_KEYWORDS
    )
    if any(keyword in dataset_lower for keyword in inhouse_keywords):
        return True

    for _category_name, category_datasets in NORMALIZATION["datasets"].items():
        for canonical_name, variations in category_datasets.items():
            candidates = [canonical_name, *(variations or [])]
            for candidate in candidates:
                candidate_clean = str(candidate).strip()
                if not candidate_clean:
                    continue

                if dataset_lower == candidate_clean.lower():
                    return True

                if _is_explicit_alias_match(dataset_clean, candidate_clean):
                    return True

    return False


def normalize_modality(modality: str) -> str:
    """Normalize modality names using the NORMALIZATION config mappings."""
    modality_clean = modality.strip()
    for category, variants in NORMALIZATION["modalities"].items():
        for variant in variants:
            if _is_explicit_alias_match(modality_clean, variant):
                return category
    return "Other"


def _normalize_disease_acronym(acronym: str, context_diagnoses: list = None) -> str:
    """
    Standardizes a disease acronym from NORMALIZATION["diagnosis"].
    """
    diagnosis = _diagnose_disease_term(acronym, context_diagnoses=context_diagnoses)
    return diagnosis["normalized"]


def _diagnose_disease_term(
    acronym: str, context_diagnoses: Optional[list] = None
) -> dict[str, Any]:
    """Return normalization + strict ambiguity metadata for one disease token."""
    if not acronym or not isinstance(acronym, str):
        return {
            "raw": acronym,
            "normalized": None,
            "ambiguous": False,
            "reason": "invalid_term",
        }

    acronym_clean = acronym.strip().upper()
    normalized = None
    matched_pattern = None

    for canonical, variants in NORMALIZATION["diagnosis"].items():
        canonical_upper = str(canonical).strip().upper()
        if acronym_clean == canonical_upper:
            matched_pattern = str(canonical)
            if canonical != "Unclear":
                normalized = canonical
            break

        for variant in variants:
            if acronym_clean == str(variant).strip().upper():
                matched_pattern = str(variant)
                if canonical != "Unclear":
                    normalized = canonical
                break

        if matched_pattern is not None:
            break

    ambiguous = False
    reason = None

    if normalized is None:
        unclear_variants = {
            str(variant).strip().upper()
            for variant in NORMALIZATION["diagnosis"].get("Unclear", [])
        }
        if acronym_clean in unclear_variants and reason is None:
            reason = "term_in_unclear_bucket"

    if normalized is None and reason is None:
        reason = "unmapped_term"

    return {
        "raw": acronym,
        "normalized": normalized,
        "ambiguous": ambiguous,
        "reason": reason,
    }


def _get_total_subjects_from_repartition(text: str) -> int:
    """
    Calculates the total number of subjects by summing individual counts
    from the 'Repartition' column string. This is more robust than trusting a
    pre-calculated total which might be inconsistent.
    e.g., "195: 85AD, 91CN, 19FTLD" -> 85 + 91 + 19 = 195
    """
    if not text or not isinstance(text, str):
        return 0

    disease_count_regex = r"(\d+)\s*[A-Za-z]+"
    counts = re.findall(disease_count_regex, text)

    total_subjects = sum(int(c) for c in counts)

    if total_subjects == 0:
        leading_total_regex = r"^\s*(\d+)\s*:"
        match = re.match(leading_total_regex, text)
        if match:
            return int(match.group(1))

    return total_subjects


def get_inverse_color(color: Union[str, Sequence[float]]) -> tuple[float, float, float]:
    """Return black or white for best contrast on a given background color."""
    try:
        r, g, b = to_rgb(color)
    except (TypeError, ValueError):
        logger.warning("Unable to parse color '%s'. Falling back to black text.", color)
        return (0.0, 0.0, 0.0)

    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    return (1.0, 1.0, 1.0) if luminance < 0.6 else (0.0, 0.0, 0.0)


def normalize_method_label(method: str) -> str:
    """Normalize method labels to the canonical plotting/reporting naming."""
    method_str = str(method).strip()
    method_aliases = dict(_DEFAULT_METHOD_LABEL_ALIASES)
    method_aliases.update(NORMALIZATION.get("method_label_aliases", {}))
    for alias, canonical_name in method_aliases.items():
        if method_str.lower() == str(alias).strip().lower():
            return canonical_name
    return method_str


def normalize_boolean_like(
    value: Any,
    allow_partial: bool = True,
) -> Optional[str]:
    """Normalize boolean-like values to 'Yes', 'No', or optional 'Partial'."""
    if _is_null_like(value):
        return None

    value_clean = str(value).strip().lower()
    true_values = {
        str(v).strip().lower()
        for v in NORMALIZATION.get("boolean_true_values", _DEFAULT_BOOL_TRUE_VALUES)
    }
    false_values = {
        str(v).strip().lower()
        for v in NORMALIZATION.get("boolean_false_values", _DEFAULT_BOOL_FALSE_VALUES)
    }
    partial_values = {
        str(v).strip().lower()
        for v in NORMALIZATION.get(
            "boolean_partial_values", _DEFAULT_BOOL_PARTIAL_VALUES
        )
    }

    if value_clean in true_values:
        return "Yes"
    if value_clean in false_values:
        return "No"
    if allow_partial and value_clean in partial_values:
        return "Partial"
    return None


def normalize_metric_name(metric: str) -> str:
    """Normalize metrics using YAML aliases and return a canonical label preserving original casing."""
    if not metric or not isinstance(metric, str):
        return ""

    metric_clean = metric.strip()
    metric_upper = metric_clean.upper()

    metrics_aliases = NORMALIZATION.get("metrics", {})
    for canonical, aliases in metrics_aliases.items():
        canonical_upper = str(canonical).strip().upper()
        if metric_upper == canonical_upper:
            return str(canonical).strip()
        for alias in aliases:
            if metric_upper == str(alias).strip().upper():
                return str(canonical).strip()

    return metric_clean


def parse_probast_assessment(text: Any) -> dict[str, Optional[str]]:
    """Parse PROBAST semi-structured text into P1..P4 and overall labels."""
    parsed = {
        "P1": None,
        "P2": None,
        "P3": None,
        "P4": None,
        "overall": None,
    }

    if _is_null_like(text):
        return parsed

    raw = str(text)

    def _normalize_risk_label(label: str) -> Optional[str]:
        cleaned = str(label or "").strip().lower()
        if not cleaned:
            return None
        if cleaned.startswith("high"):
            return "High"
        if cleaned.startswith("low"):
            return "Low"
        if cleaned.startswith("unclear"):
            return "Unclear"
        if cleaned.startswith("unknown"):
            return "Unknown"
        return cleaned.title()

    for domain in ("P1", "P2", "P3", "P4"):
        domain_match = re.search(
            rf"{domain}\s*\([^)]*\)\s*=\s*([A-Za-z]+)", raw, flags=re.IGNORECASE
        )
        if domain_match:
            parsed[domain] = _normalize_risk_label(domain_match.group(1))
            continue

        shorthand_match = re.search(
            rf"{domain}\s*\(\s*([A-Za-z]+)\s*\)", raw, flags=re.IGNORECASE
        )
        if shorthand_match:
            parsed[domain] = _normalize_risk_label(shorthand_match.group(1))
            continue

        # Supports compact forms like "P1=Low" or "P1 : Unclear".
        direct_match = re.search(
            rf"{domain}\s*[:=]\s*([A-Za-z][A-Za-z\s\-]*)",
            raw,
            flags=re.IGNORECASE,
        )
        if direct_match:
            parsed[domain] = _normalize_risk_label(direct_match.group(1))

    overall_match = re.search(r"overall\s*=\s*([A-Za-z]+)", raw, flags=re.IGNORECASE)
    if overall_match:
        parsed["overall"] = _normalize_risk_label(overall_match.group(1))
    else:
        # Supports compact forms like "overall: High".
        overall_direct_match = re.search(
            r"overall\s*[:=]\s*([A-Za-z][A-Za-z\s\-]*)",
            raw,
            flags=re.IGNORECASE,
        )
        if overall_direct_match:
            parsed["overall"] = _normalize_risk_label(overall_direct_match.group(1))

    return parsed


def _known_metric_tokens_upper() -> set:
    """Return all canonical metric names and aliases in uppercase."""
    metric_aliases = NORMALIZATION.get("metrics", {})
    known = set()

    for canonical_name, aliases in metric_aliases.items():
        canonical_clean = str(canonical_name).strip()
        if canonical_clean:
            known.add(canonical_clean.upper())

        for alias in aliases or []:
            alias_clean = str(alias).strip()
            if alias_clean:
                known.add(alias_clean.upper())

    return known


def _extract_metric_tokens_from_normalized_performances(text: Any) -> list[str]:
    """Extract metric tokens from lines like '[ID|AUC] AD vs. CN = 0.91'."""
    if _is_null_like(text):
        return []

    metric_tokens = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^\[([^\]]+)\]", line)
        if not match:
            continue

        raw_token = match.group(1).strip()
        if not raw_token:
            continue

        parts = [part.strip() for part in raw_token.split("|", 1)]
        if len(parts) == 2:
            left_upper = parts[0].upper()
            right_upper = parts[1].upper()
            if left_upper in {"ID", "OOD"}:
                metric_tokens.append(parts[1])
                continue
            if right_upper in {"ID", "OOD"}:
                metric_tokens.append(parts[0])
                continue

        metric_tokens.append(raw_token)

    return metric_tokens


def collect_normalization_issues(
    data: list[dict[str, Any]],
    include_modalities: bool = True,
    include_datasets: bool = True,
    include_metrics: bool = True,
) -> list[dict[str, Any]]:
    """Collect row-level strict normalization issues for fail-fast validation."""
    issues: list[dict[str, Any]] = []
    known_metrics_upper = _known_metric_tokens_upper() if include_metrics else set()

    for row_index, row in enumerate(data):
        csv_row = row_index + 2

        diseases_raw = row.get("Diseases")
        if not pd.isna(diseases_raw):
            disease_terms = [
                term.strip() for term in str(diseases_raw).split(",") if term.strip()
            ]
            for term in disease_terms:
                if _is_null_like(term):
                    continue
                if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                    continue

                diagnosis = _diagnose_disease_term(
                    term,
                    context_diagnoses=disease_terms,
                )

                if diagnosis["normalized"] is None:
                    issues.append(
                        {
                            "row": csv_row,
                            "column": "Diseases",
                            "term": term,
                            "reason": diagnosis["reason"],
                            "normalized": None,
                            "issue_type": "unmapped",
                        }
                    )

        architectures_raw = row.get("Architecture(s) Used")
        if not pd.isna(architectures_raw):
            architecture_terms = _split_and_clean_terms(str(architectures_raw))
            for term in architecture_terms:
                if _is_null_like(term):
                    continue
                if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                    continue

                normalized_arch = normalize_architecture(term)
                if normalized_arch == "Other":
                    issues.append(
                        {
                            "row": csv_row,
                            "column": "Architecture(s) Used",
                            "term": term,
                            "reason": "unmapped_term",
                            "normalized": None,
                            "issue_type": "unmapped",
                        }
                    )

        if include_modalities:
            modalities_raw = row.get("Modalities")
            if not pd.isna(modalities_raw):
                modality_terms = _split_and_clean_terms(str(modalities_raw))
                for term in modality_terms:
                    if _is_null_like(term):
                        continue
                    if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                        continue
                    normalized_modality = normalize_modality(term)
                    if normalized_modality == "Other":
                        issues.append(
                            {
                                "row": csv_row,
                                "column": "Modalities",
                                "term": term,
                                "reason": "unmapped_term",
                                "normalized": None,
                                "issue_type": "unmapped",
                            }
                        )

        if include_datasets:
            datasets_raw = row.get("Datasets")
            if not pd.isna(datasets_raw):
                dataset_terms = _split_and_clean_terms(str(datasets_raw))
                for term in dataset_terms:
                    if _is_null_like(term):
                        continue
                    if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                        continue

                    normalized_dataset = normalize_dataset(term)
                    has_explicit_mapping = _has_explicit_dataset_mapping(term)
                    if normalized_dataset == "Other" or not has_explicit_mapping:
                        issues.append(
                            {
                                "row": csv_row,
                                "column": "Datasets",
                                "term": term,
                                "reason": (
                                    "unmapped_term"
                                    if normalized_dataset == "Other"
                                    else "implicit_match_without_alias"
                                ),
                                "normalized": (
                                    None
                                    if normalized_dataset == "Other"
                                    else normalized_dataset
                                ),
                                "issue_type": "unmapped",
                            }
                        )

        if include_metrics:
            metrics_used_raw = row.get("Metrics Used")
            if not pd.isna(metrics_used_raw):
                metrics_used_terms = list(
                    dict.fromkeys(_split_and_clean_terms(str(metrics_used_raw)))
                )
                for term in metrics_used_terms:
                    if _is_null_like(term):
                        continue
                    if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                        continue

                    if str(term).strip().upper() not in known_metrics_upper:
                        issues.append(
                            {
                                "row": csv_row,
                                "column": "Metrics Used",
                                "term": term,
                                "reason": "unmapped_term",
                                "normalized": None,
                                "issue_type": "unmapped",
                            }
                        )

            normalized_perf_raw = row.get("Normalized Performances")
            if not pd.isna(normalized_perf_raw):
                perf_metric_terms = list(
                    dict.fromkeys(
                        _extract_metric_tokens_from_normalized_performances(
                            normalized_perf_raw
                        )
                    )
                )
                for term in perf_metric_terms:
                    if _is_null_like(term):
                        continue
                    if term.strip().lower() in _STRICT_NORMALIZATION_IGNORED_TERMS:
                        continue

                    if str(term).strip().upper() not in known_metrics_upper:
                        issues.append(
                            {
                                "row": csv_row,
                                "column": "Normalized Performances",
                                "term": term,
                                "reason": "unmapped_metric",
                                "normalized": None,
                                "issue_type": "unmapped",
                            }
                        )

    return issues


def _create_comparison_pattern(diseases: list[str]) -> str:
    """Create a standardized comparison pattern from a list of diseases."""
    if not diseases or len(diseases) < 2:
        return None

    disease_counts = {}
    for disease in diseases:
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    # If both the parent label and subtype bucket appear in the same comparison,
    # keep only the subtype bucket to avoid labels like "FTD vs. FTD subtypes".
    if "FTD subtypes" in disease_counts and "FTD" in disease_counts:
        disease_counts["FTD subtypes"] += disease_counts.pop("FTD")

    unique_diseases = list(disease_counts.keys())

    if len(unique_diseases) == 1:
        disease = unique_diseases[0]
        count = disease_counts[disease]
        if count > 1:
            if disease.lower().endswith("subtypes"):
                return disease
            return f"{disease} subtypes"
        else:
            return None

    diseases_with_duplicates = []
    for disease in unique_diseases:
        count = disease_counts[disease]
        if count > 1:
            if disease.lower().endswith("subtypes"):
                diseases_with_duplicates.append(disease)
            else:
                diseases_with_duplicates.append(f"{disease} subtypes")
        else:
            diseases_with_duplicates.append(disease)

    diseases_with_duplicates.sort()

    return " vs. ".join(diseases_with_duplicates)
