import argparse
import logging
import os
import sys
from collections import Counter
from typing import Any, Optional

import pandas as pd

from . import analyzer, plotting
from .config import (
    ANALYSIS_CONFIG,
    OUTPUT_FORMATS,
    REQUIRED_FIELDS,
    set_scientific_style,
)
from .utils import collect_normalization_issues, normalize_boolean_like

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_FINAL_OUTPUT_EXPECTED_COLUMNS = [
    "Title",
    "Year",
    "Authors",
    "Labs",
    "Venue",
    "Diseases",
    "Repartition",
    "Modalities",
    "Datasets",
    "Neuropathological",
    "Gold Standard",
    "Multi-Center",
    "Cross-Validation Strategy",
    "Confidence Intervals Reported",
    "Statistical Test",
    "Comparison to Clinical Baseline",
    "Data Augmentation",
    "Preprocessing Steps",
    "OOD",
    "OOD_evidence",
    "OOD_notes",
    "ML / DL",
    "Architecture(s) Used",
    "Interpretability Method",
    "Metrics Used",
    "Normalized Performances",
    "Code Available",
    "Code",
    "Pretrained Weights Available",
    "Data Availability",
    "GPUs",
    "PROBAST",
    "Additional notes",
]


def _log_final_output_schema(df: pd.DataFrame) -> None:
    """Log expected/missing columns for final_output schema."""
    missing = [col for col in _FINAL_OUTPUT_EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in _FINAL_OUTPUT_EXPECTED_COLUMNS]

    if missing:
        logger.warning("Missing expected final_output columns: %s", ", ".join(missing))
    else:
        logger.info(
            "All expected final_output columns are present (%d).",
            len(_FINAL_OUTPUT_EXPECTED_COLUMNS),
        )

    if extra:
        logger.info("Additional columns detected (kept as-is): %s", ", ".join(extra))


def _log_boolean_column_quality(df: pd.DataFrame) -> None:
    """Log normalization quality for key boolean-like columns (compact format)."""
    columns = [
        "Multi-Center",
        "Confidence Intervals Reported",
        "Code Available",
        "OOD",
        "Data Augmentation",
        "Pretrained Weights Available",
    ]

    for col in columns:
        if col not in df.columns:
            continue

        normalized_counts = Counter()
        for value in df[col].tolist():
            normalized = normalize_boolean_like(value, allow_partial=True)
            normalized_counts[normalized or "Unknown"] += 1

        ordered_labels = ["Yes", "No", "Partial", "Unknown"]
        formatted_counts = ", ".join(
            [
                f"{label}={normalized_counts.get(label, 0)}"
                for label in ordered_labels
                if normalized_counts.get(label, 0) > 0
            ]
        )
        logger.info("%s distribution: %s", col, formatted_counts)


def _sanitize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only fully empty rows to preserve all included studies."""
    initial_count = len(df)

    # Remove rows that are entirely empty.
    non_empty_mask = ~df.isna().all(axis=1)
    df = df.loc[non_empty_mask].copy()

    dropped = initial_count - len(df)
    if dropped > 0:
        logger.info("Dropped %d fully empty rows before validation.", dropped)

    return df


def _clean_reason_series(series: pd.Series) -> pd.Series:
    """Normalize rejection-reason text: trim and convert empty strings to NaN."""
    series_obj = series.astype("object")
    return (
        series_obj.where(~series_obj.isna(), pd.NA)
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def _format_count_ratio(count: int, total: int) -> str:
    """Return count/total formatted with one decimal percentage."""
    if total <= 0:
        return f"{count}/{total} (0.0%)"
    return f"{count}/{total} ({count / total * 100:.1f}%)"


def _parse_target_comparisons_argument(
    raw_values: Optional[list[str]],
) -> list[list[str]]:
    """Parse --target-comparison values into disease-term lists.

    Accepted separators inside each value: '|', ',', or ' vs. '.
    Example:
        --target-comparison "AD|CN" "PD,SWEDD" "MSA-C vs. MSA-P"
    """
    if not raw_values:
        return []

    parsed_targets: list[list[str]] = []
    for raw in raw_values:
        cleaned = str(raw).strip()
        if not cleaned:
            continue

        normalized = cleaned.replace(" vs. ", "|").replace(",", "|")
        terms = [term.strip() for term in normalized.split("|") if term.strip()]
        if len(terms) >= 2:
            parsed_targets.append(terms)

    return parsed_targets


def _filter_to_included_studies(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only studies with no rejection reason at screening and eligibility."""
    screening_col = "Reason of rejection (Screening)"
    eligibility_col = "Reason of rejection (Eligibility)"

    if screening_col not in df.columns or eligibility_col not in df.columns:
        logger.info(
            "Inclusion filter skipped: rejection columns not found in input CSV."
        )
        return df

    screening_reasons_clean = _clean_reason_series(df[screening_col])
    eligibility_reasons_clean = _clean_reason_series(df[eligibility_col])

    included_mask = screening_reasons_clean.isna() & eligibility_reasons_clean.isna()
    dropped_count = int((~included_mask).sum())

    if dropped_count > 0:
        logger.info(
            "Excluded %d rows from final analysis because they were rejected "
            "at screening and/or eligibility.",
            dropped_count,
        )

    return df.loc[included_mask].copy()


def log_problematic_rows(df: pd.DataFrame, max_rows: int = 5) -> None:
    """
    Print rows that are likely problematic for downstream analysis.

    Current checks:
    - Missing/invalid Year
    - Missing ML / DL value
    """
    required_cols = ["Year", "ML / DL"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.info(
            "Problematic-row report skipped: missing columns %s",
            ", ".join(missing_cols),
        )
        return

    year_missing_mask = df["Year"].isna()
    ml_missing_mask = df["ML / DL"].isna() | (
        df["ML / DL"].astype(str).str.strip() == ""
    )
    problematic_mask = year_missing_mask | ml_missing_mask

    problematic_df = df.loc[problematic_mask]
    if problematic_df.empty:
        logger.info("No problematic rows detected (Year / ML-DL checks).")
        return

    logger.info(
        "Problematic rows detected: %d (showing up to %d)",
        len(problematic_df),
        max_rows,
    )

    title_col = "Title" if "Title" in df.columns else None
    diseases_col = "Diseases" if "Diseases" in df.columns else None

    for idx, row in problematic_df.head(max_rows).iterrows():
        row_number = idx + 2  # +2 because CSV has header and DataFrame index is 0-based
        year_val = row.get("Year", pd.NA)
        ml_val = row.get("ML / DL", pd.NA)
        title_val = row.get(title_col, "") if title_col else ""
        diseases_val = row.get(diseases_col, "") if diseases_col else ""

        logger.info(
            "  CSV line ~%d | Year=%s | ML / DL=%s | Title=%s | Diseases=%s",
            row_number,
            year_val,
            ml_val,
            title_val,
            diseases_val,
        )


def analyze_number_of_diseases_summary(data: list[dict[str, Any]]) -> dict[str, int]:
    """
    Build the PRISMA-friendly summary by number of diseases/classes in each paper.

    Returns a dict with keys:
    - '3-classes'
    - '4-classes'
    - '5+ classes'
    - '1-2 classes'
    - 'missing diseases'
    - 'categorized total' (3/4/5+ only)
    - 'all studies total'
    """
    disease_counts: dict[int, int] = {}
    missing_diseases_count = 0

    for row in data:
        diseases_val = row.get("Diseases")
        if pd.isna(diseases_val) or str(diseases_val).strip() == "":
            missing_diseases_count += 1
            continue
        num_diseases = len(
            [d.strip() for d in str(diseases_val).split(",") if d.strip()]
        )
        disease_counts[num_diseases] = disease_counts.get(num_diseases, 0) + 1

    three_classes = disease_counts.get(3, 0)
    four_classes = disease_counts.get(4, 0)
    five_plus_classes = sum(v for k, v in disease_counts.items() if k >= 5)
    one_two_classes = sum(v for k, v in disease_counts.items() if 1 <= k <= 2)
    categorized_total = three_classes + four_classes + five_plus_classes

    summary = {
        "3-classes": three_classes,
        "4-classes": four_classes,
        "5+ classes": five_plus_classes,
        "1-2 classes": one_two_classes,
        "missing diseases": missing_diseases_count,
        "categorized total": categorized_total,
        "all studies total": len(data),
    }
    return summary


def validate_data(data: list[dict[str, Any]]) -> bool:
    """
    Validate the input data structure to ensure it's usable.

    Args:
        data: List of dictionaries containing review data.

    Returns:
        True if data is valid, False otherwise.
    """
    if not isinstance(data, list):
        logger.error("Data must be a list of dictionaries.")
        return False

    if not data:
        logger.warning("Input data is empty.")
        return False

    records_with_required_fields = 0
    for item in data:
        if isinstance(item, dict) and all(key in item for key in REQUIRED_FIELDS):
            records_with_required_fields += 1

    if records_with_required_fields == 0:
        logger.error(
            "Data validation failed. No records contain the required keys: %s",
            REQUIRED_FIELDS,
        )
        return False

    if records_with_required_fields < len(data):
        logger.warning(
            "Some rows are missing required fields; they may be ignored by downstream analyses."
        )

    year_missing = sum(1 for row in data if pd.isna(row.get("Year")))
    if year_missing:
        logger.warning("Missing or invalid 'Year' in %d rows.", year_missing)

    method_missing = sum(
        1
        for row in data
        if pd.isna(row.get("ML / DL")) or str(row.get("ML / DL", "")).strip() == ""
    )
    if method_missing:
        logger.warning("Missing 'ML / DL' in %d rows.", method_missing)

    return True


def _write_normalization_issues_report(
    issues: list[dict[str, Any]], output_path: str
) -> None:
    """Write strict-normalization issues for manual paper-level review."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Strict Normalization Issues\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"Total issues: {len(issues)}\n\n")

        by_type = Counter(issue["issue_type"] for issue in issues)
        handle.write("By issue type:\n")
        for issue_type, count in sorted(by_type.items()):
            handle.write(f"- {issue_type}: {count}\n")
        handle.write("\n")

        handle.write("Details:\n")
        handle.write("-" * 80 + "\n")
        for issue in issues:
            handle.write(
                "row={row}\tcolumn={column}\tterm={term}\tissue={issue_type}\t"
                "reason={reason}\tnormalized={normalized}\n".format(
                    row=issue.get("row"),
                    column=issue.get("column"),
                    term=issue.get("term"),
                    issue_type=issue.get("issue_type"),
                    reason=issue.get("reason"),
                    normalized=issue.get("normalized"),
                )
            )


def enforce_strict_normalization(
    data: list[dict[str, Any]], output_folder: str
) -> None:
    """Fail fast when unmapped normalization terms are detected."""
    issues = collect_normalization_issues(
        data,
        include_datasets=True,
        include_modalities=True,
        include_metrics=True,
    )
    if not issues:
        logger.info("Strict normalization check passed with no issues.")
        return

    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, "strict_normalization_issues.txt")
    _write_normalization_issues_report(issues, report_path)

    unmapped_count = sum(1 for issue in issues if issue["issue_type"] == "unmapped")

    raise ValueError(
        "Strict normalization failed: "
        f"{unmapped_count} unmapped terms found. "
        f"See {report_path} for manual review."
    )


def run_analysis_and_plotting(
    data: list[dict[str, Any]],
    output_folder: str,
    show_titles: bool,
    output_format: Any,
    target_comparisons: Optional[list[list[str]]] = None,
) -> None:
    """
    Orchestrates the full analysis and plotting workflow.

    This function calls the analyzer module to process raw data and then
    passes the results to the plotting module for visualization.

    Args:
        data: The raw data from the CSV file.
        output_folder: The directory to save plots.
        show_titles: A boolean to show titles on plots.
        output_format: The file format for the plots (e.g., 'pdf') or a list
            of formats (e.g., ['pdf', 'png']).
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Starting data analysis and plotting...")
    total_studies = len(data)
    min_dataset_frequency = ANALYSIS_CONFIG["min_dataset_frequency"]
    min_disease_frequency = ANALYSIS_CONFIG["min_disease_frequency"]
    network_min_node_frequency = ANALYSIS_CONFIG.get(
        "network_min_node_frequency",
        ANALYSIS_CONFIG.get("network_min_disease_frequency", min_disease_frequency),
    )
    network_min_edge_weight = ANALYSIS_CONFIG.get("network_min_edge_weight", 2)
    min_metric_frequency = ANALYSIS_CONFIG["min_metric_frequency"]
    comparison_small_slice_threshold_pct = ANALYSIS_CONFIG[
        "comparison_small_slice_threshold_pct"
    ]
    min_studies_per_comparison = ANALYSIS_CONFIG["min_studies_per_comparison"]
    max_performance_plots = ANALYSIS_CONFIG["max_performance_plots"]
    requested_target_comparisons = target_comparisons or []
    ml_dl_by_year = analyzer.analyze_ml_dl_trends(data)

    year_counts = analyzer.analyze_year_distribution(data)
    if year_counts:
        plotting.create_line_plot(
            data_dict=year_counts,
            title="Distribution of publications by year",
            xlabel="Year",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "year_distribution"),
            show_titles=show_titles,
            output_format=output_format,
        )

    if ml_dl_by_year:
        plotting.create_ml_dl_stacked_area_plot(
            data_dict=ml_dl_by_year,
            title="Publications over years by ML/DL category",
            xlabel="Year",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "year_distribution_ml_dl"),
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.12,
        )

    diagnosis_counts = analyzer.analyze_diseases(data)
    # Filter by minimum frequency after counting
    diagnosis_counts = Counter(
        {
            disease: count
            for disease, count in diagnosis_counts.items()
            if count >= min_disease_frequency
        }
    )
    if diagnosis_counts:
        plotting.create_lollipop_plot(
            data_dict=diagnosis_counts,
            title=(f"Diagnosis in publications (≥{min_disease_frequency} occurrences)"),
            xlabel="Diagnosis",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "disease_focus"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.12,
        )

    many_disease_studies = analyzer.analyze_studies_with_many_diseases(
        data, min_diseases=7
    )
    logger.info(
        "Studies with >6 normalized disease classes: %d/%d (%.1f%%)",
        many_disease_studies,
        total_studies,
        (many_disease_studies / total_studies * 100.0) if total_studies else 0.0,
    )

    ml_dl_counts = analyzer.analyze_ml_dl_methods(data)
    if ml_dl_counts:
        ml_count = ml_dl_counts.get("ML", 0)
        dl_count = ml_dl_counts.get("DL", 0)
        hybrid_count = ml_dl_counts.get("Hybrid (ML+DL)", 0)

        logger.info("ML/DL Methods Distribution:")
        logger.info("  ML only: %s", _format_count_ratio(ml_count, total_studies))
        logger.info("  DL only: %s", _format_count_ratio(dl_count, total_studies))
        logger.info(
            "  Hybrid (ML+DL): %s",
            _format_count_ratio(hybrid_count, total_studies),
        )

        disease_summary = analyze_number_of_diseases_summary(data)
        logger.info("Number of diseases/classes per study:")
        logger.info("  3-classes: %d", disease_summary["3-classes"])
        logger.info("  4-classes: %d", disease_summary["4-classes"])
        logger.info("  5+ classes: %d", disease_summary["5+ classes"])
        logger.info(
            "  categorized total (3/4/5+): %d/%d",
            disease_summary["categorized total"],
            disease_summary["all studies total"],
        )
        if disease_summary["categorized total"] != disease_summary["all studies total"]:
            logger.info(
                "  remaining studies: %d (1-2 classes: %d, missing diseases: %d)",
                disease_summary["all studies total"]
                - disease_summary["categorized total"],
                disease_summary["1-2 classes"],
                disease_summary["missing diseases"],
            )

        plotting.create_pie_chart(
            data_dict=ml_dl_counts,
            title="Distribution of ML/DL methods",
            output_path=os.path.join(output_folder, "ml_dl_methods"),
            top_n=None,
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.75,
            radius=0.82,
        )

    architecture_counts = analyzer.analyze_architectures(data)
    if architecture_counts:
        plotting.create_lollipop_plot(
            data_dict=architecture_counts,
            title="Architectures used",
            xlabel="Architecture",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "architectures"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
        )

        svm_count = architecture_counts.get("SVM", 0)
        rf_count = architecture_counts.get("Random Forest", 0)
        lda_family_count = architecture_counts.get("Linear Methods", 0)
        logger.info(
            "Architecture usage (normalized): SVM=%d, Random Forest=%d, Linear Methods (incl. LDA)=%d",
            svm_count,
            rf_count,
            lda_family_count,
        )

    modality_counts = analyzer.analyze_modalities(data)
    if modality_counts:
        plotting.create_lollipop_plot(
            data_dict=modality_counts,
            title="Modalities used",
            xlabel="Modality",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "modalities"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
        )

        modality_overlap = analyzer.analyze_modality_overlap_sets(data, top_n=6)
        top_modalities = modality_overlap.get("top_modalities", [])
        study_sets = modality_overlap.get("study_sets", [])
        if study_sets and top_modalities:
            plotting.create_modality_overlap_diagram(
                modality_sets=study_sets,
                modality_labels=top_modalities,
                output_path=os.path.join(output_folder, "modalities_overlap"),
                show_titles=show_titles,
                output_format=output_format,
                title="Top modality overlap",
            )

        pet_count = modality_counts.get("PET", 0)
        dti_dwi_count = modality_counts.get("DTI/DWI", 0)
        logger.info(
            "Target modality usage: PET=%d/%d (%.1f%%), DTI/DWI=%d/%d (%.1f%%)",
            pet_count,
            total_studies,
            (pet_count / total_studies * 100.0) if total_studies else 0.0,
            dti_dwi_count,
            total_studies,
            (dti_dwi_count / total_studies * 100.0) if total_studies else 0.0,
        )

    neuropath_data = analyzer.analyze_neuropathological_data(data)
    if neuropath_data:
        neuropath_yes = neuropath_data.get("Yes", 0)
        logger.info(
            "Neuropathological data: %s",
            _format_count_ratio(neuropath_yes, total_studies),
        )

    ood_validation = analyzer.analyze_ood_validation(data)
    if ood_validation:
        ood_yes = ood_validation.get("Yes", 0)
        logger.info(
            "OOD validation: %s",
            _format_count_ratio(ood_yes, total_studies),
        )

    inhouse_usage = analyzer.analyze_inhouse_dataset_usage(data)
    if inhouse_usage:
        inhouse_yes = inhouse_usage.get("Yes", 0)
        logger.info(
            "In-house dataset usage: %s",
            _format_count_ratio(inhouse_yes, total_studies),
        )

    data_augmentation_yes = 0
    for row in data:
        if normalize_boolean_like(row.get("Data Augmentation")) == "Yes":
            data_augmentation_yes += 1
    logger.info(
        "Data augmentation: %s",
        _format_count_ratio(data_augmentation_yes, total_studies),
    )

    multi_center_counts = analyzer.analyze_multi_center(data)
    if multi_center_counts:
        multi_center_yes = multi_center_counts.get("Multi-center", 0)
        single_center_no = multi_center_counts.get("Single-center", 0)
        logger.info(
            "Centers: %d/%d multi-center (%.1f%%), %d/%d single-center (%.1f%%)",
            multi_center_yes,
            total_studies,
            (multi_center_yes / total_studies * 100.0) if total_studies else 0.0,
            single_center_no,
            total_studies,
            (single_center_no / total_studies * 100.0) if total_studies else 0.0,
        )

    multi_center_by_year = analyzer.analyze_multi_center_trends(data)
    if multi_center_by_year:
        plotting.create_stacked_area_plot_percentage(
            data_dict=multi_center_by_year,
            title="Multi-center trends over time",
            xlabel="Year",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "multi_center_trends"),
            show_titles=show_titles,
            output_format=output_format,
            color_palette=None,
            normalize_to_percentage=False,
        )

    ci_reporting = analyzer.analyze_ci_reporting_status(data)
    if ci_reporting:
        ci_yes = ci_reporting.get("Yes", 0)
        logger.info(
            "Report CI: %d/%d Yes (%.1f%%)",
            ci_yes,
            total_studies,
            (ci_yes / total_studies * 100.0) if total_studies else 0.0,
        )

    ci_consistency = analyzer.analyze_ci_consistency(data)
    if ci_consistency:
        plotting.create_lollipop_plot(
            data_dict=ci_consistency,
            title="CI reporting consistency (column vs. inline ±)",
            xlabel="Consistency bucket",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "ci_reporting_consistency"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
            font_scale=0.95,
        )

    ci_widths = analyzer.analyze_ci_width_by_metric(data)
    if ci_widths:
        plotting.create_ci_width_boxplot(
            ci_widths_by_metric=ci_widths,
            output_path=os.path.join(output_folder, "ci_widths_by_metric"),
            show_titles=show_titles,
            output_format=output_format,
        )

    probast = analyzer.analyze_probast_risk(data)
    if probast.get("overall"):
        high_risk = probast["overall"].get("High", 0)
        low_risk = probast["overall"].get("Low", 0)
        unclear_risk = probast["overall"].get("Unclear", 0) + probast["overall"].get(
            "Unknown", 0
        )
        logger.info(
            "Probast: %d high, %d low, %d unclear",
            high_risk,
            low_risk,
            unclear_risk,
        )
    if probast.get("domains"):
        plotting.create_probast_domain_dot_grid_plot(
            data_by_group=probast["domains"],
            output_path=os.path.join(output_folder, "probast_domains"),
            show_titles=show_titles,
            output_format=output_format,
            title="PROBAST domain-level risk",
            total_studies=total_studies,
        )

    ood_rigor = analyzer.analyze_ood_rigor(data)
    if ood_rigor.get("categories"):
        plotting.create_lollipop_plot(
            data_dict=ood_rigor["categories"],
            title="OOD rigor categories",
            xlabel="OOD rigor category",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "ood_rigor_categories"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
            font_scale=0.98,
        )
    if ood_rigor.get("yearly"):
        plotting.create_stacked_area_plot_percentage(
            data_dict=ood_rigor["yearly"],
            title="OOD rigor trends over time",
            xlabel="Year",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "ood_rigor_trends"),
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.02,
            color_palette=None,
            normalize_to_percentage=False,
        )

    code_yes = 0
    pretrained_yes = 0
    for row in data:
        code_value = row.get("Code")
        has_code_entry = pd.notna(code_value) and str(code_value).strip() not in {
            "",
            "/",
        }
        if normalize_boolean_like(row.get("Code Available")) == "Yes" or has_code_entry:
            code_yes += 1

        if normalize_boolean_like(row.get("Pretrained Weights Available")) == "Yes":
            pretrained_yes += 1

    logger.info(
        "Code available: %d/%d (%.1f%%)",
        code_yes,
        total_studies,
        (code_yes / total_studies * 100.0) if total_studies else 0.0,
    )
    logger.info(
        "Pretrained weights: %d/%d (%.1f%%)",
        pretrained_yes,
        total_studies,
        (pretrained_yes / total_studies * 100.0) if total_studies else 0.0,
    )

    reproducibility_matrix = analyzer.analyze_reproducibility_matrix(data)
    if reproducibility_matrix:
        plotting.create_reproducibility_matrix_heatmap(
            matrix_data=reproducibility_matrix,
            output_path=os.path.join(output_folder, "reproducibility_matrix"),
            show_titles=show_titles,
            output_format=output_format,
            title="Reproducibility matrix (Code × Data availability)",
        )

    interpretability_usage = analyzer.analyze_interpretability_usage(data)
    if interpretability_usage:
        plotting.create_lollipop_plot(
            data_dict=interpretability_usage,
            title="Interpretability / xAI usage",
            xlabel="Method category",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "interpretability_usage"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
        )

    method_radar_profile = analyzer.analyze_method_radar_profile(data)
    if method_radar_profile.get("profiles"):
        plotting.create_radar_plot(
            profiles_by_method=method_radar_profile["profiles"],
            axes_labels=method_radar_profile["axes"],
            output_path=os.path.join(output_folder, "method_radar_profile"),
            show_titles=show_titles,
            output_format=output_format,
            title="Study readiness profile by method family",
            method_counts=method_radar_profile.get("counts"),
        )

    validation_rigor = analyzer.analyze_validation_rigor_by_method(data)
    if validation_rigor.get("profiles"):
        plotting.create_radar_plot(
            profiles_by_method=validation_rigor["profiles"],
            axes_labels=validation_rigor["axes"],
            output_path=os.path.join(output_folder, "validation_rigor_radar"),
            show_titles=show_titles,
            output_format=output_format,
            title="Validation rigor profile by method family",
            smoothing=True,
            method_counts=validation_rigor.get("counts"),
        )

    if requested_target_comparisons:
        target_comparison_counts = analyzer.analyze_target_comparison_counts(
            data, requested_target_comparisons
        )
        if target_comparison_counts:
            logger.info(
                "Target comparison counts in 'Normalized Performances' (study-level, order-invariant):"
            )
            for comparison, count in target_comparison_counts.items():
                logger.info(
                    "  %s: %d/%d (%.1f%%)",
                    comparison,
                    count,
                    total_studies,
                    (count / total_studies * 100.0) if total_studies else 0.0,
                )
        else:
            logger.warning(
                "Target comparisons were provided but none could be normalized to valid disease categories."
            )

    metrics_counts = analyzer.analyze_metrics(data)
    excluded_metric_labels = {"NPV"}
    metrics_counts = {
        k: v
        for k, v in metrics_counts.items()
        if v >= min_metric_frequency and str(k).upper() not in excluded_metric_labels
    }
    if metrics_counts:
        plotting.create_lollipop_plot(
            data_dict=metrics_counts,
            title="Metrics used",
            xlabel="Metric",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "metrics"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
        )

    if ml_dl_by_year:
        plotting.create_multiple_line_plot_percentage(
            data_dict=ml_dl_by_year,
            title="ML/DL methods trends over years (% of publications)",
            xlabel="Year",
            ylabel="Percentage of publications",
            output_path=os.path.join(output_folder, "ml_dl_trends"),
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.33,
            legend_font_scale=1.55,
        )

    gpu_availability, dl_studies_count = analyzer.analyze_gpu_usage(data)
    if gpu_availability:
        gpu_yes = gpu_availability.get("Yes", 0)
        logger.info(
            "GPU availability (DL studies only): %s",
            _format_count_ratio(gpu_yes, dl_studies_count),
        )

    dataset_counts = analyzer.analyze_datasets(
        data, min_frequency=min_dataset_frequency
    )
    if dataset_counts:
        plotting.create_lollipop_plot(
            data_dict=dataset_counts,
            title=(f"Top datasets used (≥{min_dataset_frequency} occurrences)"),
            xlabel="Dataset",
            ylabel="Number of publications",
            output_path=os.path.join(output_folder, "datasets"),
            top_n=None,
            horizontal=True,
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.12,
        )

    disease_modality_freq = analyzer.analyze_diseases_vs_modalities(data)
    if disease_modality_freq:
        plotting.create_diseases_vs_modalities_plot(
            frequencies=disease_modality_freq,
            output_path=os.path.join(output_folder, "diseases_vs_modalities"),
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.12,
        )

    differential_diagnosis_data = analyzer.analyze_differential_diagnosis_patterns(
        data,
        min_disease_frequency=min_disease_frequency,
        preserve_ftd_subtypes=False,
    )
    differential_diagnosis_data_network = (
        analyzer.analyze_differential_diagnosis_patterns(
            data,
            min_disease_frequency=1,
            preserve_ftd_subtypes=False,
        )
    )
    differential_diagnosis_data_subtypes = (
        analyzer.analyze_differential_diagnosis_patterns(
            data,
            min_disease_frequency=min_disease_frequency,
            preserve_ftd_subtypes=True,
        )
    )

    if differential_diagnosis_data["pairwise_comparisons"]:
        plotting.create_differential_diagnosis_network(
            pairwise_comparisons=differential_diagnosis_data_network[
                "pairwise_comparisons"
            ],
            output_path=os.path.join(output_folder, "differential_diagnosis_network"),
            show_titles=show_titles,
            output_format=output_format,
            min_node_frequency=network_min_node_frequency,
            min_edge_weight=network_min_edge_weight,
        )

        if differential_diagnosis_data_subtypes["comparison_types"]:
            plotting.create_comparison_types_plot(
                comparison_types=differential_diagnosis_data_subtypes[
                    "comparison_types"
                ],
                output_path=os.path.join(output_folder, "comparison_types"),
                show_titles=show_titles,
                output_format=output_format,
                exclude_cn_comparisons=False,
                top_n=20,
                small_slice_threshold_pct=comparison_small_slice_threshold_pct,
                dark_for_high_values=False,
                use_pastel_gradient=True,
                font_scale=1.65,
                legend_anchor_x=0.82,
                pie_radius=0.95,
                compact_layout=True,
            )

            plotting.create_disease_comparison_heatmap(
                pairwise_comparisons=differential_diagnosis_data_network[
                    "pairwise_comparisons"
                ],
                output_path=os.path.join(output_folder, "disease_comparison_heatmap"),
                show_titles=show_titles,
                output_format=output_format,
                title="Disease comparison adjacency heatmap",
            )

        complexity_points = analyzer.analyze_accuracy_vs_complexity(data)
        if complexity_points:
            plotting.create_accuracy_vs_complexity_scatter(
                study_points=complexity_points,
                output_path=os.path.join(output_folder, "accuracy_vs_complexity"),
                show_titles=show_titles,
                output_format=output_format,
                title="Reported performance vs. diagnostic complexity",
            )

    global_disease_counts = analyzer.analyze_global_disease_repartition(
        data, min_frequency=min_disease_frequency
    )
    if global_disease_counts:
        plotting.create_lollipop_plot_with_totals(
            data_dict=global_disease_counts,
            title=(
                "Total number of subjects per disease across all studies "
                f"(≥{min_disease_frequency} studies)"
            ),
            xlabel="Disease",
            ylabel="Total subjects",
            output_path=os.path.join(output_folder, "global_disease_distributions"),
            top_n=None,
            horizontal=False,
            show_titles=show_titles,
            output_format=output_format,
        )

    sizes_by_method, _years_by_method = analyzer.analyze_dataset_size_by_method(data)
    if sizes_by_method:
        plotting.create_violin_plot(
            data_dict=sizes_by_method,
            title="Distribution of dataset sizes by AI method",
            xlabel="AI method",
            ylabel="Number of subjects (log scale)",
            output_path=os.path.join(output_folder, "dataset_size_by_method"),
            show_titles=show_titles,
            output_format=output_format,
            font_scale=1.12,
            height_scale=0.86,
            y_top_multiplier=2.15,
        )

    flow_data = analyzer.analyze_disease_dataset_architecture_flow(
        data, min_disease_frequency=min_disease_frequency
    )
    if flow_data:
        plotting.create_sankey_diagram(
            flow_data=flow_data,
            output_path=os.path.join(output_folder, "sankey_diagram"),
            show_titles=show_titles,
            output_format=output_format,
            min_dataset_frequency=min_dataset_frequency,
            min_disease_frequency=min_disease_frequency,
        )

    performance_time_data = analyzer.analyze_performance_over_time(
        data,
        preserve_ftd_subtypes=True,
    )
    if performance_time_data:
        performance_plots_folder = os.path.join(output_folder, "performance_over_time")
        os.makedirs(performance_plots_folder, exist_ok=True)

        forest_plots_folder = os.path.join(output_folder, "performance_forest")
        os.makedirs(forest_plots_folder, exist_ok=True)
        logger.info(
            "Creating forest plots for common differential comparisons (>=5 studies, ID)..."
        )
        plotting.create_forest_plots_for_common_comparisons(
            performance_data=performance_time_data,
            output_folder=forest_plots_folder,
            show_titles=show_titles,
            output_format=output_format,
            min_studies_per_comparison=5,
            include_ood=False,
            include_single_class=False,
            excluded_metrics=["NPV"],
            font_scale=1.15,
        )

        target_metrics = [
            "ACC",
            "SENSITIVITY",
            "SPECIFICITY",
            "BACC",
            "AUC",
            "F1",
            "PRECISION",
            "MCC",
        ]

        for domain in ["ID", "OOD"]:
            domain_folder = os.path.join(
                performance_plots_folder,
                domain.lower(),
            )
            os.makedirs(domain_folder, exist_ok=True)

            for existing_name in os.listdir(domain_folder):
                if existing_name.startswith("NPV_"):
                    os.remove(os.path.join(domain_folder, existing_name))

            for metric in target_metrics:
                logger.info(
                    "Creating performance plots for %s (%s)...",
                    metric,
                    domain,
                )
                plotting.create_performance_over_time_plots(
                    performance_data=performance_time_data,
                    output_folder=domain_folder,
                    show_titles=show_titles,
                    output_format=output_format,
                    target_metric=metric,
                    domain_filter=domain,
                    min_studies_per_comparison=min_studies_per_comparison,
                    max_plots=max_performance_plots,
                    label_points_with_authors=False,
                    font_scale=1.62,
                )

    logger.info("All plots have been saved to: %s", output_folder)
    if not show_titles:
        logger.info("Plots were generated without titles for paper integration.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze and visualize review data.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_folder", help="Path to the output folder for plots")
    title_group = parser.add_mutually_exclusive_group()
    title_group.add_argument(
        "--title",
        action="store_true",
        help="Enable titles inside figures (default: titles are disabled; use captions instead)",
    )
    title_group.add_argument(
        "--no-titles",
        action="store_true",
        help="Disable titles inside figures (kept for backward-compat; default behavior)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=OUTPUT_FORMATS,
        default=["pdf"],
        help="Output format(s), e.g. --format pdf png (default: pdf)",
    )
    parser.add_argument(
        "--target-comparison",
        nargs="+",
        default=None,
        help=(
            "Target disease comparisons to count in normalized performances. "
            "Each value can use '|', ',' or ' vs. ' separators. "
            "Example: --target-comparison 'AD|CN' 'PD,SWEDD' 'MSA-C vs. MSA-P'"
        ),
    )
    args = parser.parse_args()

    selected_output_formats = args.format
    selected_target_comparisons = _parse_target_comparisons_argument(
        args.target_comparison
    )

    try:
        logger.info("Reading data from %s", args.input_csv)
        parsed_data_raw = pd.read_csv(args.input_csv, encoding="utf-8", dtype=str)
        _log_final_output_schema(parsed_data_raw)

        parsed_data = parsed_data_raw.copy()
        parsed_data = _filter_to_included_studies(parsed_data)
        parsed_data["Year"] = pd.to_numeric(parsed_data["Year"], errors="coerce")
        parsed_data = _sanitize_input_dataframe(parsed_data)

        input_dir = os.path.dirname(args.input_csv)
        input_filename = os.path.basename(args.input_csv)
        filename_without_ext = os.path.splitext(input_filename)[0]
        filtered_csv_path = os.path.join(
            input_dir, f"{filename_without_ext}_filtered.csv"
        )
        parsed_data.to_csv(filtered_csv_path, index=False, encoding="utf-8")
        logger.info(
            "Exported filtered data (%d studies) to: %s",
            len(parsed_data),
            filtered_csv_path,
        )

        log_problematic_rows(parsed_data)

        data = parsed_data.to_dict("records")
        logger.info("Successfully loaded %d entries.", len(data))

        if not validate_data(data):
            return 1  # Exit with error code

        enforce_strict_normalization(data, args.output_folder)
        logger.info("Strict normalization is enabled (mandatory).")

        set_scientific_style()

        run_analysis_and_plotting(
            data,
            args.output_folder,
            show_titles=bool(args.title),
            output_format=selected_output_formats,
            target_comparisons=selected_target_comparisons,
        )

    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input_csv)
        return 1
    except pd.errors.EmptyDataError:
        logger.error("Input file is empty: %s", args.input_csv)
        return 1
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        return 1

    logger.info("Processing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
