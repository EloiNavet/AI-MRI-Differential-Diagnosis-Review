#!/usr/bin/env python3
"""
CSV to LaTeX Converter
Script for converting research papers CSV to LaTeX format.

Example:
    python src/review_analysis/csv_to_latex.py \
        "Review - Script v3 output.csv" \
        review_output.tex
"""

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CSVToLatexConverter:
    """CSV to LaTeX converter with advanced formatting capabilities"""

    def __init__(
        self,
        input_file: str,
        merged_columns: Optional[list[list[str]]] = None,
        bibliography_file: Optional[str] = None,
    ):
        self.input_file = Path(input_file)
        self.df = None
        self.merged_columns = merged_columns or []
        self.bibliography_file = Path(bibliography_file) if bibliography_file else None
        self.bib_keys: set[str] = set()
        self.bibliography_entry: Optional[str] = None

        self.latex_escapes = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "^": r"\textasciicircum{}",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
        }

        self.column_aliases: dict[str, list[str]] = {
            "Title": ["Title", "Title (old)"],
            "Year": ["Year", "Year (old)"],
            "Authors": ["Authors", "Author", "Authors (old)"],
            "Bib citations": ["Bib citations", "Bib citation", "Bibcitation"],
            "Diseases": ["Diseases", "Disease"],
            "Repartition": ["Repartition"],
            "Modalities": ["Modalities", "Modelities"],
            "Datasets": ["Datasets"],
            "Neuropathological": ["Neuropathological"],
            "OOD": ["OOD", "Out of Distribution"],
            "Architecture(s) Used": ["Architecture(s) Used", "Architectures"],
            "Metrics Used": ["Metrics Used"],
            "Normalized Performances": [
                "Normalized Performances",
                "Performances",
            ],
            "Code": ["Code"],
            "GPUs": ["GPUs"],
        }

        self.header_labels = {
            "Title": "Title",
            "Year": "Year",
            "Authors": "Authors",
            "Bib citations": "Bib citations",
            "Diseases": "Diseases",
            "Repartition": "Data Split",
            "Modalities": "Modalities",
            "Datasets": "Datasets",
            "Neuropathological": "Neuropathological",
            "OOD": "Out of Distribution",
            "Architecture(s) Used": "ML Architectures",
            "Metrics Used": "Evaluation Metrics",
            "Normalized Performances": "Performance Results",
            "Code": "Code Available",
            "GPUs": "GPU Usage",
        }

        self.width_map = {
            "Title": 0.16,
            "Year": 0.026,
            "Authors": 0.067,
            "Bib citations": 0.05,
            "Diseases": 0.067,
            "Repartition": 0.13,
            "Modalities": 0.07,
            "Datasets": 0.11,
            "Neuropathological": 0.02,
            "OOD": 0.02,
            "Architecture(s) Used": 0.11,
            "Metrics Used": 0.07,
            "Normalized Performances": 0.1,
            "Code": 0.035,
            "GPUs": 0.035,
            "PROBAST": 0.06,
        }

        self.stats = {"original_rows": 0, "processing_time": 0}

        self._load_csv()
        self._validate_and_clean_data()

    def set_bibliography_file(self, bibliography_file: Optional[str]) -> None:
        """Set bibliography file and preload citation keys for validation."""
        if not bibliography_file:
            self.bibliography_file = None
            self.bib_keys = set()
            self.bibliography_entry = None
            return

        self.bibliography_file = Path(bibliography_file)
        self.bib_keys = self._load_bib_keys(self.bibliography_file)

    def set_bibliography_entry(self, bibliography_entry: Optional[str]) -> None:
        """Set \bibliography{...} entry path (without extension)."""
        self.bibliography_entry = bibliography_entry

    def _load_bib_keys(self, bibliography_path: Path) -> set[str]:
        """Extract BibTeX keys from a .bib/.biblatex file."""
        if not bibliography_path.exists():
            logger.warning("Bibliography file not found: %s", bibliography_path)
            return set()

        try:
            content = bibliography_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = bibliography_path.read_text(encoding="latin-1")

        keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", content))
        logger.info("Loaded %d bibliography keys from %s", len(keys), bibliography_path)
        return keys

    def _parse_citation_keys(self, raw_value: str) -> list[str]:
        """Parse, normalize, and validate citation keys from CSV value."""
        if not isinstance(raw_value, str):
            raw_value = str(raw_value)

        value = raw_value.strip()
        if not value or value.lower() in {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
            "/",
            "-",
            "---",
        }:
            return []

        if value.startswith("\\cite{") and value.endswith("}"):
            value = value[6:-1]

        candidate_keys: list[str] = []
        for part in re.split(r"[;,|]", value):
            key = part.strip()
            if key:
                candidate_keys.append(key)

        deduped_keys: list[str] = []
        seen = set()
        for key in candidate_keys:
            if key not in seen:
                deduped_keys.append(key)
                seen.add(key)

        if not self.bib_keys:
            return deduped_keys

        valid_keys = [key for key in deduped_keys if key in self.bib_keys]
        if deduped_keys and not valid_keys:
            logger.warning(
                "No bibliography keys matched for citation value: '%s'", raw_value
            )
        return valid_keys

    def _load_csv(self) -> None:
        """Load and validate CSV file"""
        start_time = time.time()

        try:
            self.df = pd.read_csv(
                self.input_file,
                encoding="utf-8",
                dtype=str,
                skipinitialspace=True,
                on_bad_lines="warn",
                quoting=csv.QUOTE_MINIMAL,
            )
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, retrying with latin-1")
            self.df = pd.read_csv(
                self.input_file,
                encoding="latin-1",
                dtype=str,
                skipinitialspace=True,
                on_bad_lines="warn",
                quoting=csv.QUOTE_MINIMAL,
            )

        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.fillna("")
        self.stats["original_rows"] = len(self.df)
        self.stats["processing_time"] = time.time() - start_time
        logger.info(
            "Loaded CSV with %d rows and %d columns",
            len(self.df),
            len(self.df.columns),
        )

    def _resolve_column_name(self, preferred_name: str) -> Optional[str]:
        """Resolve a requested column name against aliases in a case-insensitive way."""
        if self.df is None:
            return None

        if preferred_name in self.df.columns:
            return preferred_name

        candidates = self.column_aliases.get(preferred_name, [preferred_name])
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate

        lower_to_original = {col.lower(): col for col in self.df.columns}
        for candidate in [preferred_name] + candidates:
            resolved = lower_to_original.get(candidate.lower())
            if resolved:
                return resolved

        return None

    def _canonical_column_name(self, column_name: str) -> str:
        """Return canonical column name for formatting and width rules."""
        for canonical, aliases in self.column_aliases.items():
            if column_name == canonical or column_name in aliases:
                return canonical
        return column_name

    def resolve_columns(
        self, requested_columns: list[str]
    ) -> tuple[list[str], list[str]]:
        """Resolve requested columns to existing CSV columns.

        Returns:
            resolved_columns: Existing columns to use.
            missing_columns: Requested columns not found.
        """
        resolved_columns: list[str] = []
        missing_columns: list[str] = []

        for col in requested_columns:
            actual = self._resolve_column_name(col)
            if actual is None:
                missing_columns.append(col)
                continue

            if actual not in resolved_columns:
                resolved_columns.append(actual)

        return resolved_columns, missing_columns

    def _validate_and_clean_data(self) -> None:
        """Clean and filter data to remove invalid entries"""
        if self.df is None:
            return

        before_non_empty = len(self.df)
        non_empty_mask = ~self.df.isnull().all(axis=1) & ~(self.df == "").all(axis=1)
        self.df = self.df[non_empty_mask].reset_index(drop=True)
        logger.info("Dropped %d empty rows", before_non_empty - len(self.df))

        title_col = self._resolve_column_name("Title")
        if title_col:
            before_title_filter = len(self.df)
            title_series = self.df[title_col].astype(str)
            title_mask = (
                title_series.str.strip().ne("")
                & ~title_series.str.lower().isin(["nan", "n/a", "null", "none"])
                & (title_series.str.len() >= 5)
            )
            self.df = self.df[title_mask].reset_index(drop=True)
            logger.info(
                "Dropped %d rows with invalid titles using column '%s'",
                before_title_filter - len(self.df),
                title_col,
            )

    def _escape_latex(self, text: str) -> str:
        """Escape LaTeX-reserved characters in plain text."""
        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return ""

        # Detect URLs and mark them for special handling
        url_pattern = r"https?://[^\s]+|ftp://[^\s]+"
        urls = list(set(re.findall(url_pattern, text)))
        url_tokens = {f"@@URL{i}@@": url for i, url in enumerate(urls)}
        for token, url in url_tokens.items():
            text = text.replace(url, token)

        for char, escape in self.latex_escapes.items():
            text = text.replace(char, escape)

        text = text.replace("<", r"\textless{}")
        text = text.replace(">", r"\textgreater{}")
        text = text.replace("|", r"\textbar{}")

        # Re-substitute URLs with \url{} wrapper for proper line breaking
        for token, url in url_tokens.items():
            text = text.replace(token, f"\\url{{{url}}}")

        return text

    def _format_column_header(self, column_name: str) -> str:
        """Format column headers for LaTeX table"""
        canonical = self._canonical_column_name(column_name)
        header_text = self.header_labels.get(canonical, canonical.replace("_", " "))
        escaped_header = self._escape_latex(header_text)

        if len(escaped_header) > 20 and " " in escaped_header:
            words = escaped_header.split()
            mid = len(words) // 2
            first_part = " ".join(words[:mid])
            second_part = " ".join(words[mid:])
            return f"\\textbf{{\\makecell{{{first_part} \\\\ {second_part}}}}}"

        return f"\\textbf{{{escaped_header}}}"

    def _truncate_authors(self, authors_str: str) -> str:
        """Truncate authors to first author + et al."""
        if not isinstance(authors_str, str) or not authors_str.strip():
            return "---"

        normalized = authors_str.strip().lower()
        placeholder_authors = {
            "not specified in the provided text",
            "not specified",
            "unknown",
            "n.a.",
            "n/a",
            "---",
        }
        if normalized in placeholder_authors:
            return "---"

        authors = re.split(r"[,;]", authors_str)
        authors = [author.strip() for author in authors if author.strip()]

        if len(authors) == 0:
            return "---"
        if len(authors) == 1:
            return self._escape_latex(authors[0])

        first_author = self._escape_latex(authors[0])
        return f"{first_author} \\textit{{et al.}}"

    def _merge_cell_content(self, merged_group: list[tuple[str, str]]) -> str:
        """Merge multiple column values into a single cell.

        Args:
            merged_group: List of (column_name, cell_value) tuples to merge

        Returns:
            Formatted merged content for display in cell
        """
        parts = []
        for col_name, content in merged_group:
            cleaned = self._clean_cell_content(content, col_name)
            if cleaned != "---":
                parts.append(cleaned)

        if not parts:
            return "---"

        return " \\newline ".join(parts)

    def _format_inline_reference_group(
        self, merged_group: list[str], row: pd.Series
    ) -> str:
        """Format title/author/year/key as a single inline citation block."""
        group_map = {self._canonical_column_name(col): col for col in merged_group}

        title_col = group_map.get("Title", "")
        authors_col = group_map.get("Authors", "")
        year_col = group_map.get("Year", "")
        bib_col = group_map.get("Bib citations", "")

        title = (
            self._clean_cell_content(row.get(title_col, ""), title_col)
            if title_col
            else ""
        )
        authors = (
            self._clean_cell_content(row.get(authors_col, ""), authors_col)
            if authors_col
            else ""
        )
        year = (
            self._clean_cell_content(row.get(year_col, ""), year_col)
            if year_col
            else ""
        )
        citation_keys = (
            self._parse_citation_keys(row.get(bib_col, "")) if bib_col else []
        )

        parts = []
        if title and title != "---":
            parts.append(title)
        if authors and authors != "---":
            if parts:
                parts[-1] = f"{parts[-1]}, {authors}"
            else:
                parts.append(authors)
        if year and year != "---":
            if parts:
                parts[-1] = f"{parts[-1]} ({year})"
            else:
                parts.append(f"({year})")
        if citation_keys:
            if parts:
                parts[-1] = f"{parts[-1]}~\\cite{{{','.join(citation_keys)}}}"
            else:
                parts.append(f"\\cite{{{','.join(citation_keys)}}}")

        return parts[0] if parts else "---"

    def _clean_cell_content(self, content: str, column_name: str = "") -> str:
        if not isinstance(content, str):
            content = str(content)

        content = content.strip()
        if not content or content.lower() in {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
            "/",
            "-",
        }:
            return "---"

        canonical = self._canonical_column_name(column_name)
        if canonical in ["Author", "Authors"]:
            return self._truncate_authors(content)

        token_map = {
            # Explicitly normalize uppercase Greek alpha used in some datasets (e.g. MCIΑβ)
            "Α": "A",
            "ε": "@@EPSILON@@",
            "β": "@@BETA@@",
            "α": "@@ALPHA@@",
            "μ": "@@MU@@",
            "≈": "@@APPROX@@",
        }
        latex_map = {
            "@@EPSILON@@": "$\\epsilon$",
            "@@BETA@@": "$\\beta$",
            "@@ALPHA@@": "$\\alpha$",
            "@@MU@@": "$\\mu$",
            "@@APPROX@@": "$\\approx$",
        }

        for symbol, token in token_map.items():
            content = content.replace(symbol, token)

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            lines = [content]

        escaped_lines = [self._escape_latex(line) for line in lines]

        # For Repartition, don't add newlines - just join with space
        if canonical == "Repartition":
            merged = " ".join(escaped_lines)
        else:
            merged = "\\newline ".join(escaped_lines)

        for token, latex_cmd in latex_map.items():
            merged = merged.replace(token, latex_cmd)

        return merged

    def generate_longtable(
        self,
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = None,
        caption: str = "Systematic Review of Machine Learning Applications",
    ) -> str:
        """Generate LaTeX longtable, handling merged columns if specified"""
        if self.df is None:
            raise RuntimeError("Dataframe is empty. Did CSV loading fail?")

        if columns is None:
            columns = list(self.df.columns)

        # Filter columns based on merged_columns config
        # Collect all columns that are in merge groups
        all_merged_cols = set()
        for merge_group in self.merged_columns:
            resolved_group = []
            for col in merge_group:
                resolved = self._resolve_column_name(col)
                if resolved:
                    resolved_group.append(resolved)
                    all_merged_cols.add(resolved)

        # Build display_columns: either merged header or individual column
        display_columns = []
        for col in columns:
            # Check if this column is part of a merge group
            in_merge = False
            for merge_group_idx, merge_group in enumerate(self.merged_columns):
                resolved_group = []
                for col_spec in merge_group:
                    resolved = self._resolve_column_name(col_spec)
                    if resolved:
                        resolved_group.append(resolved)

                if col in resolved_group:
                    # Only add once per merge group (use first position)
                    if col == resolved_group[0]:
                        display_columns.append(
                            ("MERGED_GROUP", merge_group_idx, resolved_group)
                        )
                    in_merge = True
                    break

            if not in_merge:
                display_columns.append(("SINGLE", col, None))

        # Create df_subset with all needed columns
        all_needed_cols = [col for col in columns if col in self.df.columns]
        df_subset = self.df[all_needed_cols].copy()
        if max_rows is not None and max_rows > 0:
            df_subset = df_subset.head(max_rows)

        num_cols = len(display_columns)

        # Calculate widths for display columns
        display_col_widths = []
        for item in display_columns:
            if item[0] == "MERGED_GROUP":
                merge_group_idx, resolved_group = item[1], item[2]
                merged_width = sum(
                    self.width_map.get(self._canonical_column_name(col), 0.07)
                    for col in resolved_group
                )
                display_col_widths.append(merged_width)
            else:
                col = item[1]
                canonical = self._canonical_column_name(col)
                display_col_widths.append(self.width_map.get(canonical, 0.07))

        col_widths = self._normalize_column_widths(display_col_widths)
        col_spec = (
            "|" + "|".join([f"p{{{width}\\textwidth}}" for width in col_widths]) + "|"
        )

        escaped_caption = self._escape_latex(caption)
        latex_code = [
            "\\documentclass[a4paper]{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[landscape,margin=0.2in]{geometry}",
            "\\usepackage{longtable}",
            "\\usepackage{booktabs}",
            "\\usepackage{makecell}",
            "\\usepackage{xcolor}",
            "\\usepackage{colortbl}",
            "\\usepackage[hyphens]{url}",
            "\\usepackage{xurl}",
            "\\usepackage[breaklinks=true]{hyperref}",
            "\\definecolor{headercolor}{RGB}{44,62,80}",
            "\\definecolor{headertextcolor}{RGB}{248,249,250}",
            "\\definecolor{rowcolor1}{RGB}{248,249,250}",
            "\\definecolor{rowcolor2}{RGB}{255,255,255}",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\renewcommand{\\arraystretch}{1.16}",
            "\\hyphenpenalty=10000",
            "\\exhyphenpenalty=10000",
            "\\sloppy",
            "\\begin{document}",
            "\\section*{Table A}",
            "\\scriptsize",
            f"\\begin{{longtable}}{{{col_spec}}}",
            f"\\caption{{{escaped_caption}}} \\\\",
            "\\toprule",
        ]

        header_cells = []
        for item in display_columns:
            if item[0] == "MERGED_GROUP":
                merge_group_idx, resolved_group = item[1], item[2]
                canonical_group = {
                    self._canonical_column_name(col) for col in resolved_group
                }
                if canonical_group.issuperset(
                    {"Title", "Authors", "Year", "Bib citations"}
                ):
                    formatted_header = "\\textbf{Reference}"
                else:
                    merged_header = " / ".join(
                        self.header_labels.get(
                            self._canonical_column_name(col), col.replace("_", " ")
                        )
                        for col in resolved_group
                    )
                    escaped_header = self._escape_latex(merged_header)
                    formatted_header = f"\\textbf{{{escaped_header}}}"
            else:
                col = item[1]
                formatted_header = self._format_column_header(col)

            header_cells.append(
                f"\\cellcolor{{headercolor}}\\textcolor{{headertextcolor}}{{{formatted_header}}}"
            )

        header_row = " & ".join(header_cells)
        latex_code.extend([f"{header_row} \\\\", "\\midrule", "\\endfirsthead"])

        latex_code.extend(
            [
                f"\\multicolumn{{{num_cols}}}{{c}}{{\\tablename\\ \\thetable\\ -- \\textit{{Continued from previous page}}}} \\\\",
                "\\midrule",
                f"{header_row} \\\\",
                "\\midrule",
                "\\endhead",
                "\\midrule",
                f"\\multicolumn{{{num_cols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\",
                "\\endfoot",
                "\\bottomrule",
                "\\endlastfoot",
            ]
        )

        for row_idx, (_, row) in enumerate(df_subset.iterrows()):
            row_data = []
            row_color = "rowcolor1" if row_idx % 2 == 0 else "rowcolor2"

            for item in display_columns:
                if item[0] == "MERGED_GROUP":
                    merge_group_idx, resolved_group = item[1], item[2]
                    canonical_group = {
                        self._canonical_column_name(col) for col in resolved_group
                    }
                    if canonical_group.issuperset(
                        {"Title", "Authors", "Year", "Bib citations"}
                    ):
                        cell_content = self._format_inline_reference_group(
                            resolved_group, row
                        )
                    else:
                        merged_values = [
                            (col, row.get(col, "")) for col in resolved_group
                        ]
                        cell_content = self._merge_cell_content(merged_values)
                else:
                    col = item[1]
                    cell_content = self._clean_cell_content(row[col], col)

                cell_content = f"\\cellcolor{{{row_color}}}{cell_content}"
                row_data.append(cell_content)

            latex_row = " & ".join(row_data) + " \\\\"
            latex_code.append(latex_row)

        latex_code.extend(["\\end{longtable}", "\\end{document}"])

        if self.bibliography_entry:
            latex_code = latex_code[:-1] + [
                "\\bibliographystyle{naturemag}",
                f"\\bibliography{{{self.bibliography_entry}}}",
                "\\end{document}",
            ]

        return "\n".join(latex_code)

    def _normalize_column_widths(self, widths: list[float]) -> list[str]:
        """Normalize column widths to fit textwidth"""
        total_width = sum(widths)
        if total_width > 0.98:
            scale_factor = 0.98 / total_width
            widths = [w * scale_factor for w in widths]

        return [f"{w:.3f}" for w in widths]

    def save_latex(self, output_file: str, content: str) -> None:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Error saving LaTeX file: {e}") from e


DEFAULT_COLUMN_PREFERENCES = [
    "Title",
    "Year",
    "Authors",
    "Bib citations",
    "Repartition",
    "Modalities",
    "Datasets",
    "Neuropathological",
    "OOD",
    "Architecture(s) Used",
    "Code",
    "GPUs",
]


def _parse_columns_argument(raw_values: Optional[list[str]]) -> Optional[list[str]]:
    """Parse --columns values that may be space-separated and/or comma-separated."""
    if not raw_values:
        return None

    parsed: list[str] = []
    for raw in raw_values:
        parsed.extend([part.strip() for part in raw.split(",") if part.strip()])

    return parsed or None


def _parse_merge_argument(raw_values: Optional[list[str]]) -> Optional[list[list[str]]]:
    """Parse --merge values where each value is 'Col1|Col2|Col3' format.

    Returns:
        List of column groups to merge, or None if no merges specified.
    """
    if not raw_values:
        return None

    merged_groups: list[list[str]] = []
    for raw in raw_values:
        group = [part.strip() for part in raw.split("|") if part.strip()]
        if group:
            merged_groups.append(group)

    return merged_groups or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a review CSV file into a formatted LaTeX longtable."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="Review - Script v3 output.csv",
        help="Input CSV file path.",
    )
    parser.add_argument(
        "output_tex",
        nargs="?",
        default=None,
        help="Output LaTeX file path. Defaults to <input_name>.tex.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=(
            "Columns to include (space and/or comma separated). "
            "Aliases are supported, e.g. 'Modalities' or 'Modelities'."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows in output.",
    )
    parser.add_argument(
        "--caption",
        default="",
        help="Caption for the LaTeX longtable.",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help=(
            "Merge multiple columns into a single column. Format: 'Col1|Col2|Col3'. "
            "Example: --merge 'Title|Year|Authors' 'Datasets|Architecture(s) Used'"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs.",
    )
    parser.add_argument(
        "--bibliography-file",
        default="biblio.bib",
        help=(
            "Path to bibliography file used for citation validation and LaTeX "
            "\\bibliography{...} insertion. Use empty string to disable."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_path = Path(args.input_csv)
    output_path = (
        Path(args.output_tex) if args.output_tex else input_path.with_suffix(".tex")
    )

    try:
        merged_config = _parse_merge_argument(args.merge)
        converter = CSVToLatexConverter(
            str(input_path),
            merged_columns=merged_config,
            bibliography_file=args.bibliography_file,
        )

        bib_arg = (args.bibliography_file or "").strip()
        if bib_arg:
            bib_path = Path(bib_arg)
            if not bib_path.is_absolute():
                bib_path = Path.cwd() / bib_path
            converter.set_bibliography_file(str(bib_path))

            bibliography_entry = bib_path.stem

            converter.set_bibliography_entry(bibliography_entry)
        else:
            converter.set_bibliography_file(None)
            converter.set_bibliography_entry(None)

        requested_columns = (
            _parse_columns_argument(args.columns) or DEFAULT_COLUMN_PREFERENCES
        )
        resolved_columns, missing_columns = converter.resolve_columns(requested_columns)

        if missing_columns:
            logger.warning(
                "Some requested columns were not found: %s", ", ".join(missing_columns)
            )

        if not resolved_columns:
            logger.error("None of the requested columns were found in the CSV file.")
            return 1

        latex_content = converter.generate_longtable(
            columns=resolved_columns,
            max_rows=args.max_rows,
            caption=args.caption,
        )
        converter.save_latex(str(output_path), latex_content)

        logger.info("Generated LaTeX file: %s", output_path)
        logger.info("Rows in output: %d", len(converter.df))
        logger.info("Columns in output: %s", ", ".join(resolved_columns))
        return 0
    except Exception as e:
        logger.error("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
