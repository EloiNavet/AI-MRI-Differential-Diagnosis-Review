#!/usr/bin/env python3
"""Map review CSV rows to BibLaTeX citation keys and export filtered sources."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd
from rapidfuzz import fuzz


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BibEntry:
    key: str
    entry_type: str
    raw: str
    title: str
    authors: str
    year: str
    order: int
    normalized_title: str
    first_author_signature: str
    signature: str


@dataclass
class MatchResult:
    row_index: int
    title: str
    year: str
    authors: str
    status: str
    citation_key: str = ""
    title_score: float = 0.0
    author_score: float = 0.0
    combined_score: float = 0.0
    candidate_summaries: list[str] = field(default_factory=list)
    reason: str = ""


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_wrapping_delimiters(text: str) -> str:
    value = text.strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1].strip()
            if inner:
                value = inner
                changed = True
                continue
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].strip()
            changed = True
    return value


def _replace_latex_commands_with_content(text: str) -> str:
    value = str(text)
    pattern = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]+\])?\{([^{}]*)\}")

    while True:
        new_value = pattern.sub(r"\1", value)
        if new_value == value:
            return value
        value = new_value


def _expand_title_abbreviations(text: str) -> str:
    value = str(text)
    replacements = [
        (r"\bbvftd\b", "behavioral variant frontotemporal dementia"),
        (r"\bad\b", "alzheimer disease"),
        (r"\bftd\b", "frontotemporal dementia"),
        (r"\bmri\b", "magnetic resonance imaging"),
        (r"\bpet\b", "positron emission tomography"),
        (r"\bpsp\b", "progressive supranuclear palsy"),
        (r"\bmsa\b", "multiple system atrophy"),
    ]

    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

    return value


def _normalize_text(text: str) -> str:
    value = _strip_wrapping_delimiters(str(text))
    value = value.replace("~", " ")
    value = _replace_latex_commands_with_content(value)
    value = _expand_title_abbreviations(value)
    value = value.replace("{", " ").replace("}", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return _collapse_whitespace(value)


def _normalize_person_name(text: str) -> str:
    value = _strip_wrapping_delimiters(str(text))
    value = value.replace("~", " ")
    value = _replace_latex_commands_with_content(value)
    value = value.replace("{", " ").replace("}", " ")
    value = value.replace(".", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) >= 2:
            value = " ".join(parts[1:] + [parts[0]])
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return _collapse_whitespace(value)


def _extract_first_csv_author(authors: str) -> str:
    value = str(authors).strip()
    if not value:
        return ""

    for separator in [";", " and "]:
        if separator in value:
            return value.split(separator, 1)[0].strip()

    if "," in value:
        return value.split(",", 1)[0].strip()

    return value


def _extract_first_bib_author(authors: str) -> str:
    value = str(authors).strip()
    if not value:
        return ""

    first_author = re.split(r"\s+and\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return first_author.strip()


def _extract_year_from_text(value: str) -> str:
    match = re.search(r"(19|20)\d{2}", str(value))
    return match.group(0) if match else ""


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    brace_depth = 0
    in_quote = False
    escaped = False

    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue

        if char == "\\":
            buffer.append(char)
            escaped = True
            continue

        if char == '"' and brace_depth == 0:
            in_quote = not in_quote
            buffer.append(char)
            continue

        if not in_quote:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(0, brace_depth - 1)
            elif char == "," and brace_depth == 0:
                part = "".join(buffer).strip()
                if part:
                    parts.append(part)
                buffer = []
                continue

        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)

    return parts


def _extract_bib_entries(raw_text: str) -> list[tuple[str, str, str, str, int]]:
    entries: list[tuple[str, str, str, str, int]] = []
    cursor = 0
    order = 0

    while cursor < len(raw_text):
        at_index = raw_text.find("@", cursor)
        if at_index < 0:
            break

        open_index = None
        for idx in range(at_index + 1, len(raw_text)):
            if raw_text[idx] in "{(":
                open_index = idx
                break
            if raw_text[idx] == "@":
                break

        if open_index is None:
            cursor = at_index + 1
            continue

        open_char = raw_text[open_index]
        close_char = "}" if open_char == "{" else ")"
        depth = 1
        in_quote = False
        escaped = False
        end_index = None

        for idx in range(open_index + 1, len(raw_text)):
            char = raw_text[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"' and not in_quote:
                in_quote = True
                continue
            if char == '"' and in_quote:
                in_quote = False
                continue
            if in_quote:
                continue
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    end_index = idx + 1
                    break

        if end_index is None:
            raise RuntimeError(
                f"Unterminated BibLaTeX entry starting at byte {at_index}"
            )

        raw_entry = raw_text[at_index:end_index]
        entry_kind = raw_text[at_index + 1 : open_index].strip()
        body = raw_text[open_index + 1 : end_index - 1].strip()
        entries.append(
            (entry_kind, raw_entry, body, raw_text[at_index:end_index], order)
        )
        cursor = end_index
        order += 1

    return entries


def _parse_entry_body(body: str) -> tuple[str, dict[str, str]]:
    first_comma = body.find(",")
    if first_comma < 0:
        return body.strip(), {}

    key = body[:first_comma].strip()
    fields_blob = body[first_comma + 1 :].strip()
    fields: dict[str, str] = {}

    for field_entry in _split_top_level_commas(fields_blob):
        if "=" not in field_entry:
            continue
        field_name, value = field_entry.split("=", 1)
        field_name = field_name.strip().lower()
        cleaned_value = _strip_wrapping_delimiters(value.strip())
        cleaned_value = _collapse_whitespace(cleaned_value.replace("~", " "))
        fields[field_name] = cleaned_value

    return key, fields


class BibLatexIndex:
    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.entries: list[BibEntry] = []
        self.entries_by_year: dict[str, list[BibEntry]] = {}
        self.entries_without_year: list[BibEntry] = []
        self.duplicate_keys: dict[str, int] = {}
        self.conflicting_duplicates: list[str] = []

        self._load()

    def _load(self) -> None:
        raw_text = self.source_path.read_text(encoding="utf-8")
        parsed_entries = _extract_bib_entries(raw_text)

        seen_by_key: dict[str, BibEntry] = {}
        years: dict[str, list[BibEntry]] = {}

        for entry_kind, raw_entry, body, _, order in parsed_entries:
            key, fields = _parse_entry_body(body)
            if not key:
                continue

            title = fields.get("title", "")
            authors = fields.get("author", "")
            year = _extract_year_from_text(
                fields.get("year", "") or fields.get("date", "")
            )
            normalized_title = _normalize_text(title)
            first_author_signature = _normalize_person_name(
                _extract_first_bib_author(authors)
            )
            signature = f"{normalized_title}|{first_author_signature}|{year}"
            entry = BibEntry(
                key=key,
                entry_type=entry_kind,
                raw=raw_entry,
                title=title,
                authors=authors,
                year=year,
                order=order,
                normalized_title=normalized_title,
                first_author_signature=first_author_signature,
                signature=signature,
            )

            previous = seen_by_key.get(key)
            if previous is None:
                seen_by_key[key] = entry
                self.entries.append(entry)
                if year:
                    years.setdefault(year, []).append(entry)
                else:
                    self.entries_without_year.append(entry)
                continue

            self.duplicate_keys[key] = self.duplicate_keys.get(key, 1) + 1
            if previous.signature != entry.signature:
                self.conflicting_duplicates.append(key)

        self.entries_by_year = years

        if not self.entries:
            raise RuntimeError(f"No BibLaTeX entries found in {self.source_path}")

        if self.conflicting_duplicates:
            conflict_list = ", ".join(sorted(set(self.conflicting_duplicates)))
            raise RuntimeError(
                f"Conflicting duplicate BibLaTeX keys found in {self.source_path}: {conflict_list}"
            )

        if self.duplicate_keys:
            duplicate_list = ", ".join(
                f"{key} x{count}" for key, count in sorted(self.duplicate_keys.items())
            )
            logger.warning(
                "Duplicate BibLaTeX keys collapsed to first occurrence: %s",
                duplicate_list,
            )

    def candidates_for_year(self, year: str) -> list[BibEntry]:
        return self.entries_by_year.get(year, [])

    def get_entries_by_keys(self, keys: Sequence[str]) -> list[BibEntry]:
        wanted = set(keys)
        return [entry for entry in self.entries if entry.key in wanted]


class CitationMapper:
    def __init__(
        self,
        csv_path: Path,
        biblatex_path: Path,
        title_threshold: float = 90.0,
        author_threshold: float = 85.0,
        combined_threshold: float = 92.0,
        ambiguity_margin: float = 3.0,
    ):
        self.csv_path = csv_path
        self.biblatex_path = biblatex_path
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold
        self.combined_threshold = combined_threshold
        self.ambiguity_margin = ambiguity_margin
        self.bib_index = BibLatexIndex(biblatex_path)

        self.df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            dtype=str,
            skipinitialspace=True,
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
        ).fillna("")
        self.df.columns = self.df.columns.str.strip()

    def _score_entry(
        self, csv_title: str, csv_author: str, entry: BibEntry
    ) -> tuple[float, float, float]:
        title_score = (
            fuzz.WRatio(csv_title, entry.normalized_title) if csv_title else 0.0
        )
        author_score = (
            fuzz.token_sort_ratio(csv_author, entry.first_author_signature)
            if csv_author and entry.first_author_signature
            else 0.0
        )
        combined_score = (0.75 * title_score) + (0.25 * author_score)
        return float(title_score), float(author_score), float(combined_score)

    def _rank_candidates(
        self, csv_title: str, csv_author: str, candidates: Sequence[BibEntry]
    ) -> list[tuple[float, float, float, BibEntry]]:
        scored_candidates: list[tuple[float, float, float, BibEntry]] = []
        for entry in candidates:
            title_score, author_score, combined_score = self._score_entry(
                csv_title, csv_author, entry
            )
            scored_candidates.append((title_score, author_score, combined_score, entry))

        scored_candidates.sort(
            key=lambda item: (item[2], item[0], item[1], -item[3].order),
            reverse=True,
        )
        return scored_candidates

    def _row_context(self, row: pd.Series) -> tuple[str, str, str]:
        title = str(row.get("Title", "")).strip()
        year = _extract_year_from_text(str(row.get("Year", "")))
        authors = str(row.get("Authors", "")).strip()
        return title, year, authors

    def match_rows(self) -> tuple[pd.DataFrame, list[MatchResult], list[BibEntry]]:
        results: list[MatchResult] = []
        selected_keys_in_order: list[str] = []
        selected_key_set = set()

        for row_index, (_, row) in enumerate(self.df.iterrows(), start=1):
            title, year, authors = self._row_context(row)
            csv_title = _normalize_text(title)
            csv_author = _normalize_person_name(_extract_first_csv_author(authors))

            if not year:
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="unmatched",
                        reason="missing year in CSV row",
                    )
                )
                continue

            candidates = self.bib_index.candidates_for_year(year)
            if not candidates:
                scored_candidates = []
            else:
                scored_candidates = self._rank_candidates(
                    csv_title, csv_author, candidates
                )

            candidate_summaries = [
                f"{entry.key} (title={title_score:.1f}, author={author_score:.1f}, combined={combined_score:.1f})"
                for title_score, author_score, combined_score, entry in scored_candidates[
                    :5
                ]
            ]

            if not scored_candidates and not self.bib_index.entries_without_year:
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="unmatched",
                        reason=f"no BibLaTeX entries found for year {year}",
                    )
                )
                continue

            fallback_candidates: list[tuple[float, float, float, BibEntry]] = []
            if not scored_candidates:
                fallback_candidates = self._rank_candidates(
                    csv_title, csv_author, self.bib_index.entries_without_year
                )

            active_candidates = scored_candidates or fallback_candidates
            candidate_summaries = [
                f"{entry.key} (title={title_score:.1f}, author={author_score:.1f}, combined={combined_score:.1f})"
                for title_score, author_score, combined_score, entry in active_candidates[
                    :5
                ]
            ]

            if not active_candidates:
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="unmatched",
                        reason=f"no BibLaTeX candidates available for {year}",
                    )
                )
                continue

            best_title_score, best_author_score, best_combined_score, best_entry = (
                active_candidates[0]
            )

            if best_title_score >= 99.5:
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="matched",
                        citation_key=best_entry.key,
                        title_score=best_title_score,
                        author_score=best_author_score,
                        combined_score=best_combined_score,
                        candidate_summaries=candidate_summaries,
                    )
                )

                if best_entry.key not in selected_key_set:
                    selected_key_set.add(best_entry.key)
                    selected_keys_in_order.append(best_entry.key)
                continue

            if self.bib_index.entries_without_year:
                yearless_candidates = self._rank_candidates(
                    csv_title, csv_author, self.bib_index.entries_without_year
                )
                if yearless_candidates and yearless_candidates[0][0] >= 99.5:
                    (
                        best_title_score,
                        best_author_score,
                        best_combined_score,
                        best_entry,
                    ) = yearless_candidates[0]
                    candidate_summaries = [
                        f"{entry.key} (title={title_score:.1f}, author={author_score:.1f}, combined={combined_score:.1f})"
                        for title_score, author_score, combined_score, entry in yearless_candidates[
                            :5
                        ]
                    ]
                    results.append(
                        MatchResult(
                            row_index=row_index,
                            title=title,
                            year=year,
                            authors=authors,
                            status="matched",
                            citation_key=best_entry.key,
                            title_score=best_title_score,
                            author_score=best_author_score,
                            combined_score=best_combined_score,
                            candidate_summaries=candidate_summaries,
                        )
                    )

                    if best_entry.key not in selected_key_set:
                        selected_key_set.add(best_entry.key)
                        selected_keys_in_order.append(best_entry.key)
                    continue

            all_exact_candidates = self._rank_candidates(
                csv_title, csv_author, self.bib_index.entries
            )
            if all_exact_candidates and all_exact_candidates[0][0] >= 99.5:
                best_title_score, best_author_score, best_combined_score, best_entry = (
                    all_exact_candidates[0]
                )
                candidate_summaries = [
                    f"{entry.key} (title={title_score:.1f}, author={author_score:.1f}, combined={combined_score:.1f})"
                    for title_score, author_score, combined_score, entry in all_exact_candidates[
                        :5
                    ]
                ]
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="matched",
                        citation_key=best_entry.key,
                        title_score=best_title_score,
                        author_score=best_author_score,
                        combined_score=best_combined_score,
                        candidate_summaries=candidate_summaries,
                    )
                )

                if best_entry.key not in selected_key_set:
                    selected_key_set.add(best_entry.key)
                    selected_keys_in_order.append(best_entry.key)
                continue

            if all_exact_candidates:
                top_title_score, top_author_score, top_combined_score, top_entry = (
                    all_exact_candidates[0]
                )
                second_title_score = (
                    all_exact_candidates[1][0]
                    if len(all_exact_candidates) > 1
                    else -1.0
                )
                if (
                    top_title_score >= 95.0
                    and top_combined_score >= 70.0
                    and (
                        len(all_exact_candidates) == 1
                        or top_title_score - second_title_score >= 5.0
                    )
                ):
                    candidate_summaries = [
                        f"{entry.key} (title={title_score:.1f}, author={author_score:.1f}, combined={combined_score:.1f})"
                        for title_score, author_score, combined_score, entry in all_exact_candidates[
                            :5
                        ]
                    ]
                    results.append(
                        MatchResult(
                            row_index=row_index,
                            title=title,
                            year=year,
                            authors=authors,
                            status="matched",
                            citation_key=top_entry.key,
                            title_score=top_title_score,
                            author_score=top_author_score,
                            combined_score=top_combined_score,
                            candidate_summaries=candidate_summaries,
                        )
                    )

                    if top_entry.key not in selected_key_set:
                        selected_key_set.add(top_entry.key)
                        selected_keys_in_order.append(top_entry.key)
                    continue

            if (
                best_title_score < self.title_threshold
                or best_author_score < self.author_threshold
                or best_combined_score < self.combined_threshold
            ):
                results.append(
                    MatchResult(
                        row_index=row_index,
                        title=title,
                        year=year,
                        authors=authors,
                        status="unmatched",
                        title_score=best_title_score,
                        author_score=best_author_score,
                        combined_score=best_combined_score,
                        candidate_summaries=candidate_summaries,
                        reason=(
                            "best candidate did not reach acceptance thresholds; "
                            f"best={best_entry.key}"
                        ),
                    )
                )
                continue

            if len(active_candidates) > 1:
                second_combined = active_candidates[1][2]
                if best_combined_score - second_combined <= self.ambiguity_margin:
                    results.append(
                        MatchResult(
                            row_index=row_index,
                            title=title,
                            year=year,
                            authors=authors,
                            status="ambiguous",
                            title_score=best_title_score,
                            author_score=best_author_score,
                            combined_score=best_combined_score,
                            candidate_summaries=candidate_summaries,
                            reason=(
                                f"top candidates are too close: {best_entry.key} vs. {active_candidates[1][3].key}"
                            ),
                        )
                    )
                    continue

            results.append(
                MatchResult(
                    row_index=row_index,
                    title=title,
                    year=year,
                    authors=authors,
                    status="matched",
                    citation_key=best_entry.key,
                    title_score=best_title_score,
                    author_score=best_author_score,
                    combined_score=best_combined_score,
                    candidate_summaries=candidate_summaries,
                )
            )

            if best_entry.key not in selected_key_set:
                selected_key_set.add(best_entry.key)
                selected_keys_in_order.append(best_entry.key)

        selected_entries = [
            entry for entry in self.bib_index.entries if entry.key in selected_key_set
        ]

        enriched_df = self.df.copy()
        citation_values: list[str] = []
        for result in results:
            citation_values.append(
                result.citation_key if result.status == "matched" else ""
            )

        if "Bib citations" in enriched_df.columns:
            enriched_df["Bib citations"] = citation_values
        else:
            insert_at = (
                enriched_df.columns.get_loc("Authors") + 1
                if "Authors" in enriched_df.columns
                else len(enriched_df.columns)
            )
            enriched_df.insert(insert_at, "Bib citations", citation_values)

        return enriched_df, results, selected_entries


def _build_report_rows(results: Sequence[MatchResult]) -> pd.DataFrame:
    report_rows = []
    for result in results:
        report_rows.append(
            {
                "row_index": result.row_index,
                "status": result.status,
                "title": result.title,
                "year": result.year,
                "authors": result.authors,
                "citation_key": result.citation_key,
                "title_score": f"{result.title_score:.1f}"
                if result.title_score
                else "",
                "author_score": f"{result.author_score:.1f}"
                if result.author_score
                else "",
                "combined_score": f"{result.combined_score:.1f}"
                if result.combined_score
                else "",
                "candidates": " | ".join(result.candidate_summaries),
                "reason": result.reason,
            }
        )
    return pd.DataFrame(report_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add BibLaTeX citation keys to a review CSV and export a filtered bibliography."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "selected_pdfs" / "final_output.csv"),
        help="Input CSV file path.",
    )
    parser.add_argument(
        "--biblatex",
        default=str(PROJECT_ROOT / "data" / "Source.biblatex"),
        help="BibLaTeX source file.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the enriched CSV. Defaults to <input_stem>_citations.csv.",
    )
    parser.add_argument(
        "--output-bib",
        default=None,
        help="Path for the filtered BibLaTeX file. Defaults to Source_filtered.biblatex beside the source bib.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the review report CSV. Defaults to <input_stem>_citations_review.csv.",
    )
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=90.0,
        help="Minimum title similarity score.",
    )
    parser.add_argument(
        "--author-threshold",
        type=float,
        default=85.0,
        help="Minimum first-author similarity score.",
    )
    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=92.0,
        help="Minimum combined score for acceptance.",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=3.0,
        help="Maximum score gap between top candidates before the match is considered ambiguous.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_csv = Path(args.input_csv)
    biblatex_path = Path(args.biblatex)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_csv.with_name(f"{input_csv.stem}_citations.csv")
    )
    output_bib = (
        Path(args.output_bib)
        if args.output_bib
        else biblatex_path.with_name("Source_filtered.bib")
    )
    report_path = (
        Path(args.report)
        if args.report
        else input_csv.with_name(f"{input_csv.stem}_citations_review.csv")
    )

    try:
        mapper = CitationMapper(
            input_csv,
            biblatex_path,
            title_threshold=args.title_threshold,
            author_threshold=args.author_threshold,
            combined_threshold=args.combined_threshold,
            ambiguity_margin=args.ambiguity_margin,
        )

        enriched_df, results, selected_entries = mapper.match_rows()
        report_df = _build_report_rows(results)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False, encoding="utf-8")
        logger.info("Wrote review report: %s", report_path)

        unmatched = [result for result in results if result.status != "matched"]
        if unmatched:
            logger.error(
                "Found %d unresolved rows (%d ambiguous, %d unmatched); review report written to %s",
                len(unmatched),
                sum(result.status == "ambiguous" for result in unmatched),
                sum(result.status == "unmatched" for result in unmatched),
                report_path,
            )
            return 1

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info("Wrote enriched CSV: %s", output_csv)

        output_bib.parent.mkdir(parents=True, exist_ok=True)
        output_bib.write_text(
            "\n\n".join(entry.raw.strip() for entry in selected_entries) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote filtered BibLaTeX file: %s", output_bib)
        logger.info(
            "Matched %d rows and %d unique citation keys",
            len(results),
            len(selected_entries),
        )
        return 0
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
