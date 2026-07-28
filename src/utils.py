"""Shared helpers for source fetchers and pruning scripts."""

import argparse
import csv
import hashlib
import json
import os
import re
import string
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_search_terms(config_file: str | os.PathLike) -> dict:
    """Load search terms from a YAML configuration file."""
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config (expected mapping): {path}")

    return config


def default_search_config_path() -> Path:
    """Return the default search config path shipped with the repository."""
    return Path(__file__).with_name("search_config.yaml")


def normalize_text(text: Any) -> str:
    """Lowercase and strip text for consistent comparison."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.lower().strip()


def normalize_string(text: Any) -> str:
    """Normalize for strict comparison: lowercase, remove punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: Any) -> int:
    """Count words in a text string, returning 0 for invalid input."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    clean_text = re.sub(r"[^\w\s]", " ", text)
    return len(clean_text.split())


def parse_year(year_val: Any) -> int:
    """Parse publication year robustly, falling back to current year."""
    try:
        if pd.isna(year_val):
            return datetime.now().year
        s = str(year_val).strip()
        if "-" in s:
            return int(s.split("-")[0])
        return int(float(s))
    except Exception:
        return datetime.now().year


def ensure_correct_extension(path: str, expected_ext: str) -> str:
    """Ensure the file path has the expected extension."""
    if not expected_ext.startswith("."):
        expected_ext = f".{expected_ext}"

    base, ext = os.path.splitext(path)
    if ext.lower() != expected_ext.lower():
        return base + expected_ext
    return path


class CacheManager:
    """Hash-based JSON cache for academic fetchers, keyed by query and year."""

    def __init__(self, output_dir: Path, prefix: str, query: str):
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / f"{prefix}_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()

    def get_cache_path(self, year: int) -> Path:
        return self.cache_dir / f"year_{year}_{self.query_hash}.json"

    def exists(self, year: int) -> bool:
        return self.get_cache_path(year).exists()

    def get(self, year: int) -> dict | None:
        """Retrieve cached data for a year, or None if absent/corrupted."""
        cache_file = self.get_cache_path(year)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("count", 0) > 0 or data.get("checked", False):
                return data
        except Exception:
            # Silently skip corrupted cache files
            pass
        return None

    def set(self, year: int, data: dict) -> None:
        cache_file = self.get_cache_path(year)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def create_base_argparser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with options shared across all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output folder for generated files (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Plot output format (default: pdf)",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        help="Enable titles inside figures (default: disabled)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to search terms configuration file",
    )
    return parser


VIS_CONFIG = {
    "style": "seaborn-v0_8-whitegrid",
    "figure_size": (12, 7),
    "dpi": 300,
    "bar_color": "#4C72B0",
    "trend_color": "#C44E52",
    "bar_alpha": 0.8,
    "min_year": 2000,
}


def plot_results(df: pd.DataFrame, output_path: str, title: str = "") -> None:
    """Plot the distribution of articles by year with a quadratic trend line."""
    if df.empty:
        return

    df = df.copy()
    df["clean_year"] = df["Year"].apply(parse_year)
    counts = df["clean_year"].value_counts().sort_index()

    min_year = VIS_CONFIG["min_year"]
    current_year = datetime.now().year
    counts = counts[(counts.index >= min_year) & (counts.index <= current_year + 1)]

    if counts.empty:
        return

    plt.style.use(VIS_CONFIG["style"])
    fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"])

    years = list(counts.index)
    values = list(counts.values)

    ax.bar(
        years,
        values,
        color=VIS_CONFIG["bar_color"],
        alpha=VIS_CONFIG["bar_alpha"],
        label="Articles",
    )

    if len(years) > 2:
        z = np.polyfit(years, values, 2)
        p = np.poly1d(z)
        smooth_years = np.linspace(min(years), max(years), 100)
        ax.plot(
            smooth_years,
            p(smooth_years),
            "--",
            color=VIS_CONFIG["trend_color"],
            alpha=0.8,
            linewidth=2,
            label="Trend",
        )

    max_count = max(values)
    max_year = years[values.index(max_count)]
    ax.annotate(
        f"Peak: {max_count}",
        (max_year, max_count),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontweight="bold",
    )

    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    print(f"Trend plot saved to: {output_path}")


class BaseAcademicSearcher(ABC):
    """Abstract base class for academic publication search APIs."""

    def __init__(self, source_name: str):
        self.source_name = source_name

    @abstractmethod
    def fetch_article_counts(self, start_year, end_year, search_terms, delay=1.0):
        """Fetch article counts and details for given years and search terms."""
        pass

    def plot_results(
        self,
        data,
        title: str,
        style: str = "seaborn-v0_8-paper",
        show_titles: bool = False,
    ):
        """Create a visualization of yearly publication counts."""
        counts = {year: year_data["count"] for year, year_data in data.items()}
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=(12, 7))

        years = list(counts.keys())
        counts = list(counts.values())

        ax.plot(years, counts, marker="o", linewidth=2, markersize=8)

        z = np.polyfit(years, counts, 2)
        p = np.poly1d(z)
        ax.plot(
            years,
            p(years),
            linestyle="--",
            color="0.35",
            alpha=0.9,
            linewidth=2,
            label="Trend Line",
        )

        if show_titles:
            ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Articles", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        for x, y in zip(years, counts):
            if y > 0:
                ax.annotate(
                    f"{y:,}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def export_to_csv(self, data, filename):
        """Export results to CSV file including article details."""
        rows = []
        for year, year_data in data.items():
            for article in year_data.get("articles", []):
                title = article.get("title", "")
                if isinstance(title, str) and title.endswith("."):
                    title = title.rstrip(".")

                rows.append(
                    {
                        "ID": article.get("id", ""),
                        "Title": title,
                        "Year": article.get("pub_date", str(year)),
                        "Authors": article.get("authors", ""),
                        "Venue": article.get("venue", ""),
                        "Abstract": article.get("abstract", ""),
                        "URL": article.get("url", ""),
                        "Citations": article.get("citations", 0),
                        "Keywords": article.get("keywords", ""),
                    }
                )

        df = pd.DataFrame(rows)
        column_order = [
            "ID",
            "Title",
            "Year",
            "Authors",
            "Venue",
            "Abstract",
            "URL",
            "Citations",
            "Keywords",
        ]
        df = df.reindex(columns=column_order)
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        print(f"Data exported to {filename}")


def run_academic_search(searcher, args, config, query):
    """Run a full search pipeline: fetch, plot, and export to CSV."""
    if not isinstance(args, dict):
        args = vars(args)

    os.makedirs(args["output"], exist_ok=True)

    start_year = config.get("start_year", 2004)
    end_year = config.get("end_year", 2024)

    results = searcher.fetch_article_counts(start_year, end_year, query)

    print("Article count: ", sum(year_data["count"] for year_data in results.values()))

    title = f"{searcher.source_name} Publications on differential diagnosis"
    plot_format = args.get("format", "pdf")
    show_titles = bool(args.get("title", False))
    fig = searcher.plot_results(results, title, show_titles=show_titles)

    plot_filename = os.path.join(
        args["output"],
        f"{searcher.source_name.lower().replace(' ', '_')}_graph.{plot_format}",
    )
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")

    csv_filename = os.path.join(
        args["output"], f"{searcher.source_name.lower().replace(' ', '_')}_data.csv"
    )
    searcher.export_to_csv(results, csv_filename)

    if plt.isinteractive():
        plt.show()
    else:
        print(f"Plot saved to {plot_filename}")
        plt.close(fig)
