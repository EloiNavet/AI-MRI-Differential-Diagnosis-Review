import os
import re
import textwrap
import warnings
from collections import Counter
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker

from .config import (
    FIGURE_PROFILES,
    ML_METHOD_COLORS,
    VIS_CONFIG,
    ensure_correct_extension,
)
from .utils import get_inverse_color, normalize_method_label


def _get_categorical_colors(n: int) -> list[str]:
    base = VIS_CONFIG.get("categorical_colors")
    if n <= 0:
        return []
    if not base:
        return sns.color_palette(
            VIS_CONFIG.get("sequential_palette", "Blues"), n
        ).as_hex()
    if n <= len(base):
        return list(base[:n])

    # For many categories, sample the sequential palette to avoid repeating colors.
    # Skip very light tones for better contrast in print.
    sampled = sns.color_palette(
        VIS_CONFIG.get("sequential_palette", "Blues"), n + 3
    ).as_hex()
    return sampled[1:-2]


def _get_sankey_like_colors(n: int) -> list[str]:
    """Return soft colors derived from the sankey palette used in this project."""
    if n <= 0:
        return []

    base = list(VIS_CONFIG.get("sankey_column_colors", {}).values())
    if not base:
        return _get_categorical_colors(n)

    if n <= len(base):
        return base[:n]

    colors = list(base)
    blend_steps = [0.18, 0.32, 0.46]
    while len(colors) < n:
        for base_color in base:
            rgb = np.array(mcolors.to_rgb(base_color))
            for blend in blend_steps:
                pastel = rgb * (1.0 - blend) + blend * np.array([1.0, 1.0, 1.0])
                colors.append(mcolors.to_hex(pastel))
                if len(colors) >= n:
                    return colors[:n]

    return colors[:n]


def _create_figure(profile: str = "standard", num_rows: Optional[int] = None):
    """Create figure with optional fixed height for barplots."""
    figsize = FIGURE_PROFILES[profile]
    if num_rows is not None and num_rows > 0:
        # Keep horizontal bar plots at a consistent fixed height.
        if profile in ("standard", "wide"):
            fixed_height = VIS_CONFIG.get("barplot_fixed_height", 4.0)
            figsize = (figsize[0], fixed_height)
    return plt.subplots(figsize=figsize, dpi=VIS_CONFIG["dpi"])


def _style_axes(
    ax,
    grid_axis: str = "both",
    grid_linestyle: str = "-",
    tick_size: Optional[float] = None,
):
    tick_label_size = VIS_CONFIG["tick_size"] if tick_size is None else tick_size
    ax.tick_params(axis="both", labelsize=tick_label_size)
    # Reset both axes first so style-level defaults never leak extra grid lines.
    ax.grid(False)

    if grid_axis in {"x", "both"}:
        ax.xaxis.grid(
            True,
            alpha=VIS_CONFIG["grid_alpha"],
            linewidth=VIS_CONFIG["grid_linewidth"],
            linestyle=grid_linestyle,
        )
    if grid_axis in {"y", "both"}:
        ax.yaxis.grid(
            True,
            alpha=VIS_CONFIG["grid_alpha"],
            linewidth=VIS_CONFIG["grid_linewidth"],
            linestyle=grid_linestyle,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(VIS_CONFIG["spine_linewidth"])
    ax.spines["bottom"].set_linewidth(VIS_CONFIG["spine_linewidth"])


def _apply_optional_title(ax, show_titles: bool, title: str):
    if show_titles:
        ax.set_title(title, fontsize=VIS_CONFIG["title_size"], pad=14)


def _normalize_output_formats(output_format: Any) -> list[str]:
    """Accept a single format or a list/tuple of formats."""
    if isinstance(output_format, str):
        return [output_format]
    if isinstance(output_format, Sequence):
        return [str(fmt).lower() for fmt in output_format if str(fmt).strip()]
    return ["pdf"]


def _save_matplotlib_figure(output_path: str, output_format: Any) -> list[str]:
    saved_paths: list[str] = []
    for fmt in _normalize_output_formats(output_format):
        format_path = ensure_correct_extension(output_path, fmt)
        plt.savefig(
            format_path,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
            edgecolor="none",
            dpi=VIS_CONFIG["dpi"],
        )
        saved_paths.append(format_path)
    return saved_paths


def _sanitize_filename_component(text: str) -> str:
    """Sanitize a text component for deterministic, filesystem-safe filenames."""
    if not text or not isinstance(text, str):
        return "unknown"

    normalized = re.sub(r"\bSubtypes\b", "subtypes", text, flags=re.IGNORECASE)
    normalized = normalized.replace(" vs. ", "_vs_")
    normalized = normalized.replace("/", "_")
    normalized = re.sub(r"[^A-Za-z0-9_()\-]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _filename_component_key(text: str) -> str:
    """Case-insensitive comparison key for deduplicating filename components."""
    if not text or not isinstance(text, str):
        return "unknown"
    normalized = re.sub(r"\bSubtypes\b", "subtypes", text, flags=re.IGNORECASE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower()


def _get_pastel_spectral_cmap():
    """Build a softer Spectral colormap for publication-friendly heatmaps."""
    spectral = sns.color_palette("Spectral", 256)
    pastel_colors = []
    white_mix = 0.28
    for color in spectral:
        pastel_colors.append(
            tuple(channel * (1.0 - white_mix) + white_mix for channel in color)
        )
    return mcolors.LinearSegmentedColormap.from_list("spectral_pastel", pastel_colors)


def _wrap_label(label: str, width: int = 18) -> str:
    if not label:
        return label
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def create_line_plot(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
):
    """Creates and saves a line plot."""
    fig, ax = _create_figure("standard")
    sorted_keys = sorted(data_dict.keys())
    values = [data_dict[key] for key in sorted_keys]

    colors = sns.color_palette(VIS_CONFIG["color_palette"], 5)

    ax.plot(
        sorted_keys,
        values,
        marker="o",
        linestyle="-",
        color=colors[3],
        linewidth=VIS_CONFIG["line_width"],
        markersize=VIS_CONFIG["marker_size"],
        markerfacecolor=colors[3],
        markeredgecolor="white",
        markeredgewidth=0.5,
        alpha=VIS_CONFIG["bar_alpha"],
    )

    ax.fill_between(
        sorted_keys,
        values,
        alpha=0.1,
        color=colors[2],
    )

    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
    _style_axes(ax)

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_ml_dl_stacked_area_plot(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
):
    """Creates a stacked-area plot for ML/DL/Hybrid publication counts over years."""
    if not data_dict:
        return

    fig, ax = _create_figure("wide")
    years = sorted(data_dict.keys())
    method_order = ["ML", "DL", "Hybrid (ML+DL)"]

    trend_colors = VIS_CONFIG.get("method_trend_colors", {})
    fallback_colors = _get_categorical_colors(len(method_order))
    method_colors = [
        trend_colors.get(method, fallback_colors[i])
        for i, method in enumerate(method_order)
    ]

    stacked_values = [
        [data_dict.get(year, {}).get(method, 0) for year in years]
        for method in method_order
    ]

    ax.stackplot(
        years,
        *stacked_values,
        labels=method_order,
        colors=method_colors,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.6,
    )

    total_counts = np.sum(np.asarray(stacked_values, dtype=float), axis=0)
    ax.plot(
        years,
        total_counts,
        color="#2F3E4E",
        linewidth=VIS_CONFIG["line_width"],
        marker="o",
        markersize=VIS_CONFIG["marker_size"],
        markerfacecolor="white",
        markeredgecolor="#2F3E4E",
        markeredgewidth=0.8,
        alpha=0.9,
        label="_nolegend_",
    )

    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    _apply_optional_title(ax, show_titles, title)

    ax.legend(
        loc="upper left",
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"] * font_scale,
        fancybox=VIS_CONFIG["legend_fancybox"],
        shadow=VIS_CONFIG["legend_shadow"],
        framealpha=VIS_CONFIG["legend_framealpha"],
        ncol=3,
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        grid_axis="y",
        grid_linestyle="--",
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_lollipop_plot(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    top_n: int = 10,
    horizontal: bool = False,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
):
    """Creates and saves a lollipop plot."""
    sorted_data_for_sizing = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_data_for_sizing = sorted_data_for_sizing[:top_n]

    num_rows = len(sorted_data_for_sizing) if horizontal else None
    fig, ax = _create_figure("standard", num_rows=num_rows)

    sorted_data = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_data = sorted_data[:top_n]

    if not sorted_data:
        plt.close(fig)
        return

    labels = [x[0] for x in sorted_data]
    values = [x[1] for x in sorted_data]

    positions = range(len(labels))

    color = VIS_CONFIG.get("fixed_bar_color", "#4C78A8")
    line_width = VIS_CONFIG.get("edge_width", 2.0)
    marker_size = VIS_CONFIG.get("lollipop_marker_size", 100) * font_scale

    if horizontal:
        ax.hlines(
            y=positions,
            xmin=0,
            xmax=values,
            color=color,
            linewidth=line_width,
            alpha=VIS_CONFIG.get("bar_alpha", 1.0),
        )
        ax.scatter(values, positions, color=color, s=marker_size, zorder=3)
        ax.set_xlim(left=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
        ax.set_ylabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
        ax.invert_yaxis()

        for pos, value in zip(positions, values):
            ax.text(
                value + max(values) * 0.02,
                pos,
                f"{int(value)}",
                ha="left",
                va="center",
                fontsize=VIS_CONFIG["value_label_size"] * font_scale,
            )
    else:
        ax.vlines(
            x=positions,
            ymin=0,
            ymax=values,
            color=color,
            linewidth=line_width,
            alpha=VIS_CONFIG.get("bar_alpha", 1.0),
        )
        ax.scatter(positions, values, color=color, s=marker_size, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
        ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)

        for pos, value in zip(positions, values):
            ax.text(
                pos,
                value + max(values) * 0.02,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=VIS_CONFIG["value_label_size"] * font_scale,
            )

    _apply_optional_title(ax, show_titles, title)

    grid_axis = "x" if horizontal else "y"
    _style_axes(
        ax,
        grid_axis=grid_axis,
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_pie_chart(
    data_dict: dict,
    title: str,
    output_path: str,
    top_n: int = 5,
    show_titles: bool = False,
    output_format: str = "pdf",
    total_count: int = None,
    font_scale: float = 1.0,
    radius: float = 1.0,
):
    """Creates and saves a pie chart."""
    fig, ax = _create_figure("square")
    sorted_data = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None and len(sorted_data) > top_n:
        top_items = sorted_data[:top_n]
        others_sum = sum(x[1] for x in sorted_data[top_n:])
        labels = [x[0] for x in top_items] + ["Others"]
        values = [x[1] for x in top_items] + [others_sum]
    else:
        labels, values = zip(*sorted_data) if sorted_data else ([], [])

    # Special-case ML/DL pie to keep the exact same category colors as trend plots.
    ml_keys = {"ML", "DL", "Hybrid (ML+DL)"}
    if labels and set(labels).issubset(ml_keys):
        ml_colors = VIS_CONFIG.get(
            "method_trend_colors", VIS_CONFIG.get("ml_method_pie_colors", {})
        )
        colors = [ml_colors.get(label, "#4C78A8") for label in labels]
    else:
        colors = _get_categorical_colors(len(labels))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: (
            f"{pct:.1f}%" if pct >= VIS_CONFIG["small_slice_threshold_pct"] else ""
        ),
        colors=colors,
        wedgeprops={
            "edgecolor": VIS_CONFIG["edge_color"],
            "linewidth": 1.1,
            "alpha": VIS_CONFIG["bar_alpha"],
        },
        textprops={"fontsize": VIS_CONFIG["tick_size"] * font_scale},
        startangle=90,
        pctdistance=VIS_CONFIG["pct_distance"],
        radius=radius,
    )

    for i, autotext in enumerate(autotexts):
        wedge_color = colors[i % len(colors)]
        autotext.set_color(get_inverse_color(wedge_color))
        autotext.set_fontsize(VIS_CONFIG["small_text_size"] * font_scale)

    for text in texts:
        text.set_fontsize(VIS_CONFIG["tick_size"] * font_scale)

    if total_count is not None:
        ax.text(
            0.5,
            -0.15,
            f"Total studies: {total_count}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=VIS_CONFIG["legend_size"] * font_scale,
            style="italic",
            alpha=VIS_CONFIG["annotation_alpha"],
        )

    ax.axis("equal")
    _apply_optional_title(ax, show_titles, title)

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_multiple_line_plot_percentage(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
    legend_font_scale: Optional[float] = None,
    color_palette: Optional[list[str]] = None,
):
    """Creates a multiple line plot showing percentage trends."""
    fig, ax = _create_figure("wide")
    effective_legend_scale = (
        font_scale if legend_font_scale is None else legend_font_scale
    )
    years = sorted(data_dict.keys())
    categories = sorted(
        {cat for year_data in data_dict.values() for cat in year_data.keys()}
    )

    trend_colors = VIS_CONFIG.get("method_trend_colors", {})
    fallback_colors = color_palette or _get_sankey_like_colors(len(categories))
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "*", "h"]
    linestyles = ["-", "--", "-.", ":"]

    for i, category in enumerate(categories):
        percentages = []
        for year in years:
            year_total = sum(data_dict[year].values())
            percentage = (
                (data_dict[year].get(category, 0) / year_total * 100)
                if year_total > 0
                else 0
            )
            percentages.append(percentage)

        ax.plot(
            years,
            percentages,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=category,
            color=trend_colors.get(category, fallback_colors[i % len(fallback_colors)]),
            linewidth=VIS_CONFIG["line_width"],
            markersize=VIS_CONFIG["marker_size"],
            markerfacecolor=trend_colors.get(
                category, fallback_colors[i % len(fallback_colors)]
            ),
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=VIS_CONFIG["bar_alpha"],
        )

    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    _apply_optional_title(ax, show_titles, title)

    ax.legend(
        loc="best",
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"] * effective_legend_scale,
        fancybox=VIS_CONFIG["legend_fancybox"],
        shadow=VIS_CONFIG["legend_shadow"],
        framealpha=VIS_CONFIG["legend_framealpha"],
    )

    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        grid_linestyle="--",
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_stacked_area_plot_percentage(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
    color_palette: Optional[list[str]] = None,
    normalize_to_percentage: bool = True,
):
    """Create a stacked-area plot using per-year percentages or raw counts."""
    if not data_dict:
        return

    fig, ax = _create_figure("wide")
    years = sorted(data_dict.keys())
    categories = sorted(
        {cat for year_data in data_dict.values() for cat in year_data.keys()}
    )

    trend_colors = VIS_CONFIG.get("method_trend_colors", {})
    fallback_colors = color_palette or _get_sankey_like_colors(len(categories))
    area_colors = [
        trend_colors.get(category, fallback_colors[i % len(fallback_colors)])
        for i, category in enumerate(categories)
    ]

    stacked_values = []
    for category in categories:
        values = []
        for year in years:
            raw_value = float(data_dict[year].get(category, 0))
            if normalize_to_percentage:
                year_total = sum(data_dict[year].values())
                value = (raw_value / year_total * 100.0) if year_total > 0 else 0.0
            else:
                value = raw_value
            values.append(value)
        stacked_values.append(values)

    ax.stackplot(
        years,
        *stacked_values,
        labels=categories,
        colors=area_colors,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.6,
    )

    total_curve = np.sum(np.asarray(stacked_values, dtype=float), axis=0)
    ax.plot(
        years,
        total_curve,
        color="#2F3E4E",
        linewidth=VIS_CONFIG["line_width"],
        marker="o",
        markersize=VIS_CONFIG["marker_size"],
        markerfacecolor="white",
        markeredgecolor="#2F3E4E",
        markeredgewidth=0.8,
        alpha=0.9,
        label="_nolegend_",
    )

    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    _apply_optional_title(ax, show_titles, title)

    ax.legend(
        loc="best",
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"] * font_scale,
        fancybox=VIS_CONFIG["legend_fancybox"],
        shadow=VIS_CONFIG["legend_shadow"],
        framealpha=VIS_CONFIG["legend_framealpha"],
    )

    if normalize_to_percentage:
        ax.set_ylim(-2, 102)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        grid_linestyle="--",
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_diseases_vs_modalities_plot(
    frequencies: Counter,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
):
    """Creates a scatter plot of diseases vs. modalities."""
    if not frequencies:
        return  # No data to plot

    fig, ax = _create_figure("wide")

    x_values = [key[0] for key in frequencies.keys()]
    y_values = [key[1] for key in frequencies.keys()]
    sizes = [freq * 60 for freq in frequencies.values()]
    frequency_values = list(frequencies.values())

    # Use a continuous pastel Spectral gradient where low N is blue and high N is red.
    freq_min = min(frequency_values)
    freq_max = max(frequency_values)
    spectral_blue_to_red = plt.get_cmap("Spectral_r")

    if freq_min == freq_max:
        norm_values = [0.5] * len(frequency_values)
    else:
        norm_values = [
            (freq - freq_min) / (freq_max - freq_min) for freq in frequency_values
        ]

    point_colors = []
    for norm in norm_values:
        base = spectral_blue_to_red(norm)
        desat = sns.desaturate(base[:3], 0.65)
        blend_with_white = 0.35
        pastel = tuple(
            channel * (1.0 - blend_with_white) + 1.0 * blend_with_white
            for channel in desat
        )
        point_colors.append(mcolors.to_hex(pastel))

    ax.scatter(
        x_values,
        y_values,
        s=sizes,
        alpha=VIS_CONFIG["bar_alpha"],
        c=point_colors,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("Number of diseases", fontsize=VIS_CONFIG["label_size"] * font_scale)
    ax.set_ylabel(
        "Number of modalities", fontsize=VIS_CONFIG["label_size"] * font_scale
    )
    if show_titles:
        _apply_optional_title(
            ax, show_titles, "Number of diseases vs. number of modalities"
        )

    unique_freqs = sorted(set(frequency_values))

    if len(unique_freqs) > 5:
        # Show min, max, and a few intermediate values
        legend_freqs = [unique_freqs[0]]  # min
        if len(unique_freqs) > 2:
            # Add some intermediate values
            step = len(unique_freqs) // 3
            legend_freqs.extend([unique_freqs[step], unique_freqs[2 * step]])
        legend_freqs.append(unique_freqs[-1])  # max
        legend_freqs = sorted(set(legend_freqs))  # Remove duplicates and sort
    else:
        legend_freqs = unique_freqs

    # Create legend elements
    legend_elements = []
    for freq in legend_freqs:
        if freq_min == freq_max:
            norm = 0.5
        else:
            norm = (freq - freq_min) / (freq_max - freq_min)
        base = spectral_blue_to_red(norm)
        desat = sns.desaturate(base[:3], 0.65)
        blend_with_white = 0.35
        pastel = tuple(
            channel * (1.0 - blend_with_white) + 1.0 * blend_with_white
            for channel in desat
        )
        legend_color = mcolors.to_hex(pastel)

        legend_elements.append(
            plt.scatter(
                [],
                [],
                s=freq * 60,
                alpha=VIS_CONFIG["bar_alpha"],
                color=legend_color,
                edgecolors="w",
                linewidth=0.5,
                label=f"{freq} study" if freq == 1 else f"{freq} studies",
            )
        )

    # Find the best legend position to avoid data points
    # Check if there are points in the upper right area
    max_x = max(x_values)
    max_y = max(y_values)
    upper_right_points = sum(
        1 for x, y in zip(x_values, y_values) if x > max_x * 0.7 and y > max_y * 0.7
    )

    # Choose legend location based on data distribution
    if upper_right_points > 0:
        # If there are points in upper right, try upper left
        upper_left_points = sum(
            1 for x, y in zip(x_values, y_values) if x < max_x * 0.3 and y > max_y * 0.7
        )
        if upper_left_points == 0:
            legend_loc = "upper left"
            bbox_anchor = None
        else:
            # If both upper corners are occupied, place outside the plot
            legend_loc = "center left"
            bbox_anchor = (1.02, 0.5)
    else:
        # Default to upper right if no conflicts
        legend_loc = "upper right"
        bbox_anchor = None

    # Add the legend with the determined position
    legend = ax.legend(
        handles=legend_elements,
        title="Number of studies",
        loc=legend_loc,
        frameon=VIS_CONFIG["legend_frameon"],
        framealpha=VIS_CONFIG["legend_framealpha"],
        fontsize=VIS_CONFIG["legend_size"] * font_scale,
        title_fontsize=VIS_CONFIG["legend_size"] * font_scale,
        bbox_to_anchor=bbox_anchor,
    )

    legend.get_title().set_fontweight("normal")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        grid_linestyle="--",
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_violin_plot(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    font_scale: float = 1.0,
    height_scale: float = 1.0,
    y_top_multiplier: float = 2.5,
):
    """
    Creates and saves a violin plot with scientific styling and statistical annotations.
    """
    if not data_dict:
        return

    fig, ax = _create_figure("wide")
    if height_scale != 1.0:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height * height_scale, forward=True)

    # Keep method colors consistent with other ML/DL/Hybrid figures.
    trend_colors = VIS_CONFIG.get("method_trend_colors", {})

    # Define the specific order for ML methods
    method_order = ["ML", "DL", "Hybrid (ML+DL)"]

    # Filter and order the methods based on what's available in data_dict
    methods = []
    for method in method_order:
        if method in data_dict:
            methods.append(method)

    # Add any remaining methods not in the predefined order
    for method in data_dict.keys():
        if method not in methods:
            methods.append(method)

    fallback_colors = _get_categorical_colors(len(methods))
    method_colors = []
    for i, method in enumerate(methods):
        method_colors.append(
            trend_colors.get(method, fallback_colors[i % len(fallback_colors)])
        )

    positions = range(len(methods))

    # Create violin plots with subtle styling and increased spacing
    violin_parts = ax.violinplot(
        [data_dict[method] for method in methods],
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.6,  # Reduced from 0.7 to increase spacing between violins
    )

    # Style the violin bodies
    for i, pc in enumerate(violin_parts["bodies"]):
        method_color = method_colors[i]
        pc.set_facecolor(method_color)
        pc.set_edgecolor(method_color)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.35)

    # Add subtle statistical indicators
    for i, method in enumerate(methods):
        data = data_dict[method]
        method_color = method_colors[i]
        mean_val = np.mean(data)
        std_val = np.std(data)
        n_studies = len(data)

        # Add subtle mean line (more prominent)
        ax.hlines(
            mean_val,
            i - 0.3,
            i + 0.3,
            colors=method_color,
            linestyles="-",
            linewidth=2.5,
            alpha=0.9,
            zorder=3,
        )

        # Add subtle std deviation indicators (more visible)
        # Only plot if mean ± std > 0 (log axis)
        if mean_val + std_val > 0:
            ax.hlines(
                mean_val + std_val,
                i - 0.2,
                i + 0.2,
                colors=method_color,
                linestyles="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=3,
            )
        if mean_val - std_val > 0:
            ax.hlines(
                mean_val - std_val,
                i - 0.2,
                i + 0.2,
                colors=method_color,
                linestyles="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=3,
            )
        # The vertical dashed line is removed as requested.

        # Add statistical annotations above each violin
        # Use log scale to position annotations correctly
        y_max = max(data)
        y_pos_n = y_max * 2.0  # Position for annotation (higher)
        y_pos_stats = y_max * 1.3  # Position for μ, σ annotation (much lower)

        # Format statistics nicely
        mean_formatted = f"{mean_val:,.0f}" if mean_val >= 10 else f"{mean_val:,.1f}"
        std_formatted = f"{std_val:,.0f}" if std_val >= 10 else f"{std_val:,.1f}"

        # Sample size annotation (bold, higher up) - bigger font
        ax.text(
            i,
            y_pos_n,
            f"{n_studies} articles",
            ha="center",
            va="center",
            fontsize=(VIS_CONFIG["small_text_size"] + 1) * font_scale,
            fontweight="normal",
            color="black",
        )

        # Statistics annotation (smaller, lower) - bigger font
        ax.text(
            i,
            y_pos_stats,
            rf"$\mu$ = {mean_formatted}, $\sigma$ = {std_formatted}",
            ha="center",
            va="center",
            fontsize=VIS_CONFIG["small_text_size"] * font_scale,
            color="gray",
            style="italic",
        )

    # Use logarithmic scale for y-axis
    ax.set_yscale("log")

    # Set labels and formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=0, ha="center")
    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"] * font_scale)
    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"] * font_scale)

    if show_titles:
        ax.set_title(title, fontsize=VIS_CONFIG["title_size"] * font_scale, pad=20)

    # Clean up the plot appearance
    _style_axes(
        ax,
        grid_axis="y",
        grid_linestyle=":",
        tick_size=VIS_CONFIG["tick_size"] * font_scale,
    )

    # Adjust y-axis limits to accommodate annotations (more space at top)
    all_data = [val for sublist in data_dict.values() for val in sublist]
    y_min, y_max = min(all_data), max(all_data)
    ax.set_ylim(y_min * 0.8, y_max * y_top_multiplier)

    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_lollipop_plot_with_totals(
    data_dict: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    top_n: int = 10,
    horizontal: bool = False,
    show_titles: bool = False,
    output_format: str = "pdf",
):
    """Creates and saves a lollipop plot with value labels and total count."""
    sorted_data_for_sizing = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_data_for_sizing = sorted_data_for_sizing[:top_n]

    num_rows = len(sorted_data_for_sizing) if horizontal else None
    profile = "wide" if not horizontal else "standard"
    fig, ax = _create_figure(profile, num_rows=num_rows)

    sorted_data = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_data = sorted_data[:top_n]

    if not sorted_data:
        plt.close(fig)
        return

    labels = [x[0] for x in sorted_data]
    values = [x[1] for x in sorted_data]
    positions = range(len(labels))

    bar_color = VIS_CONFIG.get("fixed_bar_color", "#4C78A8")
    line_width = VIS_CONFIG.get("edge_width", 2.0)
    marker_size = VIS_CONFIG.get("lollipop_marker_size", 40)
    total_subjects = sum(data_dict.values())

    def format_number(val):
        return f"{int(val):,}"

    if horizontal:
        ax.hlines(
            y=positions,
            xmin=0,
            xmax=values,
            color=bar_color,
            linewidth=line_width,
            alpha=VIS_CONFIG.get("bar_alpha", 1.0),
        )
        ax.scatter(values, positions, color=bar_color, s=marker_size, zorder=3)

        ax.set_xlim(left=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel, fontsize=VIS_CONFIG.get("label_size", 12))
        ax.set_ylabel(xlabel, fontsize=VIS_CONFIG.get("label_size", 12))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.invert_yaxis()

        for pos, value in zip(positions, values):
            ax.text(
                value + max(values) * 0.02,
                pos,
                format_number(value),
                ha="left",
                va="center",
                fontsize=VIS_CONFIG.get("small_text_size", 10),
            )
    else:
        ax.vlines(
            x=positions,
            ymin=0,
            ymax=values,
            color=bar_color,
            linewidth=line_width,
            alpha=VIS_CONFIG.get("bar_alpha", 1.0),
        )
        ax.scatter(positions, values, color=bar_color, s=marker_size, zorder=3)

        ax.set_ylim(bottom=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(xlabel, fontsize=VIS_CONFIG.get("label_size", 12))
        ax.set_ylabel(ylabel, fontsize=VIS_CONFIG.get("label_size", 12))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

        for pos, value in zip(positions, values):
            ax.text(
                pos,
                value + max(values) * 0.02,
                format_number(value),
                ha="center",
                va="bottom",
                fontsize=VIS_CONFIG.get("small_text_size", 10),
            )

    ax.text(
        0.98,
        0.98,
        f"Total: {format_number(total_subjects)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=VIS_CONFIG.get("small_text_size", 10),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=VIS_CONFIG.get("annotation_alpha", 0.9),
        ),
        zorder=4,
    )

    _apply_optional_title(ax, show_titles, title)
    grid_axis = "x" if horizontal else "y"
    _style_axes(ax, grid_axis=grid_axis)
    plt.tight_layout()

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_sankey_diagram(
    flow_data: list[dict[str, Any]],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    min_dataset_frequency: int = 2,
    min_disease_frequency: int = 2,
) -> None:
    """
    Creates a Sankey diagram showing the flow between diseases, datasets, and architectures.

    Args:
        flow_data: list of flow records with 'disease', 'dataset', 'architecture' keys
        output_path: Path where to save the plot
        show_titles: Whether to show the title
        output_format: Output format (pdf, png, svg, etc.)
        min_dataset_frequency: Minimum frequency for datasets to be included (handled upstream)
        min_disease_frequency: Minimum frequency for diseases to be included (handled upstream)
    """
    if not flow_data:
        return

    # Count flows between categories
    disease_dataset_flows = Counter()
    dataset_architecture_flows = Counter()

    for record in flow_data:
        disease_dataset_flows[(record["disease"], record["dataset"])] += 1
        dataset_architecture_flows[(record["dataset"], record["architecture"])] += 1

    # Get unique nodes
    diseases = sorted(set(record["disease"] for record in flow_data))
    datasets = sorted(set(record["dataset"] for record in flow_data))
    architectures = sorted(set(record["architecture"] for record in flow_data))

    # Create node list and mapping
    all_nodes = diseases + datasets + architectures

    # (Previously used for Plotly Sankey) kept as documentation aid
    _ = all_nodes

    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path
    from matplotlib import colors as mcolors

    def _stack_positions(nodes: list[str], totals: dict[str, int], pad: float = 0.012):
        total = sum(totals.get(n, 0) for n in nodes) or 1
        available = 1.0 - pad * (len(nodes) + 1)
        y = 1.0 - pad
        positions: dict[str, tuple[float, float]] = {}
        for n in nodes:
            h = available * (totals.get(n, 0) / total)
            positions[n] = (y - h, y)
            y = y - h - pad
        return positions

    # Totals per node
    disease_totals = Counter()
    dataset_totals = Counter()
    arch_totals = Counter()
    for (d, ds), c in disease_dataset_flows.items():
        disease_totals[d] += c
        dataset_totals[ds] += c
    for (ds, a), c in dataset_architecture_flows.items():
        dataset_totals[ds] += 0
        arch_totals[a] += c

    disease_pos = _stack_positions(diseases, disease_totals)
    dataset_pos = _stack_positions(datasets, dataset_totals)
    arch_pos = _stack_positions(architectures, arch_totals)

    # Column colors: blue / orange / magenta (avoid red-green pairing)
    col = VIS_CONFIG.get(
        "sankey_column_colors",
        {"disease": "#6FA8DC", "dataset": "#7BC8A4", "arch": "#F2B36D"},
    )

    fig, ax = _create_figure("detailed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_d, x_ds, x_a = 0.06, 0.46, 0.86
    w_node = 0.06

    def _draw_nodes(
        nodes: list[str],
        positions: dict[str, tuple[float, float]],
        x: float,
        color: str,
        label_position: str = "inside",  # "left", "inside", "right"
    ):
        for n in nodes:
            y0, y1 = positions[n]
            rect = Rectangle(
                (x, y0),
                w_node,
                y1 - y0,
                facecolor=mcolors.to_rgba(color, 0.25),
                edgecolor=mcolors.to_rgba(color, 0.9),
                linewidth=0.8,
            )
            ax.add_patch(rect)

            # Position label based on column position
            if label_position == "left":
                # Left column: label to the LEFT of the box, right-aligned
                text_x = x - 0.015
                ha = "right"
                text_color = "black"
            elif label_position == "right":
                # Right column: label to the RIGHT of the box, left-aligned
                text_x = x + w_node + 0.015
                ha = "left"
                text_color = "black"
            else:  # "inside"
                # Middle column: label inside the box, centered
                text_x = x + w_node / 2
                ha = "center"
                text_color = "black"

            display_label = n.replace("/", "\n")
            ax.text(
                text_x,
                (y0 + y1) / 2,
                display_label,
                ha=ha,
                va="center",
                fontsize=VIS_CONFIG["small_text_size"] * 0.85,
                color=text_color,
                weight="normal",
                rotation=0,
            )

    _draw_nodes(diseases, disease_pos, x_d, col["disease"], label_position="left")
    _draw_nodes(datasets, dataset_pos, x_ds, col["dataset"], label_position="inside")
    _draw_nodes(architectures, arch_pos, x_a, col["arch"], label_position="right")

    ax.text(
        x_d + w_node / 2,
        1.02,
        "Diseases",
        ha="center",
        va="bottom",
        fontsize=VIS_CONFIG["label_size"],
    )
    ax.text(
        x_ds + w_node / 2,
        1.02,
        "Datasets",
        ha="center",
        va="bottom",
        fontsize=VIS_CONFIG["label_size"],
    )
    ax.text(
        x_a + w_node / 2,
        1.02,
        "Architectures",
        ha="center",
        va="bottom",
        fontsize=VIS_CONFIG["label_size"],
    )

    def _draw_flow(src_pos, dst_pos, x0, x1, src, dst, val, src_offsets, dst_offsets):
        sy0, sy1 = src_pos[src]
        dy0, dy1 = dst_pos[dst]

        src_total = (sy1 - sy0) or 1e-9
        dst_total = (dy1 - dy0) or 1e-9

        h = (
            val
            / max(
                1,
                (
                    disease_totals[src]
                    if src in disease_totals
                    else dataset_totals.get(src, 1)
                ),
            )
        ) * src_total
        h = max(h, 1e-6)

        s0 = sy0 + src_offsets[src]
        s1 = min(s0 + h, sy1)
        src_offsets[src] += s1 - s0

        h_dst = (
            val
            / max(
                1,
                (
                    dataset_totals[dst]
                    if dst in dataset_totals
                    else arch_totals.get(dst, 1)
                ),
            )
        ) * dst_total
        h_dst = max(h_dst, 1e-6)
        d0 = dy0 + dst_offsets[dst]
        d1 = min(d0 + h_dst, dy1)
        dst_offsets[dst] += d1 - d0

        x_start = x0 + w_node
        x_end = x1
        x_mid = (x_start + x_end) / 2

        verts = [
            (x_start, s1),
            (x_mid, s1),
            (x_mid, d1),
            (x_end, d1),
            (x_end, d0),
            (x_mid, d0),
            (x_mid, s0),
            (x_start, s0),
            (x_start, s1),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        patch = PathPatch(
            Path(verts, codes),
            facecolor=(0.2, 0.2, 0.2, 0.12),
            edgecolor=(0.2, 0.2, 0.2, 0.08),
            linewidth=0.3,
        )
        ax.add_patch(patch)

    # Offsets for stacking flows within each node
    dd_src_offsets = {d: 0.0 for d in diseases}
    dd_dst_offsets = {ds: 0.0 for ds in datasets}
    da_src_offsets = {ds: 0.0 for ds in datasets}
    da_dst_offsets = {a: 0.0 for a in architectures}

    # Draw disease->dataset flows then dataset->arch flows
    for (d, ds), c in sorted(
        disease_dataset_flows.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
    ):
        _draw_flow(
            disease_pos,
            dataset_pos,
            x_d,
            x_ds,
            d,
            ds,
            c,
            dd_src_offsets,
            dd_dst_offsets,
        )

    for (ds, a), c in sorted(
        dataset_architecture_flows.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
    ):
        _draw_flow(
            dataset_pos, arch_pos, x_ds, x_a, ds, a, c, da_src_offsets, da_dst_offsets
        )

    if show_titles:
        title_text = "Flow between diseases, datasets, and architectures"
        filters = []
        if min_dataset_frequency > 1:
            filters.append(f"datasets ≥{min_dataset_frequency}")
        if min_disease_frequency > 1:
            filters.append(f"diseases ≥{min_disease_frequency}")
        if filters:
            title_text += f" ({', '.join(filters)})"
        ax.set_title(title_text, fontsize=VIS_CONFIG["title_size"], pad=16)

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def _filter_differential_network_comparisons(
    pairwise_comparisons: Counter,
    min_node_frequency: int = 2,
    min_edge_weight: int = 2,
) -> Counter:
    """Filter network by node frequency and edge weight (min enforced at 2)."""
    if not pairwise_comparisons:
        return Counter()

    effective_min_node_frequency = max(int(min_node_frequency), 2)
    effective_min_edge_weight = max(int(min_edge_weight), 2)

    disease_frequencies = Counter()
    for (disease1, disease2), count in pairwise_comparisons.items():
        disease_frequencies[disease1] += count
        disease_frequencies[disease2] += count

    frequent_diseases = {
        disease
        for disease, freq in disease_frequencies.items()
        if freq >= effective_min_node_frequency
    }

    filtered_comparisons = Counter()
    for (disease1, disease2), count in pairwise_comparisons.items():
        if (
            disease1 in frequent_diseases
            and disease2 in frequent_diseases
            and count >= effective_min_edge_weight
        ):
            filtered_comparisons[(disease1, disease2)] = count

    return filtered_comparisons


def create_differential_diagnosis_network(
    pairwise_comparisons: Counter,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    min_node_frequency: int = 2,
    min_edge_weight: int = 2,
):
    """
    Creates a network diagram showing disease comparison relationships.
    Filters out nodes with low weighted frequency and low-weight edges.
    """
    import networkx as nx

    if not pairwise_comparisons:
        return

    effective_min_node_frequency = max(int(min_node_frequency), 2)
    effective_min_edge_weight = max(int(min_edge_weight), 2)
    filtered_comparisons = _filter_differential_network_comparisons(
        pairwise_comparisons,
        min_node_frequency=effective_min_node_frequency,
        min_edge_weight=effective_min_edge_weight,
    )

    if not filtered_comparisons:
        return

    G = nx.Graph()
    for (disease1, disease2), count in filtered_comparisons.items():
        G.add_edge(disease1, disease2, weight=count)

    if len(G.nodes()) < 2:
        return

    fig, ax = _create_figure("detailed")
    fig.set_size_inches(10, 8)

    pos = nx.spring_layout(G, k=1, iterations=8000, threshold=1e-10, seed=42)

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    max_weight = max(weights) if weights else 1

    edge_colors = sns.color_palette("Blues", 3)
    for (u, v), weight in zip(edges, weights):
        norm_weight = weight / max_weight
        if norm_weight >= 0.66:
            edge_color, edge_width, edge_alpha = edge_colors[2], 2.5, 0.75
        elif norm_weight >= 0.33:
            edge_color, edge_width, edge_alpha = edge_colors[1], 1.8, 0.6
        else:
            edge_color, edge_width, edge_alpha = edge_colors[0], 1.0, 0.4

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=edge_width,
            alpha=edge_alpha,
            edge_color=edge_color,
        )

    node_color = VIS_CONFIG.get("fixed_bar_color", "#4C78A8")
    node_sizes = []
    for node in G.nodes():
        degree = G.degree(node)
        size = 800 + (degree / max(G.degree(nbr) for nbr in G.nodes())) * 800
        node_sizes.append(size)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=node_sizes,
        alpha=0.85,
        linewidths=1.2,
        edgecolors="#2F4F4F",
    )

    # Draw labels - clean, centered, readable
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=VIS_CONFIG["tick_size"] * 1.35,
        font_weight="normal",
        font_color="white",
    )

    # Add edge labels for all connections
    edge_labels = {}
    for (u, v), weight in zip(edges, weights):
        edge_labels[(u, v)] = str(int(weight))

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=VIS_CONFIG["small_text_size"] * 1.1,
            font_color="#4C4C4C",
        )

    title_text = f"Disease comparison network (node freq >= {effective_min_node_frequency}, edge weight >= {effective_min_edge_weight})"
    if show_titles:
        ax.set_title(title_text, fontsize=VIS_CONFIG["title_size"], pad=10)

    ax.axis("off")
    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_comparison_types_plot(
    comparison_types: Counter,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    exclude_cn_comparisons: bool = False,
    top_n: Optional[int] = None,
    small_slice_threshold_pct: float = 2.0,
    dark_for_high_values: bool = False,
    use_pastel_gradient: bool = False,
    font_scale: float = 1.0,
    legend_anchor_x: float = 1.02,
    pie_radius: float = 1.0,
    compact_layout: bool = False,
):
    """
    Creates a pie chart showing the frequency of different comparison types.

    Args:
        comparison_types: Counter of comparison types and their frequencies
        output_path: Path to save the plot
        show_titles: Whether to show the title
        output_format: Output format (pdf, png, etc.)
        exclude_cn_comparisons: Whether to exclude comparisons involving only CN vs. one disease
        top_n: Maximum number of comparison types to show (others grouped as "Others"). If None, show all.
    """
    if not comparison_types:
        return

    filtered_comparisons = comparison_types.copy()
    if exclude_cn_comparisons:
        filtered_comparisons = Counter(
            {
                comp: count
                for comp, count in comparison_types.items()
                if not (comp.count(" vs. ") == 1 and "CN" in comp)
            }
        )

    if not filtered_comparisons:
        return

    if top_n is not None:
        top_comparisons = filtered_comparisons.most_common(top_n)
    else:
        top_comparisons = filtered_comparisons.most_common()

    if not top_comparisons:
        return

    # Calculate total for percentage calculation
    total_studies = sum(filtered_comparisons.values())
    top_comparison_names = {comp for comp, _ in top_comparisons}

    def _rare_comparison_bucket(label: str) -> str:
        lower_label = label.lower()
        if "subtype" in lower_label:
            return "Other subtype comparisons"
        if label.count(" vs. ") > 1:
            return "Other multi-class comparisons"
        return "Other pairwise comparisons"

    labels = []
    values = []
    other_counts: Counter[str] = Counter()

    for comp, count in filtered_comparisons.items():
        percentage = (count / total_studies) * 100
        if comp in top_comparison_names and percentage >= small_slice_threshold_pct:
            labels.append(comp)
            values.append(count)
        else:
            other_counts[_rare_comparison_bucket(comp)] += count

    # Add rare-comparison categories if there are small categories
    for other_label, other_count in other_counts.items():
        if other_count > 0:
            labels.append(other_label)
            values.append(other_count)

    # Ensure order is always from most frequent to least frequent.
    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels = [label for label, _ in sorted_pairs]
    values = [value for _, value in sorted_pairs]

    # More aggressive label wrapping for readability
    wrapped_labels: list[str] = []
    for label in labels:
        if label.startswith("Other "):
            wrapped_labels.append(label)
        elif label == "AD vs. FTD subtypes":
            wrapped_labels.append(label)
        elif len(label) > 10:  # Even more aggressive wrapping
            parts = label.split(" vs. ")
            if len(parts) == 2:
                # For two-part comparisons, always wrap if longer than 10 chars
                wrapped_labels.append(f"{parts[0]} vs.\n{parts[1]}")
            elif len(parts) > 2:
                # For multi-part comparisons, wrap after first two
                wrapped_labels.append(
                    f"{parts[0]} vs. {parts[1]}\nvs. {' vs. '.join(parts[2:])}"
                )
            else:
                wrapped_labels.append(label)
        else:
            # For short labels, ensure " vs. " formatting with period
            label_with_period = label.replace(" vs ", " vs. ")
            wrapped_labels.append(label_with_period)

    if dark_for_high_values:
        # values are ordered descending; map highest frequencies to darkest blues.
        shades = sns.color_palette("Blues", len(labels) + 2).as_hex()[1:-1]
        plot_colors = list(reversed(shades))
    elif use_pastel_gradient:
        # True continuous Spectral gradient, softened to pastel.
        spectral = sns.color_palette("Spectral", n_colors=len(labels))
        pastel_colors = []
        for color in spectral:
            desat = sns.desaturate(color, 0.65)
            blend_with_white = 0.35
            pastel = tuple(
                channel * (1.0 - blend_with_white) + 1.0 * blend_with_white
                for channel in desat
            )
            pastel_colors.append(mcolors.to_hex(pastel))
        plot_colors = pastel_colors
    else:
        pastel_palette = VIS_CONFIG.get("comparison_types_pastel_colors", [])
        if pastel_palette and len(labels) <= len(pastel_palette):
            plot_colors = pastel_palette[: len(labels)]
        elif pastel_palette:
            # Extend with a soft HUSL palette if categories exceed the base pastel list.
            extra = len(labels) - len(pastel_palette)
            extended = sns.husl_palette(extra, s=0.55, l=0.78).as_hex()
            plot_colors = pastel_palette + extended
        else:
            plot_colors = _get_categorical_colors(len(labels))

    fig, ax = _create_figure("wide")

    pie_center = (0.0, 0.0)
    pie_radius_effective = pie_radius
    if compact_layout:
        # Keep a compact pie while reserving a separate legend panel.
        pie_center = (0.0, 0.0)
        pie_radius_effective = pie_radius * 1.02

    wedges, _ = ax.pie(
        values,
        labels=None,
        colors=plot_colors,
        startangle=90,
        radius=pie_radius_effective,
        center=pie_center,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.0,
            "alpha": 0.9 if use_pastel_gradient else VIS_CONFIG["bar_alpha"],
            "clip_on": True,
        },
    )

    # Legend with label + percent keeps the figure clean (Nature-style)
    legend_labels = []
    for label, v in zip(wrapped_labels, values):
        pct = (v / total_studies) * 100 if total_studies else 0
        legend_labels.append(f"{label} ({pct:.1f}%)")

    if compact_layout:
        # Place legend outside the pie axis to guarantee no overlap.
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=VIS_CONFIG["legend_size"] * font_scale,
        )
    else:
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(legend_anchor_x, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=VIS_CONFIG["legend_size"] * font_scale,
        )

    if show_titles:
        title_text = "Distribution of disease comparison types"
        if exclude_cn_comparisons:
            title_text += " (excluding simple CN comparisons)"
        _apply_optional_title(ax, show_titles, title_text)

    ax.axis("equal")
    if compact_layout:
        # Use full axis for the pie; legend stays outside on the right.
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        ax.set_aspect("equal", adjustable="box", anchor="C")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    else:
        plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_stacked_bar_plot(
    data_by_group: dict[str, dict[str, int]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    category_order: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
):
    """Create a stacked bar plot from nested count dictionaries."""
    if not data_by_group:
        return

    groups = list(data_by_group.keys())
    inferred_categories = sorted(
        {cat for group_counts in data_by_group.values() for cat in group_counts.keys()}
    )
    categories = category_order or inferred_categories

    fig, ax = _create_figure("wide")
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups), dtype=float)
    plot_colors = colors or _get_sankey_like_colors(max(len(categories), 1))

    for i, category in enumerate(categories):
        values = np.array(
            [data_by_group.get(group, {}).get(category, 0) for group in groups],
            dtype=float,
        )
        if np.all(values == 0):
            continue

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            color=plot_colors[i % len(plot_colors)],
            alpha=VIS_CONFIG["bar_alpha"],
            edgecolor="white",
            linewidth=VIS_CONFIG["edge_width"],
            label=category,
        )

        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=VIS_CONFIG["small_text_size"],
            )

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel, fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel(ylabel, fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)
    _style_axes(ax, grid_axis="y")

    ax.legend(
        loc="best",
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"],
        framealpha=VIS_CONFIG["legend_framealpha"],
    )

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_modality_overlap_diagram(
    modality_sets: list[set],
    modality_labels: list[str],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Top modality overlap (Euler diagram)",
):
    """Create an overlap diagram for the most frequent modalities."""
    if not modality_sets or not modality_labels:
        return

    if len(modality_sets) != len(modality_labels):
        raise ValueError("modality_sets and modality_labels must have the same length")

    try:
        from matplotlib_set_diagrams import EulerDiagram
    except ImportError as exc:  # pragma: no cover - dependency issue should be explicit
        raise ImportError(
            "matplotlib-set-diagrams is required to render the modality overlap diagram"
        ) from exc

    palette = sns.color_palette("Set3", len(modality_sets)).as_hex()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=f"{label} (N={len(study_set)})",
        )
        for label, study_set, color in zip(modality_labels, modality_sets, palette)
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.4), dpi=VIS_CONFIG["dpi"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"matplotlib_set_diagrams\._diagram_classes",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Layout engine failed to find a solution.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The number of non-empty subsets is .*",
            category=UserWarning,
        )
        EulerDiagram.from_sets(
            modality_sets,
            set_labels=["" for _ in modality_sets],
            set_colors=palette,
            ax=ax,
        )

    ax.axis("off")
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        frameon=False,
        fontsize=VIS_CONFIG["legend_size"],
        handletextpad=0.6,
        columnspacing=1.4,
    )

    if show_titles:
        _apply_optional_title(ax, show_titles, title)

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_probast_domain_dot_grid_plot(
    data_by_group: dict[str, dict[str, int]],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "PROBAST domain-level risk",
    total_studies: Optional[int] = None,
):
    """Create a compact dot-grid PROBAST domain plot (P1..P4)."""
    if not data_by_group:
        return

    domain_labels = {
        "P1": "Participants",
        "P2": "Predictors / Features",
        "P3": "Outcome",
        "P4": "Statistical Analysis",
    }

    ordered_domains = [d for d in ["P1", "P2", "P3", "P4"] if d in data_by_group]
    for domain in data_by_group.keys():
        if domain not in ordered_domains:
            ordered_domains.append(domain)

    if not ordered_domains:
        return

    if total_studies is None:
        total_studies = max(
            (sum(domain_counts.values()) for domain_counts in data_by_group.values()),
            default=0,
        )
    total_studies = max(int(total_studies), 1)

    cols, rows = 10, 9
    total_slots = cols * rows
    x_coords = np.tile(np.arange(cols), rows)
    y_coords = np.repeat(np.arange(rows)[::-1], cols)

    colors = {
        "High": "#CB4335",
        "Unclear": "#D4AC0D",
        "Unknown": "#A6ACAF",
        "Low": "#2E86C1",
        "Empty": "#EAEDED",
        "BgHigh": "#FDEDEC",
        "BgUnclear": "#FEF9E7",
        "BgLow": "#F4F7F9",
        "BgUnknown": "#F2F3F4",
    }

    fig_width = 12.2 if len(ordered_domains) >= 4 else 3.05 * len(ordered_domains)
    fig, axes = plt.subplots(1, len(ordered_domains), figsize=(fig_width, 5.25))
    if len(ordered_domains) == 1:
        axes = [axes]
    fig.subplots_adjust(
        left=0.03,
        right=0.992,
        wspace=0.09,
        bottom=0.16,
        top=0.82 if show_titles else 0.99,
    )

    total_by_risk = {"High": 0, "Unclear": 0, "Unknown": 0, "Low": 0}

    for ax, domain in zip(axes, ordered_domains):
        stats = data_by_group.get(domain, {})
        normalized_stats = {}
        for key, value in stats.items():
            risk_key = str(key or "").strip().lower()
            try:
                normalized_stats[risk_key] = int(value or 0)
            except (TypeError, ValueError):
                normalized_stats[risk_key] = 0

        high = normalized_stats.get("high", 0)
        low = normalized_stats.get("low", 0)
        unclear = normalized_stats.get("unclear", 0)
        unknown = normalized_stats.get("unknown", 0) + normalized_stats.get("unk", 0)

        total_by_risk["High"] += high
        total_by_risk["Low"] += low
        total_by_risk["Unclear"] += unclear
        total_by_risk["Unknown"] += unknown

        status_seq = (
            ["High"] * high
            + ["Unclear"] * unclear
            + ["Unknown"] * unknown
            + ["Low"] * low
        )

        filled_points = min(total_studies, len(status_seq), total_slots)

        ax.set_aspect("equal")
        ax.axis("off")

        counts = {"High": high, "Unclear": unclear, "Low": low, "Unknown": unknown}
        dominant_risk = max(counts, key=counts.get)

        bg_color = colors[f"Bg{dominant_risk}"]

        bbox = patches.FancyBboxPatch(
            (-0.36, -0.36),
            cols - 0.28,
            rows - 0.28,
            boxstyle="round,pad=0.08,rounding_size=0.28",
            facecolor=bg_color,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(bbox)

        if filled_points > 0:
            point_colors = [colors[status] for status in status_seq[:filled_points]]
            ax.scatter(
                x_coords[:filled_points],
                y_coords[:filled_points],
                c=point_colors,
                s=118,
                zorder=1,
            )

        if total_slots > filled_points:
            ax.scatter(
                x_coords[filled_points:total_slots],
                y_coords[filled_points:total_slots],
                facecolors="none",
                edgecolors=colors["Empty"],
                s=118,
                linewidth=1.0,
                zorder=1,
            )

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.52, rows - 0.48)

        ax.text(
            0.5,
            -0.012,
            domain,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            0.5,
            -0.08,
            domain_labels.get(domain, domain),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            color="#777777",
        )

        y_offset = -0.155
        summary = [
            ("Low", low),
            ("High", high),
            ("Unclear", unclear),
            ("Unknown", unknown),
        ]

        # Filter to only non-zero categories
        summary_nonzero = [(cat, count) for cat, count in summary if count > 0]

        # Calculate exact percentages and their fractional parts for smart rounding
        if summary_nonzero:
            exact_pcts = [
                (cat, (count / total_studies) * 100) for cat, count in summary_nonzero
            ]
            # Start with floor values
            rounded_pcts_dict = {cat: int(pct) for cat, pct in exact_pcts}

            # Calculate remainder and fractional parts
            remainder = 100 - sum(rounded_pcts_dict.values())
            fractions = {cat: pct - int(pct) for cat, pct in exact_pcts}

            # Distribute remainder to categories with largest fractional parts
            if remainder > 0:
                sorted_by_frac = sorted(fractions.items(), key=lambda x: (-x[1], x[0]))
                for i in range(remainder):
                    cat_to_increment = sorted_by_frac[i][0]
                    rounded_pcts_dict[cat_to_increment] += 1
        else:
            rounded_pcts_dict = {}

        for category, count in summary:
            if count <= 0:
                continue

            pct = rounded_pcts_dict.get(category, 0)
            txt = f"{pct}% {category.lower()}"
            c_hex = colors[category]

            if category in {"Unclear", "Unknown"}:
                ax.text(
                    0.5,
                    y_offset,
                    txt,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=12,
                    color=c_hex,
                )
            else:
                ax.text(
                    0.5,
                    y_offset,
                    txt,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=12,
                    fontweight="bold",
                    color=c_hex,
                )
            y_offset -= 0.072

    if show_titles:
        fig.suptitle(title, fontsize=VIS_CONFIG["title_size"], y=0.985)

    legend_specs = [
        ("High", "High risk"),
        ("Unclear", "Unclear"),
        ("Unknown", "Unknown"),
        ("Low", "Low risk"),
    ]
    legend_elements = []
    for risk_key, legend_label in legend_specs:
        if total_by_risk.get(risk_key, 0) <= 0:
            continue
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[risk_key],
                markersize=9,
                label=legend_label,
            )
        )

    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93 if show_titles else 0.875),
            ncol=min(4, len(legend_elements)),
            frameon=False,
            fontsize=VIS_CONFIG["legend_size"] * 1.5,
            handletextpad=0.06,
            columnspacing=1.1,
        )

    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_radar_plot(
    profiles_by_method: dict[str, dict[str, float]],
    axes_labels: list[str],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Study readiness profile by method family",
    smoothing: bool = False,
    method_counts: dict[str, int] = None,
):
    """Create a radar chart comparing method families across study-quality axes.

    Args:
        profiles_by_method: dict mapping method -> {axis: proportion}
        axes_labels: list of axis labels
        output_path: Output file path
        show_titles: Whether to show plot title
        output_format: Output format (pdf, png, etc.)
        title: Plot title
        smoothing: If True, use spline interpolation for smoother curves
        method_counts: dict mapping method -> count of studies for legend annotation
    """
    if not profiles_by_method or not axes_labels:
        return

    method_order = ["ML", "DL", "Hybrid (ML+DL)"]
    trend_colors = VIS_CONFIG.get("method_trend_colors", {})
    method_colors = [
        trend_colors.get(
            method, ML_METHOD_COLORS.get(method, _get_categorical_colors(3)[i % 3])
        )
        for i, method in enumerate(method_order)
    ]

    num_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig_width, fig_height = FIGURE_PROFILES["square"]
    fig, ax = plt.subplots(
        figsize=(fig_width * 1.26, fig_height * 1.18),
        subplot_kw={"polar": True},
        dpi=VIS_CONFIG["dpi"],
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"], fontsize=VIS_CONFIG["tick_size"] + 2
    )
    ax.grid(
        color="#D7E0EA", linewidth=VIS_CONFIG["grid_linewidth"], alpha=0.9, zorder=1
    )
    ax.spines["polar"].set_color("#C9D4E2")

    wrapped_labels = [_wrap_label(label, width=14) for label in axes_labels]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        wrapped_labels,
        fontsize=VIS_CONFIG["tick_size"] + 3,
    )
    ax.tick_params(axis="x", pad=24)

    for i, method in enumerate(method_order):
        values = [
            profiles_by_method.get(method, {}).get(axis, 0.0) for axis in axes_labels
        ]
        if not any(values):
            continue

        color = method_colors[i]

        # Create legend label with study count if available
        legend_label = method
        if method_counts and method in method_counts:
            legend_label = f"{method} (N={method_counts[method]})"

        # Apply spline smoothing if requested
        if smoothing and len(values) > 3:
            from scipy.interpolate import CubicSpline

            # Build periodic spline: duplicate first point at the end for periodic bc
            angles_open = np.array(angles[:-1])
            values_open = np.array(values)
            angles_periodic = np.concatenate(
                [angles_open, [angles_open[0] + 2 * np.pi]]
            )
            values_periodic = np.concatenate([values_open, [values_open[0]]])

            cs = CubicSpline(angles_periodic, values_periodic, bc_type="periodic")
            smooth_angles = np.linspace(0, 2 * np.pi, 300, endpoint=True)
            smooth_values = cs(smooth_angles)
            smooth_values = np.clip(smooth_values, 0, 1)
            ax.plot(
                smooth_angles,
                smooth_values,
                color=color,
                linewidth=VIS_CONFIG["line_width"] + 0.4,
                label=legend_label,
                zorder=5,
            )
            ax.fill(smooth_angles, smooth_values, color=color, alpha=0.14, zorder=4)
        else:
            values_closed = values + values[:1]
            ax.plot(
                angles,
                values_closed,
                color=color,
                linewidth=VIS_CONFIG["line_width"] + 0.4,
                marker="o",
                markersize=7,
                label=legend_label,
                zorder=5,
            )
            ax.fill(angles, values_closed, color=color, alpha=0.14, zorder=4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"] + 2,
        framealpha=VIS_CONFIG["legend_framealpha"],
        ncol=3,
    )

    _apply_optional_title(ax, show_titles, title)
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.88)
    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_disease_comparison_heatmap(
    pairwise_comparisons: Counter,
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Disease comparison adjacency heatmap",
):
    """Create a stylized adjacency heatmap from disease pairwise comparison counts."""
    if not pairwise_comparisons:
        return

    node_frequencies = Counter()
    for (disease1, disease2), count in pairwise_comparisons.items():
        node_frequencies[disease1] += count
        node_frequencies[disease2] += count

    diseases = [
        disease
        for disease, _ in sorted(
            node_frequencies.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    if len(diseases) < 2:
        return

    disease_index = {disease: i for i, disease in enumerate(diseases)}
    matrix = np.zeros((len(diseases), len(diseases)), dtype=int)
    for (disease1, disease2), count in pairwise_comparisons.items():
        if disease1 not in disease_index or disease2 not in disease_index:
            continue
        i = disease_index[disease1]
        j = disease_index[disease2]
        matrix[i, j] = count
        matrix[j, i] = count

    annotations = np.empty(matrix.shape, dtype=object)
    annotations[:] = ""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                annotations[i, j] = str(int(matrix[i, j]))

    cmap = _get_pastel_spectral_cmap()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    size = max(6.5, 0.45 * len(diseases) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size), dpi=VIS_CONFIG["dpi"])

    mask = (matrix == 0) | np.eye(len(diseases), dtype=bool)
    plot_matrix = matrix.astype(float)
    plot_matrix[mask] = np.nan
    max_comparisons = int(matrix.max()) if matrix.size > 0 else 1
    min_comparisons = 1

    sns.heatmap(
        plot_matrix,
        mask=mask,
        cmap=cmap,
        vmin=min_comparisons,
        vmax=max(max_comparisons, min_comparisons),
        square=True,
        linewidths=0.8,
        linecolor="white",
        annot=annotations,
        fmt="",
        annot_kws={"size": VIS_CONFIG["small_text_size"], "weight": "bold"},
        cbar_kws={
            "shrink": 0.62,
            "label": "Comparison count",
            "fraction": 0.045,
            "pad": 0.02,
        },
        ax=ax,
    )

    ax.set_xticklabels(diseases, rotation=45, ha="right")
    ax.set_yticklabels(diseases, rotation=0)
    ax.set_xlabel("Disease", fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel("Disease", fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)
    ax.tick_params(axis="both", labelsize=VIS_CONFIG["tick_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_accuracy_vs_complexity_scatter(
    study_points: list[dict[str, Any]],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Reported performance vs. diagnostic complexity",
):
    """Create a scatter plot of performance against the number of diagnostic classes."""
    if not study_points:
        return

    fig, ax = _create_figure("wide")
    trend_colors = VIS_CONFIG.get("method_trend_colors", {})
    domain_styles = {
        "ID": {
            "marker": "o",
            "color": trend_colors.get("ML", "#6FA8DC"),
            "label": "ID",
        },
        "OOD": {
            "marker": "^",
            "color": trend_colors.get("DL", "#7BC8A4"),
            "label": "OOD",
        },
    }

    for domain, style in domain_styles.items():
        domain_points = [
            point for point in study_points if point.get("domain") == domain
        ]
        if not domain_points:
            continue

        x_values = []
        y_values = []
        for index, point in enumerate(domain_points):
            offset = ((point.get("source_row_index", index) % 7) - 3) * 0.04
            x_values.append(point["classes"] + offset)
            y_values.append(point["performance"])

        ax.scatter(
            x_values,
            y_values,
            s=42,
            alpha=0.74,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.6,
            label=style["label"],
        )

        # No connecting trend line: only scatter points for visual clarity.

    ax.set_xlabel("Number of classes", fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel("Reported performance (%)", fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax, grid_axis="y")

    ax.legend(
        loc="best",
        frameon=VIS_CONFIG["legend_frameon"],
        fontsize=VIS_CONFIG["legend_size"],
        framealpha=VIS_CONFIG["legend_framealpha"],
    )

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_reproducibility_matrix_heatmap(
    matrix_data: dict[str, Any],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Reproducibility matrix (Code × Data availability)",
):
    """Create a matrix heatmap with counts and pretrained-weights overlays."""
    if not matrix_data:
        return

    row_labels = matrix_data.get("row_labels", [])
    col_labels = matrix_data.get("col_labels", [])
    counts = np.array(matrix_data.get("counts", []), dtype=float)
    weights = np.array(matrix_data.get("weights_yes", []), dtype=float)

    if counts.size == 0 or counts.shape != weights.shape:
        return

    annotations = np.empty(counts.shape, dtype=object)
    annotations[:] = ""
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            n = int(counts[i, j])
            w = int(weights[i, j])
            if n > 0:
                annotations[i, j] = f"{n}\nW:{w}"

    cmap = _get_pastel_spectral_cmap()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    masked_counts = np.ma.masked_where(counts == 0, counts)

    fig, ax = _create_figure("wide")
    sns.heatmap(
        masked_counts,
        cmap=cmap,
        linewidths=0.8,
        linecolor="white",
        annot=annotations,
        fmt="",
        annot_kws={"size": VIS_CONFIG["small_text_size"], "weight": "bold"},
        cbar_kws={"shrink": 0.9, "label": "Study count"},
        ax=ax,
    )

    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xlabel("Data availability", fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel("Code availability", fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)
    ax.tick_params(axis="both", labelsize=VIS_CONFIG["tick_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_gold_standard_performance_boxplot(
    points: list[dict[str, Any]],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    title: str = "Performance by gold-standard strength",
):
    """Create domain-split boxplots of performance by gold-standard strength."""
    if not points:
        return

    df = pd.DataFrame(points)
    if df.empty:
        return

    strength_order = [
        "Neuropathological",
        "Clinical + biomarker",
        "Clinical only",
        "Unclear / missing",
    ]
    domain_order = ["ID", "OOD"]

    fig, ax = _create_figure("wide")
    palette = {
        "ID": "#355C7D",
        "OOD": "#7AA5C6",
    }

    sns.boxplot(
        data=df,
        x="strength",
        y="performance",
        hue="domain",
        order=[s for s in strength_order if s in set(df["strength"])],
        hue_order=[d for d in domain_order if d in set(df["domain"])],
        palette=palette,
        linewidth=1.0,
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="strength",
        y="performance",
        hue="domain",
        order=[s for s in strength_order if s in set(df["strength"])],
        hue_order=[d for d in domain_order if d in set(df["domain"])],
        dodge=True,
        size=3,
        alpha=0.45,
        palette=palette,
        ax=ax,
    )

    # Remove duplicated legend caused by overlaying stripplot
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            [unique[label] for label in unique.keys()],
            list(unique.keys()),
            loc="best",
            frameon=VIS_CONFIG["legend_frameon"],
            fontsize=VIS_CONFIG["legend_size"],
            framealpha=VIS_CONFIG["legend_framealpha"],
            title="Domain",
        )

    ax.set_xlabel("Gold-standard strength", fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel("Reported performance (%)", fontsize=VIS_CONFIG["label_size"])
    _apply_optional_title(ax, show_titles, title)
    _style_axes(ax, grid_axis="y")
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_ci_width_boxplot(
    ci_widths_by_metric: dict[str, list[float]],
    output_path: str,
    show_titles: bool = False,
    output_format: str = "pdf",
):
    """Create a boxplot for CI widths by domain/metric."""
    if not ci_widths_by_metric:
        return

    ordered_items = sorted(
        ci_widths_by_metric.items(),
        key=lambda item: np.median(item[1]) if item[1] else 0,
        reverse=True,
    )
    labels = [item[0] for item in ordered_items]
    values = [item[1] for item in ordered_items]

    fig, ax = _create_figure("detailed")
    bp = ax.boxplot(
        values,
        patch_artist=True,
        labels=labels,
        showfliers=False,
    )

    colors = _get_categorical_colors(max(len(labels), 1))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.6)
        box.set_edgecolor("white")

    ax.set_xlabel("Domain | Metric", fontsize=VIS_CONFIG["label_size"])
    ax.set_ylabel("CI width (±)", fontsize=VIS_CONFIG["label_size"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _apply_optional_title(ax, show_titles, "Confidence interval widths by metric")
    _style_axes(ax, grid_axis="y")

    plt.tight_layout()
    _save_matplotlib_figure(output_path, output_format)
    plt.close(fig)


def create_performance_over_time_plots(
    performance_data: list[dict[str, Any]],
    output_folder: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    target_metric: str = "ACC",
    domain_filter: Optional[str] = None,
    min_studies_per_comparison: int = 3,
    max_plots: int = 10,
    label_points_with_authors: bool = False,
    font_scale: float = 1.0,
):
    """
    Creates separate scatter plots for each significant comparison type showing performance trends over time.

    Args:
        performance_data: list of performance records
        output_folder: Folder to save the plots
        show_titles: Whether to show titles
        output_format: Output format
        target_metric: Metric to focus on (default: 'ACC')
        min_studies_per_comparison: Minimum studies required for a comparison to be plotted
        max_plots: Maximum number of plots to create (most common comparisons)
        label_points_with_authors: Whether to label data points with author names
    """
    if not performance_data:
        return

    metric_data = [d for d in performance_data if d["metric"] == target_metric]
    if domain_filter is not None:
        metric_data = [
            d
            for d in metric_data
            if str(d.get("domain", "ID")).upper() == str(domain_filter).upper()
        ]

    if not metric_data:
        return

    metric_data = [d for d in metric_data if d.get("comparison")]
    if not metric_data:
        return

    comparison_counts = Counter(d["comparison"] for d in metric_data)
    viable_comparisons = [
        (comp, count)
        for comp, count in comparison_counts.most_common()
        if count >= min_studies_per_comparison
    ]

    if not viable_comparisons:
        return

    viable_comparisons = viable_comparisons[:max_plots]

    # Calculate global ranges for consistent scaling across all plots
    all_performances = [d["performance"] for d in metric_data]
    all_dataset_sizes = [d["dataset_size"] for d in metric_data]

    # Determine global y-axis range based on metric type
    if target_metric in ["ACC", "SENSITIVITY", "SPECIFICITY", "BACC", "AUC"]:
        global_y_min = max(40, min(all_performances) - 5)  # Don't go below 40%
        global_y_max = min(105, max(all_performances) + 5)  # Don't go above 105%
    else:
        global_y_min = 0
        global_y_max = max(all_performances) * 1.1

    # Calculate size scaling parameters for consistent circle sizes across plots
    min_dataset_size = min(all_dataset_sizes)
    max_dataset_size = max(all_dataset_sizes)

    # Size scaling with multiplication factor for better visibility - increased multiplier
    size_multiplier = 5.0  # Increased from 3.0 to make markers bigger
    min_circle_size = 40 * size_multiplier  # Increased minimum circle size
    max_circle_size = 400 * size_multiplier  # Increased maximum circle size

    def calculate_circle_size(dataset_size):
        if min_dataset_size == max_dataset_size:
            return min_circle_size
        # Linear scaling between min and max circle sizes
        normalized = (dataset_size - min_dataset_size) / (
            max_dataset_size - min_dataset_size
        )
        return min_circle_size + normalized * (max_circle_size - min_circle_size)

    for _i, (comparison, _) in enumerate(viable_comparisons):
        # Filter data for this specific comparison
        plot_data = [d for d in metric_data if d["comparison"] == comparison]

        if not plot_data:
            continue

        safe_filename = _sanitize_filename_component(comparison)
        output_path = os.path.join(output_folder, f"{target_metric}_{safe_filename}")

        # Remove stale case/style variants (e.g., FTD_Subtypes vs. FTD_subtypes).
        safe_key = _filename_component_key(safe_filename)
        for fmt in _normalize_output_formats(output_format):
            ext = f".{str(fmt).lower()}"
            if not os.path.isdir(output_folder):
                continue
            for existing_name in os.listdir(output_folder):
                if not existing_name.lower().endswith(ext):
                    continue
                stem = existing_name[: -len(ext)]
                if "_" not in stem:
                    continue
                existing_metric, existing_component = stem.split("_", 1)
                if existing_metric.upper() != str(target_metric).upper():
                    continue
                if (
                    _filename_component_key(existing_component) == safe_key
                    and existing_component != safe_filename
                ):
                    os.remove(os.path.join(output_folder, existing_name))

        legacy_safe_filename = (
            comparison.replace(" vs. ", "_vs_").replace(" ", "_").replace("/", "_")
        )
        if legacy_safe_filename != safe_filename:
            legacy_output_path = os.path.join(
                output_folder,
                f"{target_metric}_{legacy_safe_filename}",
            )
            for fmt in _normalize_output_formats(output_format):
                legacy_path = ensure_correct_extension(legacy_output_path, fmt)
                if os.path.exists(legacy_path):
                    os.remove(legacy_path)

        fig, ax = _create_figure("detailed")

        # Keep method encoding consistent with ml_dl_trends:
        # DL = green circle, Hybrid = orange square, ML = blue triangle.
        method_symbols = {
            "DL": "o",
            "Hybrid (ML+DL)": "s",
            "ML": "^",
        }

        trend_colors = VIS_CONFIG.get("method_trend_colors", {})
        method_colors = {
            "DL": trend_colors.get("DL", "#7BC8A4"),
            "Hybrid (ML+DL)": trend_colors.get("Hybrid (ML+DL)", "#F2B36D"),
            "ML": trend_colors.get("ML", "#6FA8DC"),
        }

        # Plot data points grouped by method
        methods_plotted = []
        trend_text = ""  # Initialize trend text variable

        for method in ["ML", "DL", "Hybrid (ML+DL)"]:
            method_data = [
                d for d in plot_data if normalize_method_label(d["method"]) == method
            ]
            if not method_data:
                continue

            methods_plotted.append(method)
            years = [d["year"] for d in method_data]
            performances = [d["performance"] for d in method_data]
            # Use the new consistent circle size calculation
            sizes = [calculate_circle_size(d["dataset_size"]) for d in method_data]

            ax.scatter(
                years,
                performances,
                s=sizes,
                marker=method_symbols[method],
                color=method_colors[method],
                alpha=VIS_CONFIG["bar_alpha"],
                edgecolors="white",
                linewidth=1.0,
                label=method,
                zorder=3,
            )

            if label_points_with_authors:
                for j, data_point in enumerate(method_data):
                    author_label = _extract_first_author_name(
                        data_point.get("authors", "")
                    )
                    if author_label:
                        ax.annotate(
                            author_label,
                            (years[j], performances[j]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=VIS_CONFIG["small_text_size"] * font_scale,
                            alpha=0.9,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.8,
                                edgecolor="gray",
                                linewidth=0.5,
                            ),
                            zorder=4,
                        )

        all_years_plot = [d["year"] for d in plot_data]
        all_performances_plot = [d["performance"] for d in plot_data]
        all_weights = [d["dataset_size"] for d in plot_data]

        if len(set(all_years_plot)) > 1:
            x = np.array(all_years_plot)
            y = np.array(all_performances_plot)
            w = np.array(all_weights)

            w = w / np.sum(w) * len(w)

            X = np.column_stack([np.ones(len(x)), x])
            W = np.diag(w)

            try:
                beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)

                trend_years = np.array(sorted(set(all_years_plot)))
                trend_performance = beta[0] + beta[1] * trend_years

                ax.plot(
                    trend_years,
                    trend_performance,
                    linestyle="--",
                    color="gray",
                    alpha=0.8,
                    linewidth=VIS_CONFIG["line_width"],
                    zorder=2,
                    label="Weighted trend",
                )

                slope = beta[1]
                r_squared = _calculate_weighted_r_squared(x, y, w, beta)

                trend_text = f"Trend: {slope:+.2f}%/year (R²={r_squared:.3f})"

            except np.linalg.LinAlgError:
                # Fallback to unweighted regression if weighted fails
                z = np.polyfit(all_years_plot, all_performances_plot, 1)
                p = np.poly1d(z)
                trend_years = sorted(set(all_years_plot))
                ax.plot(
                    trend_years,
                    p(trend_years),
                    linestyle="--",
                    color="gray",
                    alpha=0.8,
                    linewidth=3,
                    zorder=2,
                    label="Trend (unweighted)",
                )

        ax.set_xlabel("Year", fontsize=VIS_CONFIG["label_size"] * font_scale)

        if target_metric in ["ACC", "SENSITIVITY", "SPECIFICITY", "BACC", "AUC"]:
            ax.set_ylabel(
                f"{target_metric} (%)", fontsize=VIS_CONFIG["label_size"] * font_scale
            )
        else:
            ax.set_ylabel(
                f"{target_metric} performance",
                fontsize=VIS_CONFIG["label_size"] * font_scale,
            )

        # Apply consistent y-axis range across all plots (but let x-axis be natural for each plot)
        ax.set_ylim(global_y_min, global_y_max)

        if show_titles:
            title_suffix = " (with author labels)" if label_points_with_authors else ""
            domain_suffix = (
                f" [{str(domain_filter).upper()}]" if domain_filter is not None else ""
            )
            _apply_optional_title(
                ax,
                show_titles,
                f"{target_metric}{domain_suffix} performance over time\n{comparison}{title_suffix}",
            )

        legend_elements = []
        for method in methods_plotted:
            legend_elements.append(
                plt.scatter(
                    [],
                    [],
                    marker=method_symbols[method],
                    color=method_colors[method],
                    s=90,
                    alpha=VIS_CONFIG["bar_alpha"],
                    edgecolors="white",
                    linewidth=1,
                    label=method,
                )
            )

        if legend_elements:
            legend1 = ax.legend(
                handles=legend_elements,
                title="Method type",
                loc="upper left",
                frameon=VIS_CONFIG["legend_frameon"],
                framealpha=VIS_CONFIG["legend_framealpha"],
                fontsize=VIS_CONFIG["legend_size"] * font_scale,
                title_fontsize=VIS_CONFIG["legend_size"] * font_scale,
            )

        size_legend_elements = []
        dataset_sizes = [d["dataset_size"] for d in plot_data]
        if dataset_sizes and len(set(dataset_sizes)) > 1:
            min_size, max_size = min(dataset_sizes), max(dataset_sizes)
            size_values = []
            if min_size != max_size:
                # Show 3-4 representative sizes
                size_values = [min_size, (min_size + max_size) // 2, max_size]
                # Add intermediate value if range is large
                if max_size - min_size > 1000:
                    size_values.insert(1, min_size + (max_size - min_size) // 4)
                    size_values.insert(-1, min_size + 3 * (max_size - min_size) // 4)
            else:
                size_values = [min_size]

            for size_val in size_values:
                if size_val > 0:
                    display_size = (
                        calculate_circle_size(size_val) / 2.5
                    )  # Adjusted scaling for legend
                    size_legend_elements.append(
                        plt.scatter(
                            [],
                            [],
                            s=display_size,
                            color="gray",
                            alpha=0.7,
                            edgecolors="white",
                            linewidth=1,
                            label=f"{int(size_val)} subjects",
                        )
                    )

        if size_legend_elements:
            # Position dataset size legend at bottom right
            ax.legend(
                handles=size_legend_elements,
                title="Dataset size",
                loc="lower right",  # Changed from "upper right" to "lower right"
                frameon=VIS_CONFIG["legend_frameon"],
                framealpha=VIS_CONFIG["legend_framealpha"],
                fontsize=VIS_CONFIG["legend_size"] * font_scale,
                title_fontsize=VIS_CONFIG["legend_size"] * font_scale,
            )
            if legend_elements:
                ax.add_artist(legend1)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
        _style_axes(
            ax,
            grid_linestyle="--",
            tick_size=VIS_CONFIG["tick_size"] * font_scale,
        )

        # Position study info and trend text at bottom left
        total_subjects = sum(d["dataset_size"] for d in plot_data)
        study_info_text = (
            f"n = {len(plot_data)} studies\nTotal subjects = {total_subjects:,}"
        )

        # Add trend text below study info if available
        if trend_text:
            combined_text = f"{study_info_text}\n{trend_text}"
        else:
            combined_text = study_info_text

        ax.text(
            0.02,
            0.02,  # Changed to bottom position
            combined_text,
            transform=ax.transAxes,
            fontsize=VIS_CONFIG["small_text_size"] * font_scale * 1.15,
            verticalalignment="bottom",  # Changed from "top" to "bottom"
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=VIS_CONFIG["annotation_alpha"],
                edgecolor="gray",
            ),
            weight="normal",
        )

        plt.tight_layout()

        _save_matplotlib_figure(output_path, output_format)
        plt.close(fig)


def create_forest_plots_for_common_comparisons(
    performance_data: list[dict[str, Any]],
    output_folder: str,
    show_titles: bool = False,
    output_format: str = "pdf",
    min_studies_per_comparison: int = 5,
    include_ood: bool = False,
    include_single_class: bool = False,
    excluded_metrics: Optional[Sequence[str]] = None,
    font_scale: float = 1.0,
):
    """Create one forest plot per frequent differential comparison."""
    if not performance_data:
        return

    excluded = {str(metric).strip().upper() for metric in (excluded_metrics or ["NPV"])}

    filtered = [record for record in performance_data if record.get("comparison")]
    if not include_single_class:
        filtered = [
            record
            for record in filtered
            if " vs. " in str(record.get("comparison", ""))
        ]
    if not include_ood:
        filtered = [
            record
            for record in filtered
            if str(record.get("domain", "ID")).upper() == "ID"
        ]
    filtered = [
        record
        for record in filtered
        if str(record.get("metric", "")).strip().upper() not in excluded
    ]

    if not filtered:
        return

    studies_by_comparison: dict[str, set] = {}
    for record in filtered:
        comparison = str(record.get("comparison", "")).strip()
        if not comparison:
            continue

        if comparison not in studies_by_comparison:
            studies_by_comparison[comparison] = set()

        study_key = record.get("source_row_index")
        if study_key is None:
            study_key = (
                str(record.get("authors", "")).strip(),
                record.get("year"),
            )
        studies_by_comparison[comparison].add(study_key)

    frequent_comparisons = [
        (comparison, len(study_set))
        for comparison, study_set in studies_by_comparison.items()
        if len(study_set) >= min_studies_per_comparison
    ]
    frequent_comparisons.sort(key=lambda item: (-item[1], item[0]))

    if not frequent_comparisons:
        return

    metric_priority = [
        "AUC",
        "ACC",
        "BACC",
        "SENSITIVITY",
        "SPECIFICITY",
        "F1",
        "PRECISION",
        "NPV",
        "MCC",
    ]

    os.makedirs(output_folder, exist_ok=True)

    for comparison, n_studies in frequent_comparisons:
        comparison_records = [
            record for record in filtered if record.get("comparison") == comparison
        ]
        if not comparison_records:
            continue

        grouped_records: dict[tuple, list[dict[str, Any]]] = {}
        for record in comparison_records:
            study_key = record.get("source_row_index")
            if study_key is None:
                study_key = (
                    str(record.get("authors", "")).strip(),
                    record.get("year"),
                )
            metric = str(record.get("metric", "")).strip().upper()
            domain = str(record.get("domain", "ID")).strip().upper()
            key = (study_key, metric, domain)
            grouped_records.setdefault(key, []).append(record)

        aggregated = []
        for (study_key, metric, domain), records in grouped_records.items():
            performances = [float(item.get("performance")) for item in records]
            ci_values = [
                float(item.get("ci")) for item in records if item.get("ci") is not None
            ]

            first = records[0]
            year = first.get("year")
            year_label = str(int(year)) if year is not None else "n.d."
            first_author = _extract_first_author_name(first.get("authors", ""))
            if not first_author:
                first_author = "Unknown"

            aggregated.append(
                {
                    "study_key": study_key,
                    "study_label": f"{first_author} ({year_label})",
                    "year": year,
                    "metric": metric,
                    "domain": domain,
                    "performance": float(np.mean(performances)),
                    "ci": float(np.mean(ci_values)) if ci_values else None,
                }
            )

        if not aggregated:
            continue

        study_meta: dict[Any, dict[str, Any]] = {}
        for row in aggregated:
            key = row["study_key"]
            existing = study_meta.get(key)
            if existing is None:
                study_meta[key] = {"year": row["year"], "label": row["study_label"]}
                continue

            existing_year = existing.get("year")
            current_year = row.get("year")
            if existing_year is None and current_year is not None:
                existing["year"] = current_year

        sorted_studies = sorted(
            study_meta.items(),
            key=lambda item: (
                item[1]["year"] if item[1]["year"] is not None else 9999,
                item[1]["label"],
            ),
        )

        if not sorted_studies:
            continue

        y_positions = {
            study_key: idx for idx, (study_key, _) in enumerate(sorted_studies)
        }
        y_tick_labels = [meta["label"] for _study_key, meta in sorted_studies]

        metrics_present = sorted(
            {row["metric"] for row in aggregated},
            key=lambda metric: (
                metric_priority.index(metric)
                if metric in metric_priority
                else len(metric_priority),
                metric,
            ),
        )
        if not metrics_present:
            continue

        offsets = (
            np.linspace(-0.28, 0.28, len(metrics_present))
            if len(metrics_present) > 1
            else np.array([0.0])
        )
        palette = sns.color_palette("Set2", len(metrics_present)).as_hex()

        fig_height = max(4.8, min(14.5, 2.4 + 0.44 * len(sorted_studies)))
        fig, ax = plt.subplots(figsize=(10.8, fig_height), dpi=VIS_CONFIG["dpi"])

        for index, metric in enumerate(metrics_present):
            metric_rows = [row for row in aggregated if row["metric"] == metric]
            if not metric_rows:
                continue

            x = [row["performance"] for row in metric_rows]
            y = [y_positions[row["study_key"]] + offsets[index] for row in metric_rows]
            xerr = np.array(
                [
                    row["ci"] if row.get("ci") is not None else 0.0
                    for row in metric_rows
                ],
                dtype=float,
            )

            ax.errorbar(
                x,
                y,
                xerr=xerr if np.any(xerr > 0) else None,
                fmt="o",
                markersize=5.5,
                color=palette[index],
                ecolor=palette[index],
                elinewidth=1.0,
                capsize=2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                alpha=0.9,
                label=metric,
                zorder=3,
            )

        values = [row["performance"] for row in aggregated]
        x_min = max(0.0, min(values) - 8.0)
        x_max = min(100.0, max(values) + 8.0)
        if x_max - x_min < 20:
            center = (x_min + x_max) / 2.0
            x_min = max(0.0, center - 10)
            x_max = min(100.0, center + 10)

        ax.set_xlim(x_min, x_max)
        ax.set_yticks(list(range(len(sorted_studies))))
        ax.set_yticklabels(y_tick_labels, fontsize=VIS_CONFIG["tick_size"] * font_scale)
        ax.invert_yaxis()
        ax.set_xlabel("Performance (%)", fontsize=VIS_CONFIG["label_size"] * font_scale)
        ax.set_ylabel("Study", fontsize=VIS_CONFIG["label_size"] * font_scale)

        if show_titles:
            _apply_optional_title(
                ax,
                show_titles,
                f"{comparison} (N={n_studies} studies)",
            )

        _style_axes(
            ax,
            grid_axis="x",
            tick_size=VIS_CONFIG["tick_size"] * font_scale,
        )
        ax.axvline(50, color="#BDBDBD", linestyle="--", linewidth=0.8, zorder=1)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="lower right",
                frameon=VIS_CONFIG["legend_frameon"],
                fontsize=VIS_CONFIG["legend_size"] * font_scale,
                framealpha=VIS_CONFIG["legend_framealpha"],
                title="Metric",
            )

        plt.tight_layout()
        safe_name = _sanitize_filename_component(comparison)
        output_path = os.path.join(output_folder, f"forest_{safe_name}")
        _save_matplotlib_figure(output_path, output_format)
        plt.close(fig)


def _calculate_weighted_r_squared(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray
) -> float:
    """
    Calculate weighted R-squared for the regression.

    Args:
        x: Independent variable (years)
        y: Dependent variable (performance)
        w: Weights (dataset sizes)
        beta: Regression coefficients [intercept, slope]

    Returns:
        Weighted R-squared value
    """
    y_pred = beta[0] + beta[1] * x

    y_mean = np.average(y, weights=w)

    ss_res = np.sum(w * (y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum(w * (y - y_mean) ** 2)  # Total sum of squares

    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _extract_first_author_name(authors_string: str) -> str:
    """
    Extracts the last name of the first author from an authors string.

    Args:
        authors_string: String containing author names separated by commas

    Returns:
        Formatted author name with "et al." or empty string if no valid authors
    """
    if not authors_string or pd.isna(authors_string):
        return ""

    authors_string = str(authors_string).strip()
    if not authors_string:
        return ""

    authors = [author.strip() for author in authors_string.split(",")]

    if not authors or not authors[0]:
        return ""

    first_author = authors[0].strip()

    if "," in first_author:
        last_name = first_author.split(",")[0].strip()
    else:
        name_parts = first_author.split()
        if name_parts:
            last_name = name_parts[-1].strip()
        else:
            return ""

    if len(authors) > 1:
        return f"{last_name} et al."
    else:
        return last_name
