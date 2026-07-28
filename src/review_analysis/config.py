import logging
import os
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

from .normalization_loader import get_normalization_config


# Monochrome blue palette for publication-style consistency.
BLUE_SERIES = [
    "#08306B",
    "#08519C",
    "#2171B5",
    "#4292C6",
    "#6BAED6",
    "#9ECAE1",
    "#C6DBEF",
    "#DEEBF7",
]

ML_METHOD_COLORS = {
    "ML": BLUE_SERIES[1],
    "DL": BLUE_SERIES[3],
    "Hybrid (ML+DL)": BLUE_SERIES[5],
}

FIGURE_PROFILES = {
    "standard": (6.2, 3.2),
    "wide": (8.4, 4.8),
    "square": (6.0, 6.0),
    "detailed": (9.0, 6.0),
}


VIS_CONFIG = {
    "figure_size": FIGURE_PROFILES["standard"],
    "dpi": 400,
    # Prefer perceptually-uniform colormaps for accessibility
    "color_palette": "Blues",
    "color_palette_reversed": "Blues_r",
    "categorical_colors": BLUE_SERIES,
    "accent_palette": BLUE_SERIES,
    "sequential_palette": "Blues",
    # Distinct pastel colors for 3-method trend plots (ML / DL / Hybrid)
    "method_trend_colors": {
        "ML": "#6FA8DC",  # pastel blue
        "DL": "#7BC8A4",  # pastel green
        "Hybrid (ML+DL)": "#F2B36D",  # pastel orange
    },
    # Sankey uses a distinct pastel triad to avoid confusion with ML/DL/Hybrid colors.
    "sankey_column_colors": {
        "disease": "#B8A1D9",  # pastel lavender
        "dataset": "#E7A3B1",  # soft pastel rose
        "arch": "#D9C08A",  # muted pastel sand
    },
    # Improve readability of ML/DL pie with clear dark/medium/light blue separation.
    "ml_method_pie_colors": {
        "ML": "#0B3C6F",
        "DL": "#3F7FB6",
        "Hybrid (ML+DL)": "#A7C7E7",
    },
    # Single blue for all bar charts/histograms.
    "fixed_bar_color": "#4C78A8",
    "diverging_palette": "vlag",
    # Matplotlib will be forced to LaTeX via rcParams when available
    "font_family": "serif",
    "label_size": 11,
    "title_size": 12,
    "tick_size": 10,
    "legend_size": 9,
    "value_label_size": 9,
    "small_text_size": 8,
    "line_width": 1.8,
    "marker_size": 7,
    "lollipop_marker_size": 40,
    "grid_alpha": 0.18,
    "grid_linewidth": 0.5,
    "bar_width": 0.62,
    "barplot_fixed_height": 4.1,
    "edge_width": 0.8,
    "bar_alpha": 0.72,
    "wedge_width": 0.3,
    "edge_color": "white",
    "pct_distance": 0.76,
    "spine_linewidth": 0.9,
    "tick_major_size": 4,
    "tick_minor_size": 2,
    "legend_frameon": True,
    "legend_fancybox": True,
    "legend_shadow": False,
    "legend_framealpha": 0.86,
    "annotation_alpha": 0.82,
    "small_slice_threshold_pct": 3.5,
    # Pastel qualitative palette (ColorBrewer Set3-inspired) for comparison-type pies.
    # Chosen for clear category separation when many slices are present (e.g., 11+).
    "comparison_types_pastel_colors": [
        "#8DD3C7",
        "#FFFFB3",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        "#D9D9D9",
        "#BC80BD",
        "#CCEBC5",
        "#FFED6F",
    ],
    # Soft but more contrasted gradient stops for comparison-type pie.
    "comparison_types_gradient_stops": [
        "#E6B7CF",  # dusty pastel pink
        "#C7B8E8",  # pastel lavender
        "#AECBED",  # pastel sky blue
        "#ABDCC8",  # pastel mint
    ],
}

ANALYSIS_CONFIG = {
    "min_dataset_frequency": 2,
    "min_disease_frequency": 2,
    "network_min_node_frequency": 2,
    "network_min_edge_weight": 2,
    "network_min_disease_frequency": 2,
    "min_metric_frequency": 2,
    "comparison_small_slice_threshold_pct": 2.0,
    "min_studies_per_comparison": 3,
    "max_performance_plots": 15,
}

NORMALIZATION = get_normalization_config()

OUTPUT_FORMATS = ["pdf", "png"]

# Required fields for data validation
REQUIRED_FIELDS = ["Year", "ML / DL"]


def set_scientific_style():
    """Set scientific paper-like style for plots with enhanced aesthetics."""
    # Suppress noisy per-glyph font subsetting logs during vector export.
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

    use_tex_env = os.environ.get("REVIEW_USE_TEX", "0").strip().lower()
    use_tex = (
        use_tex_env not in {"0", "false", "no"} and shutil.which("latex") is not None
    )

    plt.style.use("default")
    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "axes.linewidth": VIS_CONFIG["spine_linewidth"],
            "grid.linewidth": VIS_CONFIG["grid_linewidth"],
            "grid.alpha": VIS_CONFIG["grid_alpha"],
            "axes.edgecolor": "0.15",
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    # Set context for publication
    sns.set_context("paper", font_scale=1.0)

    # Configure matplotlib for publication quality
    plt.rcParams.update(
        {
            # LaTeX typography
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
                "serif",
            ],
            "mathtext.fontset": "cm",
            "text.latex.preamble": (
                r"\usepackage{lmodern}\usepackage{amsmath,amssymb}" if use_tex else ""
            ),
            "font.size": VIS_CONFIG["tick_size"],
            # Axes settings - less bold
            "axes.labelsize": VIS_CONFIG["label_size"],
            "axes.titlesize": VIS_CONFIG["title_size"],
            "axes.linewidth": VIS_CONFIG["spine_linewidth"],
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlepad": 15,
            "axes.labelpad": 8,
            # Tick settings
            "xtick.labelsize": VIS_CONFIG["tick_size"],
            "ytick.labelsize": VIS_CONFIG["tick_size"],
            "xtick.major.size": VIS_CONFIG["tick_major_size"],
            "ytick.major.size": VIS_CONFIG["tick_major_size"],
            "xtick.minor.size": VIS_CONFIG["tick_minor_size"],
            "ytick.minor.size": VIS_CONFIG["tick_minor_size"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend settings
            "legend.fontsize": VIS_CONFIG["legend_size"],
            "legend.frameon": VIS_CONFIG["legend_frameon"],
            "legend.fancybox": VIS_CONFIG["legend_fancybox"],
            "legend.shadow": VIS_CONFIG["legend_shadow"],
            "legend.framealpha": VIS_CONFIG["legend_framealpha"],
            "legend.edgecolor": "0.8",
            # Figure settings
            "figure.dpi": VIS_CONFIG["dpi"],
            "savefig.dpi": VIS_CONFIG["dpi"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.0,
            "savefig.format": "pdf",
            # PDF text as TrueType for better compatibility/editing
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Line settings
            "lines.linewidth": VIS_CONFIG["line_width"],
            "lines.markersize": VIS_CONFIG["marker_size"],
            "lines.markeredgewidth": 0.5,
            # Grid settings
            "grid.alpha": VIS_CONFIG["grid_alpha"],
            "grid.linewidth": VIS_CONFIG["grid_linewidth"],
            "grid.color": "0.8",
        }
    )


def ensure_correct_extension(path: str, format_type: str) -> str:
    """Ensure the output path has the correct file extension."""
    base, ext = os.path.splitext(path)
    if not ext or ext[1:].lower() != format_type.lower():
        return f"{base}.{format_type.lower()}"
    return path
