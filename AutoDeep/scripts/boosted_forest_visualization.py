import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from features import LABELS, TABLE1_NAMES, encode_features

# Figure colours (reference data-viz palette): classes take the first three
# categorical slots in fixed order; a feature's value is one blue ramp, light
# (low) to dark (high); blank values are the muted grey; text is ink, never a
# data colour.
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # Candidate, Confident, falsepositive
VALUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, SECONDARY, MUTED, BASELINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"

# Features with a dependence plot, and their file names
DEPENDENCE = {
    "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum": "five_prime_homogeneity",
    "3' Overhang Bot": "overhang_bot",
    "3' Overhang Top": "overhang_top",
    "Folding Energy": "folding_energy",
}
COLUMN_WIDTH, PAGE_WIDTH = 4.5, 6.85  # inches


def main():
    parser = argparse.ArgumentParser(
        prog="AutoDeep visualize",
        description="Explain the model's predictions with SHAP values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  Computes each feature's SHAP value (its contribution to each class's score,
  exact TreeSHAP from XGBoost) for every locus in the data, and plots:
  - one summary plot per class (every locus's SHAP value for every feature)
  - the mean absolute SHAP value of each feature, per class
  - dependence plots for 5' homogeneity, the two overhangs and folding energy
  Text is 8 pt Helvetica (Arial or DejaVu Sans where Helvetica is missing),
  kept as text in the PDF or SVG.

OUTPUT:
  In a timestamped directory, plots<timestamp>/:
  - shap_summary_<class>.pdf, shap_importance.pdf, shap_dependence_<feature>.pdf
  - shap_values.csv      (every locus's SHAP values, one row per locus and class)
  - shap_importance.csv  (the numbers behind shap_importance.pdf)
  - feature_importance_*.png             (with --legacy_importance)
  - tree_round<NNNN>_<class>.png         (with --trees)

EXAMPLES:
  AutoDeep visualize                     (the bundled training data)
  AutoDeep visualize --data AutoDeepRun/fully_formatted_data.csv
  AutoDeep visualize --format svg
  AutoDeep visualize --legacy_importance (also the gain, weight and cover plots)
  AutoDeep visualize --trees 5           (trees of the first 5 rounds, one per class)
        """
    )

    parser.add_argument(
        "--data",
        nargs="+",
        metavar="<csv>",
        help="CSV(s) with the model's features to explain, e.g. AutoDeepRun/fully_formatted_data.csv "
             "(default: the bundled training data)"
    )

    parser.add_argument(
        "--format",
        choices=["pdf", "svg"],
        default="pdf",
        help="File format of the SHAP plots (default: pdf)"
    )

    parser.add_argument(
        "--legacy_importance",
        action="store_true",
        help="Also plot XGBoost's feature importance by gain, weight and cover (PNG), as before C4"
    )

    parser.add_argument(
        "--trees",
        type=str,
        metavar="<N|all>",
        help="Also plot the decision trees of the first N boosting rounds (one tree per class), or 'all'"
    )

    parser.add_argument(
        "--no_tree",
        action="store_true",
        help=argparse.SUPPRESS  # tree plots are off by default now; kept so old commands still run
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="plots",
        help="Output directory base name (default: plots)",
        metavar="<directory>"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model JSON to visualize (default: the bundled model)",
        metavar="<path>"
    )

    args = parser.parse_args()
    if args.trees not in (None, "all") and not (args.trees.isdigit() and int(args.trees) > 0):
        parser.error("--trees takes a positive number of rounds or 'all'")
    visualize(args)


def tree_labels(booster):
    """(round, class name) for each tree, in dump order, from the model's own records."""
    model = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]
    indptr = model["iteration_indptr"]
    labels = []
    for round_index in range(len(indptr) - 1):
        for tree in range(indptr[round_index], indptr[round_index + 1]):
            labels.append((round_index, LABELS[model["tree_info"][tree]]))
    return labels


def export_trees(booster, out_dir, rounds=None):
    """Render the trees of the first `rounds` rounds (all if None) as PNGs; return their paths."""
    import graphviz

    dumps = booster.get_dump(dump_format="dot")
    selected = [(i, r, c) for i, (r, c) in enumerate(tree_labels(booster)) if rounds is None or r < rounds]
    paths = []
    for i, round_index, class_name in tqdm(selected):
        stem = os.path.join(out_dir, f"tree_round{round_index:04d}_{class_name}")
        graphviz.Source(dumps[i]).render(stem, format="png", cleanup=True)
        paths.append(stem + ".png")
    return paths


def legacy_importance(model, out_dir):
    """XGBoost's feature importance by gain, weight and cover, as AutoDeep plotted it before C4."""
    import matplotlib.pyplot as plt
    import xgboost

    with plt.style.context("default"):
        for kind, label in (("gain", "Gain"), ("weight", "Weight"), ("cover", "Cover")):
            xgboost.plot_importance(model, importance_type=kind, values_format='{v:.2f}', xlabel=label,
                                    title=f"Feature Importance by {label}")
            plt.savefig(os.path.join(out_dir, f"feature_importance_{kind}.png"), dpi=300, bbox_inches="tight")
            plt.close("all")


def load_data(paths, names):
    """(locus IDs, the model's features with the flags encoded) from one or more CSVs."""
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [name for name in names if name not in df.columns]
        if missing:
            sys.exit(f"Error: {path} lacks features the model needs: {missing}")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    ids = data["provisional_id"] if "provisional_id" in data.columns else pd.Series(range(len(data)))
    return ids.reset_index(drop=True), encode_features(data)[names]


def set_style():
    import matplotlib as mpl

    # Font fallback and Apple-only font tables are reported at warning level; neither matters here
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none",  # text stays text, in an embedded TrueType font
        "text.color": INK, "axes.labelcolor": SECONDARY, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": INK,
        "axes.edgecolor": BASELINE, "xtick.color": BASELINE, "ytick.color": BASELINE,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def swarm(x, width=0.4, bins=100):
    """Vertical offsets that spread points with similar x into a swarm, at most `width` from the row."""
    offsets = np.zeros(len(x))
    if len(x) == 0:
        return offsets
    span = (x.max() - x.min()) or 1.0
    bin_of = np.minimum(((x - x.min()) / span * bins).astype(int), bins - 1)
    order = np.lexsort((x, bin_of))
    _, starts, counts = np.unique(bin_of[order], return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        layer = np.arange(count)
        offsets[order[start:start + count]] = ((layer + 1) // 2) * np.where(layer % 2, 1, -1)
    peak = np.abs(offsets).max()
    return offsets * (width / peak) if peak else offsets


def summary_plot(values, X, names, k, order, path):
    """Every locus's SHAP value for class k, one row per feature, shaded by the feature's value."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    ramp = LinearSegmentedColormap.from_list("value", VALUE_RAMP)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.2 * len(order) + 0.9))
    any_blank = False
    for row, j in enumerate(order):
        y = len(order) - 1 - row
        shap = values[:, k, j]
        value = X.iloc[:, j]
        blank = value.isna().to_numpy()
        any_blank |= blank.any()
        offsets = swarm(shap)
        rank = value.rank(pct=True).to_numpy()
        ax.scatter(shap[blank], y + offsets[blank], s=1.5, color=MUTED, linewidths=0)
        ax.scatter(shap[~blank], y + offsets[~blank], s=1.5, c=rank[~blank], cmap=ramp, vmin=0, vmax=1, linewidths=0)
    ax.axvline(0, color=BASELINE, linewidth=0.5, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TABLE1_NAMES[names[j]] for j in reversed(order)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"SHAP value (effect on the {LABELS[k]} score, log-odds)")
    ax.set_title(LABELS[k])
    bar = fig.colorbar(ScalarMappable(cmap=ramp), ax=ax, ticks=[0, 1], fraction=0.04, aspect=20, pad=0.02, shrink=0.6)
    bar.ax.set_yticklabels(["Low", "High"])
    bar.set_label("Feature value", color=SECONDARY)
    bar.outline.set_visible(False)
    if any_blank:
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=3, color=MUTED, label="Blank")],
                  loc="lower right", handletextpad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def importance_plot(mean_abs, names, order, path):
    """Mean absolute SHAP value per feature, one bar per class."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.27 * len(order) + 0.9))
    height = 0.26
    for k, label in enumerate(LABELS):
        rows = np.arange(len(order))[::-1] + (1 - k) * height
        ax.barh(rows, mean_abs[k, order], height=height * 0.85, color=CLASS_COLOURS[k], label=label)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TABLE1_NAMES[names[j]] for j in reversed(order)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Mean absolute SHAP value (log-odds)")  # "|" reads as "l" in Helvetica
    ax.legend(loc="lower right", handlelength=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dependence_plot(values, X, names, feature, path):
    """A feature's value against its SHAP value, one panel per class; blank values in their own column."""
    import matplotlib.pyplot as plt

    j = names.index(feature)
    x = X[feature].to_numpy(dtype=float)
    blank = np.isnan(x)
    known = x[~blank]
    lo, hi = (known.min(), known.max()) if len(known) else (0.0, 1.0)
    span = (hi - lo) or 1.0
    # Integer features with few values (the overhangs) get a little sideways jitter, so stacked points show
    discrete = len(known) > 0 and np.all(known == np.round(known)) and len(np.unique(known)) <= 25
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(x)) if discrete else np.zeros(len(x))
    blank_x = lo - 0.15 * span
    fig, axes = plt.subplots(1, len(LABELS), figsize=(PAGE_WIDTH, 2.2), sharey=True)
    for k, ax in enumerate(axes):
        ax.scatter(known + jitter[~blank], values[~blank, k, j], s=2, color=CLASS_COLOURS[k], alpha=0.5, linewidths=0)
        if blank.any():
            ax.scatter(blank_x + jitter[blank], values[blank, k, j], s=2, color=MUTED, alpha=0.5, linewidths=0)
        ax.axhline(0, color=BASELINE, linewidth=0.5, zorder=0)
        ax.set_title(LABELS[k])
        ax.set_xlabel(TABLE1_NAMES[feature])
        if blank.any():
            ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
            ax.set_xticks([blank_x] + ticks)
            ax.set_xticklabels(["Blank"] + [f"{t:g}" for t in ticks])
            ax.set_xlim(blank_x - 0.08 * span, hi + 0.05 * span)
    axes[0].set_ylabel("SHAP value (log-odds)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize(args):
    import time

    import matplotlib
    matplotlib.use("Agg")
    from xgboost import XGBClassifier

    from explain import contributions

    print("Starting model visualization")
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    model_path = args.model or os.path.join(base_path, "model_weights/miRNA_model.json")
    data_paths = args.data or sorted(glob.glob(os.path.join(base_path, "training_data_pre", "*.csv")))

    out_dir = f"{args.output}{round(time.time())}"
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print("Directory already exists: Overwriting files")

    model = XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()
    names = booster.feature_names

    ids, X = load_data(data_paths, names)
    print(f"Computing SHAP values for {len(X)} loci in {', '.join(args.data) if args.data else 'the bundled training data'}")
    values, bias = contributions(booster, X)

    tables = []
    for k, label in enumerate(LABELS):
        table = pd.DataFrame(values[:, k, :], columns=names)
        table.insert(0, "bias", bias[k])
        table.insert(0, "class", label)
        table.insert(0, "provisional_id", ids)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)

    # One feature order for every plot: by mean |SHAP value| over the three classes
    mean_abs = np.abs(values).mean(axis=0)
    order = list(np.argsort(-mean_abs.mean(axis=0), kind="stable"))
    importance = pd.DataFrame({"feature": [names[j] for j in order], "name": [TABLE1_NAMES[names[j]] for j in order],
                               **{label: mean_abs[k, order] for k, label in enumerate(LABELS)}})
    importance.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False)

    set_style()
    ext = args.format
    for k, label in enumerate(LABELS):
        summary_plot(values, X, names, k, order, os.path.join(out_dir, f"shap_summary_{label}.{ext}"))
    importance_plot(mean_abs, names, order, os.path.join(out_dir, f"shap_importance.{ext}"))
    for feature, stem in DEPENDENCE.items():
        dependence_plot(values, X, names, feature, os.path.join(out_dir, f"shap_dependence_{stem}.{ext}"))

    if args.legacy_importance:
        legacy_importance(model, out_dir)
    if args.no_tree:
        print("Note: --no_tree is no longer needed; tree plots are only made with --trees")
    if args.trees:
        rounds = None if args.trees == "all" else int(args.trees)
        n_rounds = booster.num_boosted_rounds() if rounds is None else min(rounds, booster.num_boosted_rounds())
        print(f"Plotting the decision trees of {n_rounds} rounds, one per class...")
        export_trees(booster, out_dir, rounds)

    print(f"Visualizations saved to: {out_dir}/")


if __name__ == "__main__":
    main()
