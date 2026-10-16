import argparse
import json
import os
from tqdm import tqdm

from features import LABELS


def main():
    parser = argparse.ArgumentParser(
        prog="AutoDeep visualize",
        description="Generate visualizations of model performance and feature importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  Generates visualizations and analyses of the trained model including:
  - Feature importance plots (gain, weight, cover)
  - Decision tree plots, if --trees is given

OUTPUT:
  Visualizations are saved in timestamped directory:
  - tree_plots<timestamp>/feature_importance_*.png
  - tree_plots<timestamp>/tree_round<NNNN>_<class>.png  (with --trees)

EXAMPLES:
  AutoDeep visualize
  AutoDeep visualize --trees 5      (trees of the first 5 rounds, one per class)
  AutoDeep visualize --trees all    (every tree; slow for large models)
  AutoDeep visualize -o my_plots    (custom output directory)
        """
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
        default="tree_plots",
        help="Output directory for visualizations (default: tree_plots)",
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


def visualize(args):
    import xgboost
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import time

    print("Starting model visualization")
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    model_path = args.model or os.path.join(base_path, "model_weights/miRNA_model.json")

    timestamp = str(round(time.time()))
    try:
        os.mkdir(f"{args.output}{timestamp}")
    except FileExistsError:
        print("Directory already exists: Overwriting files")

    model = XGBClassifier()
    model.load_model(model_path)
    graph_gain = xgboost.plot_importance(model, importance_type='gain', values_format='{v:.2f}', xlabel="Gain", title="Feature Importance by Gain")
    plt.savefig(f"{args.output}{timestamp}/feature_importance_gain.png", dpi=300, bbox_inches="tight")
    graph_weight = xgboost.plot_importance(model, importance_type='weight', values_format='{v:.2f}', xlabel="Weight", title="Feature Importance by Weight")
    plt.savefig(f"{args.output}{timestamp}/feature_importance_weight.png", dpi=300, bbox_inches="tight")
    graph_cover = xgboost.plot_importance(model, importance_type='cover', values_format='{v:.2f}', xlabel="Cover", title="Feature Importance by Cover")
    plt.savefig(f"{args.output}{timestamp}/feature_importance_cover.png", dpi=300, bbox_inches="tight")

    if args.no_tree:
        print("Note: --no_tree is no longer needed; tree plots are only made with --trees")
    if args.trees:
        booster = model.get_booster()
        rounds = None if args.trees == "all" else int(args.trees)
        n_rounds = booster.num_boosted_rounds() if rounds is None else min(rounds, booster.num_boosted_rounds())
        print(f"Plotting the decision trees of {n_rounds} rounds, one per class...")
        export_trees(booster, f"{args.output}{timestamp}", rounds)

    print(f"Visualizations saved to: {args.output}{timestamp}/")


if __name__ == "__main__":
    main()
