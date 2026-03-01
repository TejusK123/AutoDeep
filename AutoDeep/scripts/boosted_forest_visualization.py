import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        prog="AutoDeep visualize",
        description="Generate visualizations of model performance and feature importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  Generates visualizations and analyses of the trained model including:
  - Feature importance plots (gain, weight, cover)
  - Individual decision tree plots

REQUIREMENTS:
  - Must be run from within an AutoDeepRun directory
  - AutoDeepRun/formatted_novel_miRNA.csv must exist

OUTPUT:
  Visualizations are saved in timestamped directory:
  - tree_plots<timestamp>/feature_importance_*.png
  - tree_plots<timestamp>/tree_plot*.png

EXAMPLES:
  cd AutoDeepRun
  AutoDeep visualize
  AutoDeep visualize --no_tree     (skip tree plots)
  AutoDeep visualize -o my_plots   (custom output directory)
        """
    )
    
    parser.add_argument(
        "--no_tree",
        action="store_true",
        help="Skip generating individual decision tree plots (faster)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tree_plots",
        help="Output directory for visualizations (default: tree_plots)",
        metavar="<directory>"
    )
    
    args = parser.parse_args()
    visualize(args)


def visualize(args):
    import xgboost
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import time
     
    print("Starting model visualization")
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    model_path = os.path.join(base_path, "model_weights/miRNA_model.json")

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
        print("Skipping tree plot generation")
    else:
        import graphviz
        num_rounds = model.get_booster().num_boosted_rounds()
        print(f"Generating {num_rounds} decision tree plots...")
        
        for i in tqdm(range(num_rounds)):
            dot_data = model.get_booster().get_dump(dump_format="dot")[i]
            graph = graphviz.Source(dot_data)
            graph.render(f"{args.output}{timestamp}/tree_plot{i}", format="png")

    print(f"Visualizations saved to: {args.output}{timestamp}/")


if __name__ == "__main__":
    main()
