import click


class CustomUsageMsg(click.Command):
    def format_usage(self, ctx, formatter):
        formatter.write_text("Usage: AutoDeep visualize [OPTIONS]")


@click.command()
@click.option("--no_tree", is_flag = True, help="Do not output tree plots")
@click.option("--output", "-o", default = "tree_plots", help="Output directory for tree plots")
def visualize(no_tree, output, targets):
    import xgboost
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import os
    import time
     
    
    print("Starting model visualization")
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    model_path = os.path.join(base_path, "model_weights/miRNA_model.json")

    timestamp = str(round(time.time()))
    try:
        os.mkdir(f"{output}{timestamp}")
    except FileExistsError:
        print("Directory already exists: Overwriting files")


    model = XGBClassifier()
    model.load_model(model_path)
    graph_gain = xgboost.plot_importance(model, importance_type = 'gain', values_format = '{v:.2f}', xlabel = "Gain", title = "Feature Importance by Gain")
    plt.savefig(f"{output}{timestamp}/feature_importance_gain.png", dpi=300, bbox_inches="tight")
    graph_weight = xgboost.plot_importance(model, importance_type = 'weight', values_format = '{v:.2f}', xlabel = "Weight", title = "Feature Importance by Weight")
    plt.savefig(f"{output}{timestamp}/feature_importance_weight.png", dpi=300, bbox_inches="tight")
    graph_cover = xgboost.plot_importance(model, importance_type = 'cover', values_format = '{v:.2f}', xlabel = "Cover", title = "Feature Importance by Cover")
    plt.savefig(f"{output}{timestamp}/feature_importance_cover.png", dpi=300, bbox_inches="tight")

    
    #0 is Candidate, 2 is falsepositive, 1 is Confident
    if no_tree:
        pass
    else:
        import graphviz
        num_rounds = model.get_booster().num_boosted_rounds()

        with click.progressbar(range(num_rounds), label="Rendering tree plots") as bar:
            for i in bar:
                dot_data = model.get_booster().get_dump(dump_format="dot")[i]

                # Render the DOT data to a PNG
                graph = graphviz.Source(dot_data)
                graph.render(f"tree_plots_{timestamp}/tree_plot{i}", format="png")



if __name__ == "__main__":
    visualize()
