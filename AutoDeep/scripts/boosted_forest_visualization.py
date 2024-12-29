import pandas as pd 
import xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import os



base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
model_path = os.path.join(base_path, "model_weights/miRNA_model.json")

model = XGBClassifier()
model.load_model(model_path)

graph = xgboost.plot_importance(model, importance_type = 'gain', values_format = '{v:.2f}', xlabel = "Gain")
current_dir = os.getcwd()

plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")


import graphviz


#0 is Candidate, 2 is falsepositive, 1 is Confident
for i in range(model.get_booster().num_boosted_rounds()):
    dot_data = model.get_booster().get_dump(dump_format="dot")[i]

    # Render the DOT data to a PNG
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_plot{i}", format="png")