import pandas as pd 
import xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
import time

if len(sys.argv) > 1:
    print("Usage: AutoDeep visualize (no arguments)")
    sys.exit(1)

current_dir = os.getcwd()
if "AutoDeepRun" not in os.path.basename(current_dir):
    print("Please run in AutoDeepRun directory")
    sys.exit(1)


base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
model_path = os.path.join(base_path, "model_weights/miRNA_model.json")

model = XGBClassifier()
model.load_model(model_path)

timestamp = str(round(time.time()))
try:
    os.mkdir(f"tree_plots_{timestamp}")
except FileExistsError:
    print("Directory already exists: Overwriting files")


graph_gain = xgboost.plot_importance(model, importance_type = 'gain', values_format = '{v:.2f}', xlabel = "Gain", title = "Feature Importance by Gain")
plt.savefig(f"tree_plots_{timestamp}/feature_importance_gain.png", dpi=300, bbox_inches="tight")
graph_weight = xgboost.plot_importance(model, importance_type = 'weight', values_format = '{v:.2f}', xlabel = "Weight", title = "Feature Importance by Weight")
plt.savefig(f"tree_plots_{timestamp}/feature_importance_weight.png", dpi=300, bbox_inches="tight")
graph_cover = xgboost.plot_importance(model, importance_type = 'cover', values_format = '{v:.2f}', xlabel = "Cover", title = "Feature Importance by Cover")
plt.savefig(f"tree_plots_{timestamp}/feature_importance_cover.png", dpi=300, bbox_inches="tight")

import graphviz


#0 is Candidate, 2 is falsepositive, 1 is Confident
num_rounds = model.get_booster().num_boosted_rounds()
for i in tqdm(range(num_rounds)):
    dot_data = model.get_booster().get_dump(dump_format="dot")[i]

    # Render the DOT data to a PNG
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_plots_{timestamp}/tree_plot{i}", format="png")