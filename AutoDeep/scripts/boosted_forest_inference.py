import pandas as pd 
import xgboost
from xgboost import XGBClassifier
import numpy as np
import os
from features import LABELS, encode_features, decode_features, decode_labels


# Loading all relevant paths
base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
model_path = os.path.join(base_path, "model_weights/miRNA_model.json")
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "AutoDeepRun/fully_formatted_data.csv")

# Loading the model
model = XGBClassifier()
model.load_model(model_path)

# Loading the data
merged_data = pd.read_csv(data_dir)

# Human readable to categorical
merged_data = encode_features(merged_data)


#Extract True labels and drop them from the data
labels = merged_data['provisional_id']
merged_data = merged_data.drop(columns = ['provisional_id'])


# Inference
y_pred = model.predict(merged_data.to_numpy())
y_logits = model.predict_proba(merged_data.to_numpy())

# Create a dataframe with the results
logits_df = pd.DataFrame(y_logits, columns = [f'{label}_Prob' for label in LABELS])
inference_df = pd.concat([labels, merged_data, pd.DataFrame(y_pred, columns = ['prediction']), logits_df], axis = 1)




# Categorical to human readable
inference_df['prediction'] = decode_labels(inference_df['prediction'])
inference_df = decode_features(inference_df)


# Save the results
target_dir = os.path.join(current_dir, "AutoDeepRun/xgboost_judgement.csv")
inference_df.to_csv(target_dir, index = False)
