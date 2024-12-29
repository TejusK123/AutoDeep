import pandas as pd 
import xgboost
from xgboost import XGBClassifier
import numpy as np
import os



base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
model_path = os.path.join(base_path, "model_weights/miRNA_model.json")

model = XGBClassifier()
model.load_model(model_path)


current_dir = os.getcwd()

data_dir = os.path.join(current_dir, "AutoDeepRun/fully_formatted_data.csv")

merged_data = pd.read_csv(data_dir)

merged_data['mature_5\'u_or_3\'u'] = merged_data['mature_5\'u_or_3\'u'].astype('category').cat.codes


merged_data['homologous_seed_in_miRBase'] = merged_data['homologous_seed_in_miRBase'].astype('category').cat.codes

merged_data['significant_randfold_p-value'] = merged_data['significant_randfold_p-value'].astype('category').cat.codes

merged_data['mature_seq_on_top'] = merged_data['mature_seq_on_top'].astype('category').cat.codes


labels = merged_data['provisional_id']
merged_data = merged_data.drop(columns = ['provisional_id'])

#print(merged_data.head())

y_pred = model.predict(merged_data.to_numpy())


inference_df = pd.concat([labels, merged_data, pd.DataFrame(y_pred, columns = ['prediction'])], axis = 1)

#print(y_pred)

conversion_map = {0 : 'Candidate', 1 : 'Confident', 2 : 'falsepositive'}
inference_df['prediction'] = inference_df['prediction'].apply(lambda x: conversion_map[x])

inference_df['mature_5\'u_or_3\'u'] = inference_df['mature_5\'u_or_3\'u'].apply(lambda x: 'False' if x == 0 else 'True')
inference_df['homologous_seed_in_miRBase'] = inference_df['homologous_seed_in_miRBase'].apply(lambda x: 'False' if x == 0 else 'True')
inference_df['significant_randfold_p-value'] = inference_df['significant_randfold_p-value'].apply(lambda x: 'no' if x == 0 else 'yes')
inference_df['mature_seq_on_top'] = inference_df['mature_seq_on_top'].apply(lambda x: 'False' if x == 0 else 'True')


target_dir = os.path.join(current_dir, "AutoDeepRun/xgboost_judgement.csv")
inference_df.to_csv(target_dir, index = False)
