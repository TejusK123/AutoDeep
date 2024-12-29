import pandas as pd 
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys




base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

weights_path = os.path.join(base_path, "model_weights/miRNA_model.json")

data_path = os.path.join(base_path, "training_data")


dfs = []


if len(sys.argv) >= 3 and sys.argv[2] == "-n":
    pass
else:

    for root, dirs, files in os.walk(data_path):
        for file in files:

            tempdf = pd.read_csv(os.path.join(data_path,file))
            dfs.append(tempdf)


    if dfs != []:
        training_data = pd.concat(dfs, ignore_index = True)
        print(training_data.shape, "training_data_shape", training_data)
    else:
        pass



current_dir = os.getcwd()

if "AutoDeepRun" not in os.path.basename(current_dir):
    print("Please run in AutoDeepRun directory")
    sys.exit(1)

if len(sys.argv) <= 1:
    print("Usage: python boosted_forest_training.py <Targets>")
    sys.exit(1)


if not os.path.isfile(sys.argv[1]):
    print("sys.argv[1] is not a valid path")
    sys.exit(1)

data = pd.read_csv("fully_formatted_data.csv")

targets_dir = sys.argv[1]

target_location = pd.read_csv(targets_dir)




merged_data = pd.merge(data, target_location, on='provisional_id')

print(merged_data)
print(f"input training data shape {merged_data.shape}")

merged_data = merged_data[merged_data.iloc[:,-1].isin(["Candidate", "Confident", "falsepositive"])] 

print(f"input training data shape {merged_data.shape}")




############################
import time
merged_data.to_csv(os.path.join(data_path, f"{sys.argv[1]}_{str(round(time.time()))}.csv"), index = False) 

##########################


print(merged_data.shape, "before merging with database data")


try:
    merged_data = pd.concat([training_data, merged_data], ignore_index = True)
except:
    pass

merged_data.drop_duplicates(inplace = True)

print(merged_data.shape, "after merging with database data")





targets = merged_data.iloc[:,-1].astype('category').cat.codes


merged_data = merged_data.iloc[:,1:-1]
#print(merged_data.shape)
#print(merged_data['Unnamed: 0'])


merged_data['mature_5\'u_or_3\'u'] = merged_data['mature_5\'u_or_3\'u'].astype('category').cat.codes
merged_data['homologous_seed_in_miRBase'] = merged_data['homologous_seed_in_miRBase'].astype('category').cat.codes
merged_data['significant_randfold_p-value'] = merged_data['significant_randfold_p-value'].astype('category').cat.codes
merged_data['mature_seq_on_top'] = merged_data['mature_seq_on_top'].astype('category').cat.codes



X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2)

print("Starting training (Naive model)")
model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"naive model accuracy post-train {accuracy}")
print(f"{model.get_xgb_params()}")


#----raytune hyperparameter tuning
print("Starting hyperparameter tuning")
from ray import tune, train

def model_training(config):
    
    accuracies = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2)
        model = XGBClassifier(**config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    accuracy = np.mean(accuracies)
    #print(accuracies)
    stds = np.std(accuracies)
    train.report({'mean_accuracy': accuracy, "std_accuracy" : stds, 'done': True})

config = {
    "objective": tune.choice(["multi:softmax", "multi:softprob"]),
    "max_depth": tune.randint(1, 6),
    "min_child_weight": tune.choice([1, 2, 3, 4]),
    "subsample": tune.uniform(0.5, 1.0),
    "eta": tune.loguniform(1e-4, 1e-1),
    "n_estimator": tune.randint(100, 1000),
    "gamma": tune.uniform(0, 1),
    "base_score": tune.uniform(0.33, 0.66),
    "scale_pos_weight" : tune.uniform(0.5,3.0),
    "alpha" : tune.loguniform(1e-3, 10),
    "lambda" : tune.loguniform(1e-3, 1),
    "colsample_bytree" : tune.uniform(0.8, 1),
    "colsample_bylevel" : tune.uniform(0.8,1),

}

tuner = tune.Tuner(model_training, tune_config = tune.TuneConfig(num_samples = 25000), param_space = config)

results = tuner.fit()

best_result = results.get_best_result( 
    metric="mean_accuracy", mode="max")
# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint 
# Get a dataframe for the last reported results of all of the trials
df = results.get_dataframe() 
print(f"Max Accuracy is {np.max(df['mean_accuracy'])} with std {df.loc[df['mean_accuracy'] == np.max(df['mean_accuracy']), 'std_accuracy'].iloc[0]}")
print(f"The corresponding config is {best_result.config}")


best_model = XGBClassifier(**best_result.config)
best_model.fit(X_train, y_train)


best_model.save_model(weights_path)

print(f"Updated model saved at: {weights_path}")










# model2 = XGBClassifier()

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(model2, merged_data.to_numpy(), targets.to_numpy(), cv=3, scoring='accuracy')

# print(f"Accuracy scores for each fold: {scores}")
# print(f"Mean accuracy: {scores.mean():.4f}")

'''
dtrain = xgb.DMatrix(data = X_train, label=y_train)
dval = xgb.DMatrix(data = X_test, label = y_test)




params = {
    'objective' : 'multi:softmax',
    'num_class' : 3,
    
    'eval_metric' : 'merror'

    
}


def accuracy_metric(preds, matrix):
    labels = matrix.get_label()
    preds_class = preds.argmax(axis = 1)
    accuracy = accuracy_score(labels, preds_class)
    return('accuracy', accuracy)


evals = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(params, dtrain, num_boost_round = 100, evals = evals, early_stopping_rounds = 10)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")
'''













