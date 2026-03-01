import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog="AutoDeep train",
        description="Train a new classification model for miRNA stratification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION:
  Trains a new gradient boosted forest classification model on your dataset.
  The model can be trained using only database data, custom targets, or both.

TARGET FILE FORMAT:
  CSV file with loci names in first column and class names in second column.
  Valid classes: Candidate, Confident, falsepositive
  Loci names must match the fully_formatted_data.csv file.

EXAMPLES:
  AutoDeep train
  AutoDeep train --targets_path labels.csv --tuning_rounds 1000
  AutoDeep train --targets_path labels.csv --hyperparameters config.txt
  AutoDeep train --no_db_data --targets_path labels.csv
        """
    )
    
    parser.add_argument(
        "-t", "--targets_path",
        type=str,
        help="Path to CSV file with custom training targets",
        metavar="<path>"
    )
    
    parser.add_argument(
        "-n", "--no_db_data",
        action="store_true",
        help="Omit original database data from training (use only custom targets)"
    )
    
    parser.add_argument(
        "-r", "--tuning_rounds",
        type=int,
        default=10,
        help="Number of hyperparameter tuning rounds (default: 10)",
        metavar="<int>"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="training_log",
        help="Name of output training log file (default: training_log)",
        metavar="<name>"
    )
    
    parser.add_argument(
        "-nw", "--no_weights",
        action="store_true",
        help="Skip saving model weights (recommended for testing)"
    )
    
    parser.add_argument(
        "-hp", "--hyperparameters",
        type=str,
        help="Path to hyperparameter configuration file for manual tuning",
        metavar="<path>"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.no_db_data and args.targets_path is None:
        parser.error("Error: No data to train on. Either unset -n/--no_db_data or provide --targets_path")
    
    # Validate file paths if provided
    if args.targets_path and not os.path.isfile(args.targets_path):
        parser.error(f"Error: Targets file not found: {args.targets_path}")
    
    if args.hyperparameters and not os.path.isfile(args.hyperparameters):
        parser.error(f"Error: Hyperparameters file not found: {args.hyperparameters}")
    
    train(args)


def train(args):
    import pandas as pd 
    import xgboost as xgb
    from xgboost import XGBClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils import class_weight

    
    # Loading all relevant paths
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    weights_path = os.path.join(base_path, "model_weights/miRNA_model.json")
    data_path = os.path.join(base_path, "training_data")
    current_dir = os.getcwd()

    if "AutoDeepRun" not in os.path.basename(current_dir):
        print("Error: Please run in AutoDeepRun directory")
        sys.exit(1)
    
    # Loading all database data
    dfs = []
    if args.no_db_data:
        pass
    else:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                tempdf = pd.read_csv(os.path.join(data_path, file))
                dfs.append(tempdf)

        if dfs != []:
            training_data = pd.concat(dfs, ignore_index=True)
            print(training_data.shape, "Shape of database data")
        else:
            pass

    if args.targets_path is not None:
        data = pd.read_csv("fully_formatted_data.csv")
        merged_data = pd.merge(data, pd.read_csv(args.targets_path), on='provisional_id')
        merged_data = merged_data[merged_data.iloc[:,-1].isin(["Candidate", "Confident", "falsepositive"])] #Filter out any other classes
        print(f"input data shape before merging with database data: {merged_data.shape}")
        
        #Save the user data to the database
        import time
        merged_data.to_csv(os.path.join(data_path, f"{args.targets_path}_{str(round(time.time()))}.csv"), index=False) 

        try:
            merged_data = pd.concat([training_data, merged_data], ignore_index=True)
        except:
            pass

        print(merged_data.shape, "after merging with database data")

    else:
        merged_data = training_data
    
    merged_data.drop_duplicates(inplace=True)
    print(merged_data.shape, "After dropping duplicates")
    #Preprocessing
    targets = merged_data.iloc[:,-1].astype('category').cat.codes
    merged_data = merged_data.iloc[:,1:-1]
    
    merged_data['mature_5\'u_or_3\'u'] = merged_data['mature_5\'u_or_3\'u'].astype('category').cat.codes
    merged_data['homologous_seed_in_miRBase'] = merged_data['homologous_seed_in_miRBase'].astype('category').cat.codes
    merged_data['significant_randfold_p-value'] = merged_data['significant_randfold_p-value'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2)

    print("Starting training (Naive model)")
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"naive model accuracy post-train {accuracy}")
    
    if args.hyperparameters:
        print("Using manual hyperparameters")
        X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, stratify=targets)
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
        with open(args.hyperparameters, 'r') as f:
            lines = list(filter(lambda x: len(x) != 0, (item.split() for item in f.readlines())))
        
        valid_attributes = ['max_depth', 'min_child_weight', 'subsample', 'eta', 'n_estimators', 'gamma', 'base_score', 'alpha', 'lambda', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode']
        input_check = [(item not in valid_attributes, item) for item in [ind[0] for ind in lines]]
        
        if any([item[0] for item in input_check]):
            print(f"Invalid hyperparameter(s): {[item[1] for item in input_check if item[0] == True]}")
            sys.exit(1)

        config = {item[0] : eval(item[-1]) for item in lines}
        best_model = XGBClassifier(**config)
        best_model.fit(X_train, y_train, sample_weight=classes_weights)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1score = f1_score(y_test, y_pred, average='weighted')
        macro_f1score = f1_score(y_test, y_pred, average='macro')
        print(f"Tuned Model Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"Weighted F1 score: {weighted_f1score}")
        print(f"Macro F1 score: {macro_f1score}")
    else:
        #----raytune hyperparameter tuning
        print("Starting hyperparameter tuning")
        from ray import tune, train
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.schedulers import ASHAScheduler
        
        def model_training(config):
            weighted_f1_scores = []
            accuracies = []
            macro_f1_scores = []
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, stratify=targets)
                classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
                model = XGBClassifier(**config)
                model.fit(X_train, y_train, sample_weight=classes_weights)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                weighted_f1score = f1_score(y_test, y_pred, average='weighted')
                macro_f1score = f1_score(y_test, y_pred, average='macro')
                accuracies.append(accuracy)
                macro_f1_scores.append(macro_f1score)
                weighted_f1_scores.append(weighted_f1score)

            accuracy = np.mean(accuracies)
            stds_acc = np.std(accuracies)
            weighted_f1_score = np.mean(weighted_f1_scores)
            stds_weighted_f1_score = np.std(weighted_f1_scores)
            macro_f1_score = np.mean(macro_f1_scores)
            stds_macro_f1_score = np.std(macro_f1_scores)
            train.report({'mean_accuracy': accuracy,
                        "std_accuracy" : stds_acc, 
                        'weighted_f1_score' : weighted_f1_score,
                        'std_weighted_f1_score' : stds_weighted_f1_score,
                        "macro_f1_score" : macro_f1_score,
                        "std_macro_f1_score" : stds_macro_f1_score,
                        'done': True})

        config = {
            "objective": "multi:softprob",
            "max_depth": tune.randint(1, 6),
            "min_child_weight": tune.choice([1, 2, 3, 4]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
            "n_estimators": tune.randint(100, 1000),
            "gamma": tune.uniform(0, 1),
            "base_score": tune.uniform(0.33, 0.66),
            "alpha" : tune.loguniform(1e-3, 10),
            "lambda" : tune.loguniform(1e-3, 1),
            "colsample_bytree" : tune.uniform(0.8, 1),
            "colsample_bylevel" : tune.uniform(0.8, 1),
            "colsample_bynode" : tune.uniform(0.8, 1),
        }

        tuner = tune.Tuner(
            model_training, 
            tune_config = tune.TuneConfig(num_samples = args.tuning_rounds, 
                                        search_alg = OptunaSearch(), 
                                        scheduler = ASHAScheduler(),
                                        metric = "weighted_f1_score", 
                                        mode = "max"), 
            param_space = config
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="weighted_f1_score", mode="max")
        best_checkpoint = best_result.checkpoint 
        df = results.get_dataframe() 

        for item in best_result.metrics.keys():
            print(f"{item} : {best_result.metrics[item]}")
        print(f"The corresponding config is {best_result.config}")

        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, f"{args.output}.csv")                 
        df.to_csv(data_dir, index=False)
        print(f"Training log saved at: {data_dir}")

        best_model = XGBClassifier(**best_result.config)
        best_model.fit(X_train, y_train)

    if args.no_weights:
        print("Model weights not saved")
        pass
    else:
        best_model.save_model(weights_path)
        print(f"Updated model saved at: {weights_path}")


if __name__ == "__main__":
    main()


















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














