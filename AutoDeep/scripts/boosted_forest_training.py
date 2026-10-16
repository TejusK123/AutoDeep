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
        nargs="+",
        help="CSV file(s) with custom training targets, or user_data.csv from an earlier training run",
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
        help="Name of the tuning log CSV in the output directory (default: training_log)",
        metavar="<name>"
    )
    
    parser.add_argument(
        "-nw", "--no_weights",
        action="store_true",
        help="Skip saving the model (other outputs are still written)"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory for the model, config, metrics and user data (default: models/<timestamp>)",
        metavar="<path>"
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
    for path in args.targets_path or []:
        if not os.path.isfile(path):
            parser.error(f"Error: Targets file not found: {path}")
    
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
    import json
    import time
    from features import LABELS, LABEL_COLUMN, encode_features, encode_labels

    
    # Loading all relevant paths. Training reads the package's training data
    # but writes only to out_dir, never inside the package.
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    data_path = os.path.join(base_path, "training_data")
    current_dir = os.getcwd()

    if "AutoDeepRun" not in os.path.basename(current_dir):
        print("Error: Please run in AutoDeepRun directory")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join("models", time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(out_dir):
        print(f"Error: Output directory already exists: {out_dir}")
        sys.exit(1)
    os.makedirs(out_dir)
    print(f"Writing outputs to: {os.path.abspath(out_dir)}")
    
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
        user_frames = []
        for path in args.targets_path:
            targets_df = pd.read_csv(path)
            if targets_df.shape[1] > 2:
                # user_data.csv from an earlier run: features and labels already joined
                frame = targets_df
            else:
                data = pd.read_csv("fully_formatted_data.csv")
                frame = pd.merge(data, targets_df, on='provisional_id')
            user_frames.append(frame.rename(columns={frame.columns[-1]: LABEL_COLUMN}))
        merged_data = pd.concat(user_frames, ignore_index=True)
        merged_data = merged_data[merged_data[LABEL_COLUMN].isin(LABELS)] #Filter out any other classes
        print(f"input data shape before merging with database data: {merged_data.shape}")

        # Save the user data with the run, so it can be passed to -t again later
        merged_data.to_csv(os.path.join(out_dir, "user_data.csv"), index=False)

        if not args.no_db_data and dfs != []:
            if list(training_data.columns) != list(merged_data.columns):
                print("Error: user data columns do not match the training data columns")
                print(f"  only in user data: {sorted(set(merged_data.columns) - set(training_data.columns))}")
                print(f"  only in training data: {sorted(set(training_data.columns) - set(merged_data.columns))}")
                sys.exit(1)
            merged_data = pd.concat([training_data, merged_data], ignore_index=True)

        print(merged_data.shape, "after merging with database data")

    else:
        merged_data = training_data
    
    merged_data.drop_duplicates(inplace=True)
    print(merged_data.shape, "After dropping duplicates")
    #Preprocessing
    n_rows = int(merged_data.shape[0])
    class_counts = {str(k): int(v) for k, v in merged_data.iloc[:,-1].value_counts().items()}
    targets = encode_labels(merged_data.iloc[:,-1])
    merged_data = merged_data.iloc[:,1:-1]
    
    merged_data = encode_features(merged_data)

    X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2)

    print("Starting training (Naive model)")
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"naive model accuracy post-train {accuracy}")
    metrics = {"naive_accuracy": float(accuracy)}
    
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
        metrics.update(accuracy=float(accuracy), weighted_f1_score=float(weighted_f1score),
                       macro_f1_score=float(macro_f1score))
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

        data_dir = os.path.join(out_dir, f"{args.output}.csv")
        df.to_csv(data_dir, index=False)
        print(f"Training log saved at: {data_dir}")

        config = best_result.config
        metrics.update({key: float(value) for key, value in best_result.metrics.items()
                        if key in ("mean_accuracy", "std_accuracy", "weighted_f1_score", "std_weighted_f1_score",
                                   "macro_f1_score", "std_macro_f1_score")})
        metrics["accuracy"] = metrics["mean_accuracy"]
        best_model = XGBClassifier(**best_result.config)
        best_model.fit(X_train, y_train)

    run_config = {
        "hyperparameters": {key: (value.item() if hasattr(value, "item") else value) for key, value in config.items()},
        "arguments": vars(args),
        "training_data": None if args.no_db_data else data_path,
        "n_rows": n_rows,
        "class_counts": class_counts,
        "xgboost_version": xgb.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.no_weights:
        print("Model not saved (-nw)")
    else:
        model_path = os.path.join(out_dir, "miRNA_model.json")
        best_model.save_model(model_path)
        print(f"Model saved at: {model_path}")
        print(f"Use it with: AutoDeep infer <directory> --model {os.path.abspath(model_path)}")


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














