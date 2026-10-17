import argparse
import sys
import os

# training_data_pre/ has the features the pipeline produces and the shipped
# model was trained on; training_data/ holds the newer entropy features.
# alt_model_test.py trains on the same data.
TRAINING_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "training_data_pre")


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
  AutoDeep train --targets_path labels.csv --hyperparameters hp.json
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the data splits, XGBoost and the tuning sampler (default: 42; saved in config.json). "
             "With more than 10 tuning rounds, results also depend on the number of CPUs",
        metavar="<int>"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.no_db_data and args.targets_path is None:
        parser.error("Error: No data to train on. Either unset -n/--no_db_data or provide --targets_path")

    if not 0 <= args.seed < 2 ** 31:
        parser.error(f"Error: --seed must be between 0 and {2 ** 31 - 1}, got {args.seed}")
    
    # Validate file paths if provided
    for path in args.targets_path or []:
        if not os.path.isfile(path):
            parser.error(f"Error: Targets file not found: {path}")
    
    if args.hyperparameters and not os.path.isfile(args.hyperparameters):
        parser.error(f"Error: Hyperparameters file not found: {args.hyperparameters}")
    
    train(args)


def read_training_data(data_path):
    """The training CSVs in data_path as one frame, or None if there are none."""
    import pandas as pd

    dfs = []
    for root, dirs, files in os.walk(data_path):
        # Listing order differs between machines, and row order decides the seeded splits
        for file in sorted(files):
            dfs.append(pd.read_csv(os.path.join(data_path, file)))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_training_set(targets_paths, no_db_data, out_dir):
    """Encoded features and labels to train on: the bundled training data (unless
    no_db_data) plus the labelled loci in targets_paths, which are also saved to
    out_dir/user_data.csv. Returns (features, labels, {"n_rows", "class_counts"}).
    Raises ValueError if the user data and the training data have different columns."""
    import pandas as pd
    from features import LABELS, LABEL_COLUMN, encode_features, encode_labels

    # Loading all database data
    training_data = None if no_db_data else read_training_data(TRAINING_DATA)
    if training_data is not None:
        print(training_data.shape, "Shape of database data")

    if targets_paths is not None:
        user_frames = []
        for path in targets_paths:
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

        if training_data is not None:
            if list(training_data.columns) != list(merged_data.columns):
                raise ValueError(
                    "user data columns do not match the training data columns\n"
                    f"  only in user data: {sorted(set(merged_data.columns) - set(training_data.columns))}\n"
                    f"  only in training data: {sorted(set(training_data.columns) - set(merged_data.columns))}")
            merged_data = pd.concat([training_data, merged_data], ignore_index=True)

        print(merged_data.shape, "after merging with database data")

    else:
        merged_data = training_data

    merged_data.drop_duplicates(inplace=True)
    print(merged_data.shape, "After dropping duplicates")
    #Preprocessing
    info = {"n_rows": int(merged_data.shape[0]),
            "class_counts": {str(k): int(v) for k, v in merged_data.iloc[:,-1].value_counts().items()}}
    targets = encode_labels(merged_data.iloc[:,-1])
    features = encode_features(merged_data.iloc[:,1:-1])
    return features, targets, info


def tuning_search_space():
    """Ray Tune search space for the XGBoost hyperparameters."""
    from ray import tune

    return {
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


def tuning_search_alg(seed):
    """Optuna's TPE sampler, seeded. Its first 10 proposals are random draws in a fixed
    order; later ones are fitted to the trials finished so far. Ray runs as many trials
    at once as there are CPUs, so beyond 10 tuning rounds the results depend on the CPU count."""
    import optuna
    from ray.tune.search.optuna import OptunaSearch

    return OptunaSearch(sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))


def train(args):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils import class_weight
    import json
    import time
    from hyperparameters import load_xgboost_hyperparameters

    
    # Training reads the package's training data but writes only to out_dir,
    # never inside the package.
    data_path = TRAINING_DATA
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
    
    try:
        merged_data, targets, data_info = load_training_set(args.targets_path, args.no_db_data, out_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    n_rows, class_counts = data_info["n_rows"], data_info["class_counts"]

    X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, random_state=args.seed)

    print("Starting training (Naive model)")
    model = XGBClassifier(random_state=args.seed)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"naive model accuracy post-train {accuracy}")
    metrics = {"naive_accuracy": float(accuracy)}
    
    if args.hyperparameters:
        print("Using manual hyperparameters")
        X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, stratify=targets,
                                                            random_state=args.seed)
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
        try:
            config = load_xgboost_hyperparameters(args.hyperparameters)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        best_model = XGBClassifier(**config, random_state=args.seed)
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
        from ray.tune.schedulers import ASHAScheduler

        seed = args.seed

        def model_training(config):
            weighted_f1_scores = []
            accuracies = []
            macro_f1_scores = []
            for i in range(10):
                # Every configuration is scored on the same ten splits
                X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, stratify=targets,
                                                                    random_state=seed + i)
                classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
                model = XGBClassifier(**config, random_state=seed)
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

        tuner = tune.Tuner(
            model_training, 
            tune_config = tune.TuneConfig(num_samples = args.tuning_rounds, 
                                        search_alg = tuning_search_alg(seed), 
                                        scheduler = ASHAScheduler(),
                                        metric = "weighted_f1_score", 
                                        mode = "max"), 
            param_space = tuning_search_space()
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
        best_model = XGBClassifier(**best_result.config, random_state=seed)
        best_model.fit(X_train, y_train)

    run_config = {
        "hyperparameters": {key: (value.item() if hasattr(value, "item") else value) for key, value in config.items()},
        "seed": args.seed,
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














