import click 

class CustomUsageMsg(click.Command):
    def format_usage(self, ctx, formatter):
        formatter.write_text("Usage: AutoDeep train [OPTIONS]")


@click.command(cls = CustomUsageMsg)
@click.option("--targets_path", "-t", help="Path to targets file", metavar = "<str>", type = click.Path(exists = True, dir_okay = False))
@click.option("--no_db_data", "-n", is_flag = True, help="Flag that omits original dataset from training")
@click.option("--tuning_rounds", "-r", default = 10, help="Number of tuning rounds: Default <10>", metavar = "<int>")
@click.option("--output", "-o", default = "training_log", help="Name of output training_log file", metavar = "<str>")
def train(no_db_data, tuning_rounds, output, targets_path):
    """ Trains the model with given targets 

    
    <targets_path> is a CSV file with loci names in first column and class names in second column (Candidate, Confident, falsepositive).
    The CSV file must have loci names in accordance with the fully_formatted_data.csv file.
    

    Don't specify <targets_path> if you want to use just the database data.
    
    """
    import pandas as pd 
    import xgboost as xgb
    from xgboost import XGBClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils import class_weight
    import os
    import sys

    # Loading all relevant paths
    base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    weights_path = os.path.join(base_path, "model_weights/miRNA_model.json")
    data_path = os.path.join(base_path, "training_data")
    current_dir = os.getcwd()

    if "AutoDeepRun" not in os.path.basename(current_dir):
        print("Please run in AutoDeepRun directory")
        sys.exit(1)
        
    
    if no_db_data and targets_path is None:
        print("Error: No data to train on: Unset flag -n/--no_db_data or provide targets file")
        sys.exit(1)



    # Loading all database data
    #If -n flag is passed, then we will not load the database data

    dfs = []
    if no_db_data:
        pass
    else:
        for root, dirs, files in os.walk(data_path):
            for file in files:

                tempdf = pd.read_csv(os.path.join(data_path,file))
                dfs.append(tempdf)

        if dfs != []:
            training_data = pd.concat(dfs, ignore_index = True)
            print(training_data.shape, "Shape of database data")
        else:
            pass



    if targets_path is not None:
        data = pd.read_csv("fully_formatted_data.csv")
        merged_data = pd.merge(data, pd.read_csv(targets_path), on='provisional_id')
        merged_data = merged_data[merged_data.iloc[:,-1].isin(["Candidate", "Confident", "falsepositive"])] #Filter out any other classes
        print(f"input data shape before merging with database data: {merged_data.shape}")

        

        #Save the user data to the database
        ############################
        import time
        merged_data.to_csv(os.path.join(data_path, f"{targets_path}_{str(round(time.time()))}.csv"), index = False) 

        ##########################

        try:
            merged_data = pd.concat([training_data, merged_data], ignore_index = True)
        except:
            pass

        

        print(merged_data.shape, "after merging with database data")

    else:
        merged_data = training_data
    

    merged_data.drop_duplicates(inplace = True)
    print(merged_data.shape, "After dropping duplicates")
    #Preprocessing

    targets = merged_data.iloc[:,-1].astype('category').cat.codes


    merged_data = merged_data.iloc[:,1:-1]
    #print(merged_data.shape)
    #print(merged_data['Unnamed: 0'])
    merged_data['mature_5\'u_or_3\'u'] = merged_data['mature_5\'u_or_3\'u'].astype('category').cat.codes
    merged_data['homologous_seed_in_miRBase'] = merged_data['homologous_seed_in_miRBase'].astype('category').cat.codes
    merged_data['significant_randfold_p-value'] = merged_data['significant_randfold_p-value'].astype('category').cat.codes
    #merged_data['mature_seq_on_top'] = merged_data['mature_seq_on_top'].astype('category').cat.codes



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
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler


    def model_training(config):
        weighted_f1_scores = []
        accuracies = []
        macro_f1_scores = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(merged_data, targets, test_size=0.2, stratify = targets)
            classes_weights = class_weight.compute_sample_weight(class_weight = 'balanced', y = y_train)
            model = XGBClassifier(**config)
            model.fit(X_train, y_train, sample_weight = classes_weights)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1score = f1_score(y_test, y_pred, average = 'weighted')
            macro_f1score = f1_score(y_test, y_pred, average = 'macro')
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
        "colsample_bylevel" : tune.uniform(0.8,1),

    }


    tuner = tune.Tuner(
    model_training, 
    tune_config = tune.TuneConfig(num_samples = tuning_rounds, 
                                search_alg = OptunaSearch(), 
                                scheduler = ASHAScheduler(),
                                metric = "weighted_f1_score", 
                                mode = "max"), 
    param_space = config
    )

    results = tuner.fit()

    best_result = results.get_best_result( 
        metric="weighted_f1_score", mode="max")
    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint 
    # Get a dataframe for the last reported results of all of the trials
    df = results.get_dataframe() 


    #print(f"Max Accuracy is {np.max(df['mean_accuracy'])} with std {df.loc[df['mean_accuracy'] == np.max(df['mean_accuracy']), 'std_accuracy'].iloc[0]}")
    for item in best_result.metrics.keys():
        print(f"{item} : {best_result.metrics[item]}")
    print(f"The corresponding config is {best_result.config}")
    #print(f"Max weighted_f1_score is {np.max(df['weighted_f1_score'])}")


    best_model = XGBClassifier(**best_result.config)
    best_model.fit(X_train, y_train)


    best_model.save_model(weights_path)

    print(f"Updated model saved at: {weights_path}")

    current_dir = os.getcwd()

    data_dir = os.path.join(current_dir, f"{output}.csv")

 
                            
    df.to_csv(data_dir, index = False)


    print(f"Training log saved at: {data_dir}")


if __name__ == "__main__":
    train()


















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














