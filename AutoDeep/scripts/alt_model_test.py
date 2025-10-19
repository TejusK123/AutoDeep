import click


class CustomUsageMsg(click.Command):
	def format_usage(self, ctx, formatter):
		formatter.write_text("Usage: AutoDeep train-svm [OPTIONS]")


@click.command(cls=CustomUsageMsg)
@click.option("--model", "-m", default="svm", type=click.Choice(["svm", "knn", "mlp"]), help="Model type: svm, knn, or mlp (neural network)")
@click.option("--targets_path", "-t", help="Path to targets file", metavar="<str>", type=click.Path(exists=True, dir_okay=False))
@click.option("--no_db_data", "-n", is_flag=True, help="Flag that omits original dataset from training")
@click.option("--tuning_rounds", "-r", default=10, help="Number of tuning rounds: Default <10>", metavar="<int>")
@click.option("--output", "-o", default="model_training_log", help="Name of output training_log file", metavar="<str>")
@click.option("--no_weights", "-nw", is_flag=True, help="Flag that omits saving the model weights (recommended for testing)")
@click.option("--hyperparameters", "-hp", help="Path to hyperparameter configuration file in case of manual tuning", metavar="<str>", type=click.Path(exists=True, dir_okay=False))
def train_model(model, no_db_data, tuning_rounds, output, targets_path, no_weights, hyperparameters):
	"""Trains an SVM model using the same data layout as the other training scripts.

	Behavior mirrors `boosted_forest_training.py`:
	- Loads CSVs from `training_data/` unless `--no_db_data` is set
	- Optionally merges a user-provided `--targets_path` (CSV must have 'provisional_id' and class label)
	- Preprocesses categorical columns using category codes
	- Trains a naive SVC pipeline, then either uses manual hyperparameters or performs randomized search
	- Saves best model to `model_weights/svm_model.pkl` unless `--no_weights` is set
	"""
	import os
	import sys
	import time
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, f1_score
	from sklearn.svm import SVC
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.neural_network import MLPClassifier
	from sklearn.preprocessing import StandardScaler
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import Pipeline
	from sklearn.utils import class_weight
	import joblib

	# Loading all relevant paths
	base_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
	weights_path = os.path.join(base_path, f"model_weights/{model}_model.pkl")
	data_path = os.path.join(base_path, "training_data")
	current_dir = os.getcwd()

	if "AutoDeepRun" not in os.path.basename(current_dir):
		print("Please run in AutoDeepRun directory")
		sys.exit(1)

	if no_db_data and targets_path is None:
		print("Error: No data to train on: Unset flag -n/--no_db_data or provide targets file")
		sys.exit(1)

	# Load database CSVs
	dfs = []
	if not no_db_data:
		for root, dirs, files in os.walk(data_path):
			for file in files:
				try:
					tempdf = pd.read_csv(os.path.join(data_path, file))
					dfs.append(tempdf)
				except Exception:
					# skip non-csv or malformed files
					continue

		if dfs != []:
			training_data = pd.concat(dfs, ignore_index=True)
			print(training_data.shape, "Shape of database data")
		else:
			training_data = pd.DataFrame()
	else:
		training_data = pd.DataFrame()

	# Merge user targets if provided
	if targets_path is not None:
		data = pd.read_csv("fully_formatted_data.csv")
		merged_data = pd.merge(data, pd.read_csv(targets_path), on='provisional_id')
		merged_data = merged_data[merged_data.iloc[:, -1].isin(["Candidate", "Confident", "falsepositive"])]
		print(f"input data shape before merging with database data: {merged_data.shape}")

		# Save a copy into the training_data folder
		try:
			merged_data.to_csv(os.path.join(data_path, f"{os.path.basename(targets_path)}_{str(round(time.time()))}.csv"), index=False)
		except Exception:
			pass

		try:
			merged_data = pd.concat([training_data, merged_data], ignore_index=True)
		except Exception:
			pass

		print(merged_data.shape, "after merging with database data")
	else:
		merged_data = training_data

	if merged_data is None or merged_data.shape[0] == 0:
		print("No data available for training")
		sys.exit(1)

	merged_data.drop_duplicates(inplace=True)
	print(merged_data.shape, "After dropping duplicates")

	# Targets are expected to be the last column
	targets = merged_data.iloc[:, -1].astype('category').cat.codes

	# Features: drop first column (id) and last column (target)
	X = merged_data.iloc[:, 1:-1].copy()

	# Encode known categorical columns the same way as other scripts
	cat_columns = ['mature_5\'u_or_3\'u', 'homologous_seed_in_miRBase', 'significant_randfold_p-value']
	for col in cat_columns:
		if col in X.columns:
			X[col] = X[col].astype('category').cat.codes

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, stratify=targets if len(np.unique(targets))>1 else None)

	if model == "svm":
		print("Starting training (Naive SVM)")
		base_pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='mean')),
			('scaler', StandardScaler()),
			('svc', SVC(probability=True, class_weight='balanced', random_state=42))
		])
	elif model == "knn":
		print("Starting training (Naive KNN)")
		base_pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='mean')),
			('scaler', StandardScaler()),
			('knn', KNeighborsClassifier())
		])
	elif model == "mlp":
		print("Starting training (Naive Neural Network)")
		base_pipeline = Pipeline([
			('imputer', SimpleImputer(strategy='mean')),
			('scaler', StandardScaler()),
			('mlp', MLPClassifier(max_iter=200, random_state=42))
		])

	base_pipeline.fit(X_train, y_train)
	y_pred = base_pipeline.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"naive {model.upper()} accuracy post-train {accuracy}")

	# If manual hyperparameters provided, use them for SVC
	if hyperparameters:
		print(f"Using manual hyperparameters for {model.upper()}")
		with open(hyperparameters, 'r') as f:
			lines = list(filter(lambda x: len(x) != 0, (item.split() for item in f.readlines())))
		config = {item[0]: eval(item[-1]) for item in lines}
		if model == "svm":
			model_pipe = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('svc', SVC(probability=True, class_weight='balanced', random_state=42, **{k: v for k, v in config.items() if k in SVC().get_params()}))
			])
		elif model == "knn":
			model_pipe = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('knn', KNeighborsClassifier(**{k: v for k, v in config.items() if k in KNeighborsClassifier().get_params()}))
			])
		elif model == "mlp":
			model_pipe = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('mlp', MLPClassifier(max_iter=200, random_state=42, **{k: v for k, v in config.items() if k in MLPClassifier().get_params()}))
			])
		model_pipe.fit(X_train, y_train)
		y_pred = model_pipe.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		weighted_f1score = f1_score(y_test, y_pred, average='weighted')
		macro_f1score = f1_score(y_test, y_pred, average='macro')
		print(f"Manual-tuned Model Metrics:")
		print(f"Accuracy: {accuracy}")
		print(f"Weighted F1 score: {weighted_f1score}")
		print(f"Macro F1 score: {macro_f1score}")
		best_model = model_pipe
	else:
		print("Starting hyperparameter tuning")
		from ray import tune, train
		from ray.tune.search.optuna import OptunaSearch
		from ray.tune.schedulers import ASHAScheduler

		def model_training(config):
			weighted_f1_scores = []
			accuracies = []
			macro_f1_scores = []
			for i in range(10):
				X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, targets, test_size=0.2, stratify=targets)

				if model == "svm":
					kernel = config['kernel']
					svc_params = {
						'C': config['C'],
						'kernel': kernel,
						'probability': True,
						'class_weight': 'balanced',
						'random_state': 42,
						'shrinking': config['shrinking'],
						'tol': config['tol'],
						'cache_size': config['cache_size'],
						'max_iter': config['max_iter'],
					}
					if kernel in ['rbf', 'poly', 'sigmoid']:
						if config['gamma_type'] == 'numeric':
							svc_params['gamma'] = config['gamma_value']
						else:
							svc_params['gamma'] = config['gamma_type']
					if kernel == 'poly':
						svc_params['degree'] = config.get('degree', 3)
					if kernel in ['poly', 'sigmoid']:
						svc_params['coef0'] = config.get('coef0', 0.0)
					pipeline = Pipeline([
						('imputer', SimpleImputer(strategy='mean')),
						('scaler', StandardScaler()),
						('svc', SVC(**svc_params))
					])
				elif model == "knn":
					knn_params = {
						'n_neighbors': config['n_neighbors'],
						'weights': config['weights'],
						'p': config['p'],
						'algorithm': config['algorithm'],
					}
					pipeline = Pipeline([
						('imputer', SimpleImputer(strategy='mean')),
						('scaler', StandardScaler()),
						('knn', KNeighborsClassifier(**knn_params))
					])
				elif model == "mlp":
					hls = tuple([config['hidden_layer_sizes']] * config['num_hidden_layers'])
					mlp_params = {
						'hidden_layer_sizes': hls,
						'activation': config['activation'],
						'alpha': config['alpha'],
						'learning_rate_init': config['learning_rate_init'],
						'solver': config['solver'],
						'batch_size': config['batch_size'],
						'max_iter': config['max_iter'],
						'random_state': 42
					}
					pipeline = Pipeline([
						('imputer', SimpleImputer(strategy='mean')),
						('scaler', StandardScaler()),
						('mlp', MLPClassifier(**mlp_params))
					])

				pipeline.fit(X_train_cv, y_train_cv)
				y_pred_cv = pipeline.predict(X_test_cv)
				accuracy = accuracy_score(y_test_cv, y_pred_cv)
				weighted_f1score = f1_score(y_test_cv, y_pred_cv, average='weighted')
				macro_f1score = f1_score(y_test_cv, y_pred_cv, average='macro')
				accuracies.append(accuracy)
				macro_f1_scores.append(macro_f1score)
				weighted_f1_scores.append(weighted_f1score)

			accuracy = np.mean(accuracies)
			stds_acc = np.std(accuracies)
			weighted_f1_score = np.mean(weighted_f1_scores)
			stds_weighted_f1_score = np.std(weighted_f1_scores)
			macro_f1_score = np.mean(macro_f1_scores)
			stds_macro_f1_score = np.std(macro_f1_scores)
			train.report({
				'mean_accuracy': accuracy,
				"std_accuracy": stds_acc,
				'weighted_f1_score': weighted_f1_score,
				'std_weighted_f1_score': stds_weighted_f1_score,
				"macro_f1_score": macro_f1_score,
				"std_macro_f1_score": stds_macro_f1_score,
				'done': True
			})

		# Define search spaces for each model
		if model == "svm":
			search_space = {
				"C": tune.loguniform(1e-3, 1e3),
				"kernel": tune.choice(['rbf', 'linear', 'poly', 'sigmoid']),
				"gamma_type": tune.choice(['scale', 'auto', 'numeric']),
				"gamma_value": tune.loguniform(1e-4, 1e1),
				"degree": tune.randint(2, 5),
				"coef0": tune.uniform(0.0, 1.0),
				"shrinking": tune.choice([True, False]),
				"tol": tune.loguniform(1e-5, 1e-2),
				"cache_size": tune.choice([200, 500, 1000, 2000]),
				"max_iter": tune.choice([1000, 5000, 10000, -1]),
			}
		elif model == "knn":
			search_space = {
				"n_neighbors": tune.randint(1, 31),
				"weights": tune.choice(["uniform", "distance"]),
				"p": tune.choice([1, 2]),
				"algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
			}
		elif model == "mlp":
			search_space = {
				"hidden_layer_sizes": tune.randint(32, 257),
				"num_hidden_layers": tune.randint(1, 4),
				"activation": tune.choice(["relu", "tanh", "logistic"]),
				"alpha": tune.loguniform(1e-6, 1e-2),
				"learning_rate_init": tune.loguniform(1e-4, 1e-1),
				"solver": tune.choice(["adam", "sgd", "lbfgs"]),
				"batch_size": tune.choice([32, 64, 128]),
				"max_iter": tune.choice([200, 400, 800]),
			}

		# ...existing Ray Tune tuner code...
		tuner = tune.Tuner(
			model_training,
			tune_config=tune.TuneConfig(
				num_samples=tuning_rounds,
				search_alg=OptunaSearch(),
				scheduler=ASHAScheduler(),
				metric="weighted_f1_score",
				mode="max"
			),
			param_space=search_space
		)

		results = tuner.fit()

		best_result = results.get_best_result(
			metric="weighted_f1_score", mode="max")
		# Get a dataframe for the last reported results of all of the trials
		df = results.get_dataframe()

		for item in best_result.metrics.keys():
			print(f"{item} : {best_result.metrics[item]}")
		print(f"The corresponding config is {best_result.config}")

		data_dir = os.path.join(current_dir, f"{output}.csv")
		df.to_csv(data_dir, index=False)
		print(f"Training log saved at: {data_dir}")

		# Train final model with best config
		best_config = best_result.config
		if model == "svm":
			kernel = best_config['kernel']
			svc_params = {
				'C': best_config['C'],
				'kernel': kernel,
				'probability': True,
				'class_weight': 'balanced',
				'random_state': 42,
				'shrinking': best_config['shrinking'],
				'tol': best_config['tol'],
				'cache_size': best_config['cache_size'],
				'max_iter': best_config['max_iter'],
			}
			if kernel in ['rbf', 'poly', 'sigmoid']:
				if best_config['gamma_type'] == 'numeric':
					svc_params['gamma'] = best_config['gamma_value']
				else:
					svc_params['gamma'] = best_config['gamma_type']
			if kernel == 'poly':
				svc_params['degree'] = best_config.get('degree', 3)
			if kernel in ['poly', 'sigmoid']:
				svc_params['coef0'] = best_config.get('coef0', 0.0)
			best_model = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('svc', SVC(**svc_params))
			])
		elif model == "knn":
			knn_params = {
				'n_neighbors': best_config['n_neighbors'],
				'weights': best_config['weights'],
				'p': best_config['p'],
				'algorithm': best_config['algorithm'],
			}
			best_model = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('knn', KNeighborsClassifier(**knn_params))
			])
		elif model == "mlp":
			hls = tuple([best_config['hidden_layer_sizes']] * best_config['num_hidden_layers'])
			mlp_params = {
				'hidden_layer_sizes': hls,
				'activation': best_config['activation'],
				'alpha': best_config['alpha'],
				'learning_rate_init': best_config['learning_rate_init'],
				'solver': best_config['solver'],
				'batch_size': best_config['batch_size'],
				'max_iter': best_config['max_iter'],
				'random_state': 42
			}
			best_model = Pipeline([
				('imputer', SimpleImputer(strategy='mean')),
				('scaler', StandardScaler()),
				('mlp', MLPClassifier(**mlp_params))
			])

		tuner = tune.Tuner(
			model_training,
			tune_config=tune.TuneConfig(
				num_samples=tuning_rounds,
				search_alg=OptunaSearch(),
				scheduler=ASHAScheduler(),
				metric="weighted_f1_score",
				mode="max"
			),
			param_space=config
		)

		results = tuner.fit()

		best_result = results.get_best_result(
			metric="weighted_f1_score", mode="max")
		# Get a dataframe for the last reported results of all of the trials
		df = results.get_dataframe()

		for item in best_result.metrics.keys():
			print(f"{item} : {best_result.metrics[item]}")
		print(f"The corresponding config is {best_result.config}")

		data_dir = os.path.join(current_dir, f"{output}.csv")
		df.to_csv(data_dir, index=False)
		print(f"Training log saved at: {data_dir}")

		# Train final model with best config
		best_config = best_result.config
		kernel = best_config['kernel']
		svc_params = {
			'C': best_config['C'],
			'kernel': kernel,
			'probability': True,
			'class_weight': 'balanced',
			'random_state': 42,
			'shrinking': best_config['shrinking'],
			'tol': best_config['tol'],
			'cache_size': best_config['cache_size'],
			'max_iter': best_config['max_iter'],
		}
		
		# Add kernel-specific parameters
		if kernel in ['rbf', 'poly', 'sigmoid']:
			# Determine gamma value based on gamma_type
			if best_config['gamma_type'] == 'numeric':
				svc_params['gamma'] = best_config['gamma_value']
			else:
				svc_params['gamma'] = best_config['gamma_type']  # 'scale' or 'auto'
		if kernel == 'poly':
			svc_params['degree'] = best_config.get('degree', 3)
		if kernel in ['poly', 'sigmoid']:
			svc_params['coef0'] = best_config.get('coef0', 0.0)
		
		best_model = Pipeline([
			('imputer', SimpleImputer(strategy='mean')),  # Handle NaN values
			('scaler', StandardScaler()),
			('svc', SVC(**svc_params))
		])
		best_model.fit(X_train, y_train)
		
		y_pred = best_model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		weighted_f1score = f1_score(y_test, y_pred, average='weighted')
		macro_f1score = f1_score(y_test, y_pred, average='macro')
		print(f"Tuned Model Metrics:")
		print(f"Accuracy: {accuracy}")
		print(f"Weighted F1 score: {weighted_f1score}")
		print(f"Macro F1 score: {macro_f1score}")

	if no_weights:
		print("Model weights not saved")
	else:
		try:
			os.makedirs(os.path.dirname(weights_path), exist_ok=True)
			joblib.dump(best_model, weights_path)
			print(f"Updated SVM model saved at: {weights_path}")
		except Exception as e:
			print(f"Failed to save model weights: {e}")


if __name__ == "__main__":
	train_model()
