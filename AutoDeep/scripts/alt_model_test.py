import click


class CustomUsageMsg(click.Command):
	def format_usage(self, ctx, formatter):
		formatter.write_text("Usage: python alt_model_test.py [OPTIONS]")


def pipeline(step, estimator):
	"""Mean imputation and standardisation, then the model."""
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler

	return Pipeline([
		('imputer', SimpleImputer(strategy='mean')),
		('scaler', StandardScaler()),
		(step, estimator)
	])


def tuned_pipeline(model, config, seed):
	"""The pipeline for one tuning configuration (a point in that model's search space)."""
	from sklearn.svm import SVC
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.neural_network import MLPClassifier

	if model == "svm":
		kernel = config['kernel']
		svc_params = {
			'C': config['C'],
			'kernel': kernel,
			'probability': True,
			'class_weight': 'balanced',
			'random_state': seed,
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
		return pipeline('svc', SVC(**svc_params))
	if model == "knn":
		knn_params = {
			'n_neighbors': config['n_neighbors'],
			'weights': config['weights'],
			'p': config['p'],
			'algorithm': config['algorithm'],
		}
		return pipeline('knn', KNeighborsClassifier(**knn_params))
	if model == "mlp":
		hls = tuple([config['hidden_layer_sizes']] * config['num_hidden_layers'])
		mlp_params = {
			'hidden_layer_sizes': hls,
			'activation': config['activation'],
			'alpha': config['alpha'],
			'learning_rate_init': config['learning_rate_init'],
			'solver': config['solver'],
			'batch_size': config['batch_size'],
			'max_iter': config['max_iter'],
			'random_state': seed
		}
		return pipeline('mlp', MLPClassifier(**mlp_params))
	raise ValueError(f"Unknown model {model!r}")


@click.command(cls=CustomUsageMsg)
@click.option("--model", "-m", default="svm", type=click.Choice(["svm", "knn", "mlp"]), help="Model type: svm, knn, or mlp (neural network)")
@click.option("--targets_path", "-t", help="Path to targets file, or user_data.csv from an earlier training run", metavar="<str>", type=click.Path(exists=True, dir_okay=False))
@click.option("--no_db_data", "-n", is_flag=True, help="Flag that omits original dataset from training")
@click.option("--tuning_rounds", "-r", default=10, help="Number of tuning rounds: Default <10>", metavar="<int>")
@click.option("--output", "-o", default="model_training_log", help="Name of the tuning log CSV in the output directory", metavar="<str>")
@click.option("--no_weights", "-nw", is_flag=True, help="Skip saving the model (other outputs are still written)")
@click.option("--out_dir", help="Directory for the model, config, metrics and user data (default: models/<timestamp>)", metavar="<path>")
@click.option("--hyperparameters", "-hp", help="Path to hyperparameter configuration file in case of manual tuning", metavar="<str>", type=click.Path(exists=True, dir_okay=False))
@click.option("--seed", default=42, type=click.IntRange(0, 2 ** 31 - 1), help="Random seed for the data splits, the models and the tuning sampler (default: 42; saved in config.json)", metavar="<int>")
def train_model(model, no_db_data, tuning_rounds, output, targets_path, no_weights, out_dir, hyperparameters, seed):
	"""Trains an SVM, KNN or MLP model to compare with XGBoost (Table 2).

	Uses the same data as `AutoDeep train` (boosted_forest_training.py):
	- Loads training_data_pre/ unless `--no_db_data` is set, plus the labelled loci in `--targets_path`
	- Trains a naive pipeline (mean imputation, standardisation, model), then either uses manual
	  hyperparameters or tunes with Ray Tune and Optuna
	- Writes the model (<model>_model.pkl), config.json, metrics.json, user_data.csv and the tuning log
	  to `--out_dir`; nothing inside the AutoDeep installation is changed
	"""
	import json
	import os
	import sys
	import time
	import joblib
	import numpy as np
	import sklearn
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, f1_score
	from sklearn.svm import SVC
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.neural_network import MLPClassifier
	from boosted_forest_training import TRAINING_DATA, load_training_set, tuning_search_alg
	from hyperparameters import read_hyperparameter_file

	current_dir = os.getcwd()

	if "AutoDeepRun" not in os.path.basename(current_dir):
		print("Please run in AutoDeepRun directory")
		sys.exit(1)

	if no_db_data and targets_path is None:
		print("Error: No data to train on: Unset flag -n/--no_db_data or provide targets file")
		sys.exit(1)

	out_dir = out_dir or os.path.join("models", time.strftime("%Y%m%d-%H%M%S"))
	if os.path.exists(out_dir):
		print(f"Error: Output directory already exists: {out_dir}")
		sys.exit(1)
	os.makedirs(out_dir)
	print(f"Writing outputs to: {os.path.abspath(out_dir)}")

	# The same loci, features and encoding as AutoDeep train
	try:
		X, targets, data_info = load_training_set([targets_path] if targets_path else None, no_db_data, out_dir)
	except ValueError as e:
		print(f"Error: {e}")
		sys.exit(1)

	if data_info["n_rows"] == 0:
		print("No data available for training")
		sys.exit(1)

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, stratify=targets if len(np.unique(targets))>1 else None,
	                                                    random_state=seed)

	estimators = {
		"svm": (SVC, {'probability': True, 'class_weight': 'balanced', 'random_state': seed}),
		"knn": (KNeighborsClassifier, {}),
		"mlp": (MLPClassifier, {'max_iter': 200, 'random_state': seed}),
	}
	estimator_class, defaults = estimators[model]
	step = {"svm": "svc", "knn": "knn", "mlp": "mlp"}[model]

	print(f"Starting training (Naive {model.upper()})")
	base_pipeline = pipeline(step, estimator_class(**defaults))
	base_pipeline.fit(X_train, y_train)
	y_pred = base_pipeline.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"naive {model.upper()} accuracy post-train {accuracy}")
	metrics = {"naive_accuracy": float(accuracy)}

	if hyperparameters:
		print(f"Using manual hyperparameters for {model.upper()}")
		try:
			config = read_hyperparameter_file(hyperparameters)
		except ValueError as e:
			print(f"Error: {e}")
			sys.exit(1)
		allowed = estimator_class().get_params()
		ignored = sorted(k for k in config if k not in allowed)
		if ignored:
			print(f"Warning: ignoring hyperparameters that {estimator_class.__name__} does not take: {ignored}")
		config = {k: v for k, v in config.items() if k in allowed}
		best_model = pipeline(step, estimator_class(**{**defaults, **config}))
		best_model.fit(X_train, y_train)
		y_pred = best_model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		weighted_f1score = f1_score(y_test, y_pred, average='weighted')
		macro_f1score = f1_score(y_test, y_pred, average='macro')
		print(f"Manual-tuned Model Metrics:")
		print(f"Accuracy: {accuracy}")
		print(f"Weighted F1 score: {weighted_f1score}")
		print(f"Macro F1 score: {macro_f1score}")
		metrics.update(accuracy=float(accuracy), weighted_f1_score=float(weighted_f1score),
		               macro_f1_score=float(macro_f1score))
	else:
		print("Starting hyperparameter tuning")
		from ray import tune, train
		from ray.tune.schedulers import ASHAScheduler

		def model_training(config):
			weighted_f1_scores = []
			accuracies = []
			macro_f1_scores = []
			for i in range(10):
				# Every configuration is scored on the same ten splits
				X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, targets, test_size=0.2, stratify=targets,
				                                                                random_state=seed + i)
				cv_pipeline = tuned_pipeline(model, config, seed)
				cv_pipeline.fit(X_train_cv, y_train_cv)
				y_pred_cv = cv_pipeline.predict(X_test_cv)
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

		tuner = tune.Tuner(
			model_training,
			tune_config=tune.TuneConfig(
				num_samples=tuning_rounds,
				search_alg=tuning_search_alg(seed),
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

		data_dir = os.path.join(out_dir, f"{output}.csv")
		df.to_csv(data_dir, index=False)
		print(f"Training log saved at: {data_dir}")

		config = best_result.config
		metrics.update({key: float(value) for key, value in best_result.metrics.items()
		                if key in ("mean_accuracy", "std_accuracy", "weighted_f1_score", "std_weighted_f1_score",
		                           "macro_f1_score", "std_macro_f1_score")})
		metrics["accuracy"] = metrics["mean_accuracy"]

		# Train final model with best config
		best_model = tuned_pipeline(model, config, seed)
		best_model.fit(X_train, y_train)

		y_pred = best_model.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		weighted_f1score = f1_score(y_test, y_pred, average='weighted')
		macro_f1score = f1_score(y_test, y_pred, average='macro')
		print(f"Tuned Model Metrics:")
		print(f"Accuracy: {accuracy}")
		print(f"Weighted F1 score: {weighted_f1score}")
		print(f"Macro F1 score: {macro_f1score}")

	run_config = {
		"model": model,
		"hyperparameters": {key: (value.item() if hasattr(value, "item") else value) for key, value in config.items()},
		"seed": seed,
		"arguments": click.get_current_context().params,
		"training_data": None if no_db_data else TRAINING_DATA,
		"n_rows": data_info["n_rows"],
		"class_counts": data_info["class_counts"],
		"sklearn_version": sklearn.__version__,
		"created": time.strftime("%Y-%m-%d %H:%M:%S"),
	}
	with open(os.path.join(out_dir, "config.json"), "w") as f:
		json.dump(run_config, f, indent=2)
	with open(os.path.join(out_dir, "metrics.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	if no_weights:
		print("Model not saved (-nw)")
	else:
		model_path = os.path.join(out_dir, f"{model}_model.pkl")
		joblib.dump(best_model, model_path)
		print(f"{model.upper()} model saved at: {model_path}")


if __name__ == "__main__":
	train_model()
