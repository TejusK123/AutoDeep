"""Read hyperparameter files as data; nothing in them is ever run as code.

Two formats are accepted:
- JSON: an object of name: value pairs, or a training run's config.json
  (its "hyperparameters" entry is used).
- Text: one "name value" pair per line; '#' starts a comment line. Values are
  Python literals read with ast.literal_eval: numbers, 'strings', (tuples),
  True/False/None.
"""
import ast
import json
import numbers

XGBOOST_ALLOWED = ['max_depth', 'min_child_weight', 'subsample', 'eta', 'n_estimators', 'gamma', 'base_score',
                   'alpha', 'lambda', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode']
XGBOOST_INTEGER = ['max_depth', 'n_estimators']
OBJECTIVE = "multi:softprob"


def read_hyperparameter_file(path):
    """Return {name: value} from a JSON or "name value" text file."""
    with open(path) as f:
        text = f.read()
    if text.lstrip().startswith('{'):
        try:
            config = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from None
        if isinstance(config.get("hyperparameters"), dict):
            config = config["hyperparameters"]
        return config

    config = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path} line {number}: expected 'name value', got {line!r}")
        name, value = parts
        try:
            config[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"{path} line {number}: value of {name} is not a number, "
                             f"'string', tuple or True/False/None: {value!r}") from None
    return config


def load_xgboost_hyperparameters(path):
    """Read a hyperparameter file and check it against XGBoost's tunable parameters."""
    config = read_hyperparameter_file(path)
    for name, value in config.items():
        if name == "objective":
            if value != OBJECTIVE:
                raise ValueError(f"objective must be {OBJECTIVE!r}, got {value!r}")
            continue
        if name not in XGBOOST_ALLOWED:
            raise ValueError(f"Invalid hyperparameter {name!r}; allowed: {XGBOOST_ALLOWED}")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if name in XGBOOST_INTEGER and (not isinstance(value, numbers.Integral) or value < 1):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return config
