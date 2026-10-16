"""A1: explicit, file-independent encoding of flags and labels."""
import numpy as np
import pandas as pd
import pytest

from features import (
    LABELS,
    decode_features,
    decode_labels,
    encode_features,
    encode_labels,
)

RANDFOLD = "significant_randfold_p-value"
SEED = "homologous_seed_in_miRBase"
MATURE_U = "mature_5'u_or_3'u"


def frame(randfold, seed, mature_u):
    return pd.DataFrame({"miRDeep2_score": np.arange(len(randfold), dtype=float),
                         RANDFOLD: randfold, SEED: seed, MATURE_U: mature_u})


def test_all_yes_randfold_encodes_to_1():
    out = encode_features(frame(["yes", "yes"], [True, True], [True, True]))
    assert list(out[RANDFOLD]) == [1, 1]


def test_all_no_randfold_encodes_to_0():
    out = encode_features(frame(["no", "no"], [False, False], [False, False]))
    assert list(out[RANDFOLD]) == [0, 0]


@pytest.mark.parametrize("values, expected", [
    ([True, True], [1, 1]),
    ([False, False], [0, 0]),
    (["True", "True"], [1, 1]),
    (["False", "True"], [0, 1]),
])
def test_boolean_columns(values, expected):
    out = encode_features(frame(["yes", "no"], values, values))
    assert list(out[SEED]) == expected
    assert list(out[MATURE_U]) == expected


def test_other_columns_and_input_untouched():
    df = frame(["yes", "no"], [True, False], [False, True])
    out = encode_features(df)
    assert list(out["miRDeep2_score"]) == [0.0, 1.0]
    assert list(df[RANDFOLD]) == ["yes", "no"]


@pytest.mark.parametrize("column, bad", [
    (RANDFOLD, "maybe"),
    (RANDFOLD, np.nan),
    (RANDFOLD, "Yes"),
    (SEED, "yes"),
    (MATURE_U, np.nan),
])
def test_unknown_value_raises(column, bad):
    df = frame(["yes", "no"], [True, False], [True, False])
    df[column] = df[column].astype(object)
    df.loc[1, column] = bad
    with pytest.raises(ValueError, match=column):
        encode_features(df)


def test_missing_column_raises():
    with pytest.raises(ValueError, match=RANDFOLD):
        encode_features(frame(["yes"], [True], [True]).drop(columns=[RANDFOLD]))


def test_decode_round_trip():
    df = frame(["yes", "no"], [True, False], [False, True])
    out = decode_features(encode_features(df))
    assert list(out[RANDFOLD]) == ["yes", "no"]
    assert list(out[SEED]) == ["True", "False"]
    assert list(out[MATURE_U]) == ["False", "True"]


def test_label_order_is_fixed():
    assert LABELS == ["Candidate", "Confident", "falsepositive"]


def test_labels_encode_without_depending_on_which_classes_appear():
    assert list(encode_labels(pd.Series(["Confident", "falsepositive"]))) == [1, 2]
    assert list(encode_labels(pd.Series(["falsepositive"]))) == [2]


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="no-mirna"):
        encode_labels(pd.Series(["Candidate", "no-mirna"]))


def test_decode_labels():
    assert list(decode_labels(np.array([2, 0, 1]))) == ["falsepositive", "Candidate", "Confident"]
