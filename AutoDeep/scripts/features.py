"""Fixed encodings shared by training, inference and evaluation.

Codes never depend on which values appear in a file: a column that is
all "yes" still encodes to 1. Unknown values raise instead of being
silently given a code.
"""
import pandas as pd

YES_NO = {"yes": 1, "no": 0}
TRUE_FALSE = {True: 1, False: 0, "True": 1, "False": 0}

FLAG_MAPS = {
    "significant_randfold_p-value": YES_NO,
    "homologous_seed_in_miRBase": TRUE_FALSE,
    "mature_5'u_or_3'u": TRUE_FALSE,
}
FLAG_NAMES = {
    "significant_randfold_p-value": {1: "yes", 0: "no"},
    "homologous_seed_in_miRBase": {1: "True", 0: "False"},
    "mature_5'u_or_3'u": {1: "True", 0: "False"},
}

# Class index i is the model's output column i
LABELS = ["Candidate", "Confident", "falsepositive"]
# Name of the label column in the training data CSVs
LABEL_COLUMN = "Candidate/Confident/no-mirna"

# Each feature column's name in the paper's Table 1, for figures (B5 will rename the columns)
TABLE1_NAMES = {
    "Folding Energy": "Folding Energy",
    "3' Overhang Bot": "3' Overhang Bot",
    "3' Overhang Top": "3' Overhang Top",
    "miRDeep2_score": "miRDeep2 Score",
    "mirDeep_sensitivity": "miRDeep2 Sensitivity",
    "sensitivity_uncertainty": "miRDeep2 Sensitivity sd",
    "total_read_count": "Total Read Count",
    "mature_read_count": "Mature Read Count",
    "loop_read_count": "Loop Read Count",
    "star_read_count": "Star Read Count",
    "significant_randfold_p-value": "Significant Randfold p-value",
    "homologous_seed_in_miRBase": "Homologous Seed in miRBase",
    "consensus_mature_sequence_length": "Consensus Mature Sequence Length",
    "mature_5'u_or_3'u": "Mature 5'/3' U?",
    "consensus_star_sequence_length": "Consensus Star Sequence Length",
    "consensus_precursor_sequence_length": "Consensus Precursor Sequence Length",
    "num_locis": "Unique Alignments",
    "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum": "Five Prime Homogeneity",
}


def _lookup(values, mapping, column):
    out = []
    for value in values:
        try:
            out.append(mapping[value])
        except (KeyError, TypeError):
            raise ValueError(f"Unknown value {value!r} in column {column!r}; "
                             f"expected one of {list(mapping)}") from None
    return out


def encode_features(df):
    """Return a copy of df with the flag columns encoded as 0/1."""
    out = df.copy()
    for column, mapping in FLAG_MAPS.items():
        if column not in out.columns:
            raise ValueError(f"Missing feature column {column!r}")
        out[column] = pd.Series(_lookup(out[column], mapping, column), index=out.index, dtype="int64")
    return out


def decode_features(df):
    """Return a copy of df with 0/1 flag columns written back as text."""
    out = df.copy()
    for column, names in FLAG_NAMES.items():
        if column in out.columns:
            out[column] = pd.Series(_lookup(out[column], names, column), index=out.index, dtype=object)
    return out


def encode_labels(labels):
    """Map class names to fixed indices (Candidate 0, Confident 1, falsepositive 2)."""
    mapping = {name: i for i, name in enumerate(LABELS)}
    return pd.Series(_lookup(labels, mapping, "label"), index=getattr(labels, "index", None), dtype="int64")


def decode_labels(codes):
    """Map class indices back to class names."""
    return pd.Series(_lookup(codes, dict(enumerate(LABELS)), "label"), dtype=object)
