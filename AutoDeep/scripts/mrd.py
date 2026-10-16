"""Read alignments from miRDeep2's output.mrd and the 5' features computed from them.

Each locus in output.mrd starts with '>provisional_id' and lists every read
aligned to the precursor as '<id>_x<count>  <alignment>\t<mismatches>', where
the alignment is the read's sequence padded with '.' to the precursor's
coordinates. These are the same reads, in the same order, that miRDeep2 draws
in pdfs_*/<provisional_id>.pdf, which AutoDeep used to parse with pypdf.
"""
import re

import numpy as np
import pandas as pd

READ_LINE = re.compile(r'^(\S+)_x(\d+)\s+([.A-Za-z]+)(?:\s|$)')


def parse_mrd(path):
    """Return {provisional_id: [(alignment, count), ...]} in file order."""
    loci = {}
    reads = None
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                reads = loci.setdefault(line[1:].strip(), [])
                continue
            m = READ_LINE.match(line)
            if m and reads is not None:
                reads.append((m.group(3), int(m.group(2))))
    return loci


def five_prime_features(reads):
    """Return (5' homogeneity, number of distinct reads) for one locus.

    Homogeneity is the share of reads that start at the same position and
    with the same nucleotide as the most abundant read, out of all reads that
    start no more than 20 nt after it. Ties for most abundant read go to the
    first in file order. This reproduces the pypdf-based definition exactly.
    """
    if not reads:
        raise ValueError("locus has no aligned reads")
    positions = [len(aln) - len(aln.lstrip('.')) for aln, _ in reads]
    df = pd.DataFrame({
        'nucleotide': [aln[p] for (aln, _), p in zip(reads, positions)],
        'position': positions,
        'count': [count for _, count in reads],
    })
    top = df.iloc[np.argmax(df['count'].to_numpy())]
    selected = df[(df['nucleotide'] == top['nucleotide']) & (df['position'] == top['position'])]
    window = df[(df['position'] - top['position']) <= 20]
    return selected['count'].sum() / window['count'].sum(), df.shape[0]
