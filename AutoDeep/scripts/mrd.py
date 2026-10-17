"""Read alignments from miRDeep2's output.mrd and the features computed from them.

Each locus in output.mrd starts with '>provisional_id' and lists every read
aligned to the precursor as '<id>_x<count>  <alignment>\t<mismatches>', where
the alignment is the read's sequence padded with '.' to the precursor's
coordinates. These are the same reads, in the same order, that miRDeep2 draws
in pdfs_*/<provisional_id>.pdf, which AutoDeep used to parse with pypdf.

The locus's 'exp' line marks each precursor position as flank (f), mature (M),
loop (l) or star (S), the parts miRDeep2 scores the locus on.
"""
import re
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

READ_LINE = re.compile(r'^(\S+)_x(\d+)\s+([.A-Za-z]+)(?:\s|$)')
EXP_LINE = re.compile(r'^exp\s+(\S+)')

# miRDeep2_core_algorithm.pl, test_query: a read belongs to the first of these
# whose region contains it, allowing it to start up to 2 nt before the region
# and end up to 5 nt after it.
ARMS = (('mature', 'M'), ('star', 'S'), ('loop', 'l'))
FUZZ_BEG = 2
FUZZ_END = 5


class Locus(NamedTuple):
    exp: Optional[str]
    reads: List[Tuple[str, int]]


def parse_mrd(path):
    """Return {provisional_id: Locus(exp, [(alignment, count), ...])}, reads in file order."""
    exps = {}
    loci = {}
    pid = None
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                pid = line[1:].strip()
                loci.setdefault(pid, [])
                continue
            if pid is None:
                continue
            m = EXP_LINE.match(line)
            if m:
                exps[pid] = m.group(1)
                continue
            m = READ_LINE.match(line)
            if m:
                loci[pid].append((m.group(3), int(m.group(2))))
    return {pid: Locus(exps.get(pid), reads) for pid, reads in loci.items()}


def arm_counts(locus):
    """Return {'mature', 'star', 'loop'} read counts, as miRDeep2 reports them in result.csv.

    Reads that fit none of the three regions, such as reads spanning two arms,
    count towards none.
    """
    if locus.exp is None:
        raise ValueError("locus has no exp line")
    regions = [(arm, locus.exp.index(mark) + 1, locus.exp.rindex(mark) + 1)
               for arm, mark in ARMS if mark in locus.exp]
    counts = {arm: 0 for arm, _ in ARMS}
    for aln, count in locus.reads:
        beg = len(aln) - len(aln.lstrip('.')) + 1
        end = len(aln.rstrip('.'))
        for arm, first, last in regions:
            if first - FUZZ_BEG <= beg and end <= last + FUZZ_END:
                counts[arm] += count
                break
    return counts


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
