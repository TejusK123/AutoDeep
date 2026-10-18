"""Read alignments from miRDeep2's output.mrd and the features computed from them.

Each locus in output.mrd starts with '>provisional_id' and lists every read
aligned to the precursor as '<id>_x<count>  <alignment>\t<mismatches>', where
the alignment is the read's sequence padded with '.' to the precursor's
coordinates. These are the same reads, in the same order, that miRDeep2 draws
in pdfs_*/<provisional_id>.pdf, which AutoDeep used to parse with pypdf.

The locus's 'exp' line marks each precursor position as flank (f), mature (M),
loop (l) or star (S), the parts miRDeep2 scores the locus on. When star reads
exist, an 'obs' line marks the observed star (the most abundant star read)
instead; that is the star result.csv reports. 'pri_seq' and 'pri_struct' are
the precursor and its RNAfold structure, which miRDeep2 used to find the parts.
"""
import math
import re
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

READ_LINE = re.compile(r'^(\S+)_x(\d+)\s+([.A-Za-z]+)(?:\s|$)')
FIELD_LINE = re.compile(r'^(exp|obs|pri_seq|pri_struct)\s+(\S+)')

# miRDeep2_core_algorithm.pl, test_query: a read belongs to the first of these
# whose region contains it, allowing it to start up to 2 nt before the region
# and end up to 5 nt after it.
ARMS = (('mature', 'M'), ('star', 'S'), ('loop', 'l'))
FUZZ_BEG = 2
FUZZ_END = 5


class Locus(NamedTuple):
    exp: Optional[str]
    reads: List[Tuple[str, int]]
    obs: Optional[str] = None
    pri_seq: Optional[str] = None
    pri_struct: Optional[str] = None


def parse_mrd(path):
    """Return {provisional_id: Locus(exp, [(alignment, count), ...], obs, pri_seq, pri_struct)}, reads in file order."""
    fields = {}
    loci = {}
    pid = None
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                pid = line[1:].strip()
                loci.setdefault(pid, [])
                fields.setdefault(pid, {})
                continue
            if pid is None:
                continue
            m = FIELD_LINE.match(line)
            if m:
                fields[pid][m.group(1)] = m.group(2)
                continue
            m = READ_LINE.match(line)
            if m:
                loci[pid].append((m.group(3), int(m.group(2))))
    return {pid: Locus(reads=reads, **{k: fields[pid].get(k) for k in ('exp', 'obs', 'pri_seq', 'pri_struct')})
            for pid, reads in loci.items()}


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


def base_pairs(structure):
    """{position: partner} for a dot-bracket structure, 1-based."""
    pairs, stack = {}, []
    for pos, c in enumerate(structure, 1):
        if c == '(':
            stack.append(pos)
        elif c == ')':
            partner = stack.pop()
            pairs[pos], pairs[partner] = partner, pos
    return pairs


def overhangs(locus):
    """Return the signed 3' overhangs (Drosha end, Dicer end) of the mature-star duplex.

    The Drosha end is the base of the hairpin stem, the Dicer end the loop side.
    Only base pairs between the mature and the star count, because Drosha and
    Dicer cut off the flanks and the loop. At the Drosha end the outermost of
    these pairs is taken, at the Dicer end the innermost, and the overhang is
    the number of unpaired nucleotides beyond that pair on the strand whose 3'
    end is there, minus the number on the strand whose 5' end is there: +2 for
    a canonical cut, 0 for a blunt end, negative for a 5' overhang. The star is
    the observed one (obs line) and the structure is pri_struct. NaN for both
    when the locus has no star reads, because miRDeep2's star is then its
    prediction, or when the mature and star do not pair.
    """
    if locus.pri_struct is None:
        raise ValueError("locus has no pri_struct line")
    if locus.obs is None or 'M' not in locus.obs or 'S' not in locus.obs:
        return math.nan, math.nan
    # The strands on the 5' and 3' arms, whichever is the mature
    (beg5, end5), (beg3, end3) = sorted((locus.obs.index(mark) + 1, locus.obs.rindex(mark) + 1) for mark in 'MS')
    pairs = base_pairs(locus.pri_struct)
    duplex = sorted((i, pairs[i]) for i in range(beg5, end5 + 1) if beg3 <= pairs.get(i, 0) <= end3)
    if not duplex:
        return math.nan, math.nan
    (outer5, outer3), (inner5, inner3) = duplex[0], duplex[-1]
    drosha = (end3 - outer3) - (outer5 - beg5)
    dicer = (end5 - inner5) - (inner3 - beg3)
    return drosha, dicer
