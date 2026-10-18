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


def expected_star(pairs, mature_beg, mature_end):
    """Return the (beg, end) of the star miRDeep2 expects for this mature, or None.

    Port of find_star in miRDeep2_core_algorithm.pl: the star pairs with the
    mature so that both ends have 2-nt 3' overhangs. Where the nucleotide it
    anchors on is unpaired, it moves into the duplex to the nearest paired one
    and compensates for the distance. None if the mature has no base pair.
    """
    last = mature_end - 2
    length = last - mature_beg + 1
    shift = next((k for k in range(length) if last - k in pairs), None)
    shift_end = next((k for k in range(length) if mature_beg + k in pairs), None)
    if shift is None or shift_end is None:
        return None
    return pairs[last - shift] - shift, pairs[mature_beg + shift_end] + shift_end + 2


def overhangs(locus):
    """Return the signed 3' overhangs (Drosha end, Dicer end) of the mature-star duplex.

    The Drosha end is the base of the hairpin stem, the Dicer end the loop side.
    At each end, the overhang is how far one strand's 3' end extends past the
    partner of the other strand's 5' end: +2 for a canonical cut, 0 for a blunt
    end, negative for a 5' overhang. It is 2 plus the offset of the observed
    star (the obs line) from the star miRDeep2 expects for 2-nt overhangs, on
    the structure in output.mrd. NaN for both when the locus has no star reads,
    because miRDeep2's star is then its prediction.
    """
    if locus.pri_struct is None:
        raise ValueError("locus has no pri_struct line")
    if locus.obs is None or 'M' not in locus.obs or 'S' not in locus.obs:
        return math.nan, math.nan
    mature_beg, mature_end = locus.obs.index('M') + 1, locus.obs.rindex('M') + 1
    star_beg, star_end = locus.obs.index('S') + 1, locus.obs.rindex('S') + 1
    expected = expected_star(base_pairs(locus.pri_struct), mature_beg, mature_end)
    if expected is None:
        return math.nan, math.nan
    # The star's 3' end faces the mature's 5' end; the mature's 3' end faces the star's 5' end
    at_mature_5p = 2 + star_end - expected[1]
    at_mature_3p = 2 + star_beg - expected[0]
    if mature_beg < star_beg:  # mature on the 5' arm, so its 5' end is at the Drosha end
        return at_mature_5p, at_mature_3p
    return at_mature_3p, at_mature_5p
