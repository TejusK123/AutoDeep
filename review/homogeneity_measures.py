"""Compare ways of measuring 5' homogeneity on the training loci (B2, Q2).

Usage: python review/homogeneity_measures.py <testing_folder> > review/homogeneity_measures.txt

<testing_folder> holds the miRDeep2 run behind each training CSV (the local
testing folder in CLAUDE.local.md). Every measure is computed on the reads that
miRDeep2 assigns to the mature arm. Reported: toy cases, agreement between the
measures, how well each separates the curated labels on its own, how much each
shifts when the reads are thinned at random (as with lower sequencing depth),
and how much counting the first nucleotide as part of the start changes the
Simpson index. B2 uses the unbiased Simpson index of the start positions.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'AutoDeep', 'scripts'))
from mrd import parse_mrd, reads_by_arm  # noqa: E402

RUNS = {'Andreii': 'mirDeep_Andreii', 'Arabidopsis': 'mirDeep_Arabidopsis', 'Bemisia': 'mirDeep_Bemisicaa',
        'Drosophila': 'mirDeep_Drosophila', 'FAW': 'mirDeep_FAW', 'Human': 'mirDeep_Human',
        'Pomonella': 'mirdeep_Pomonella', 'Xylostella': 'mirdeep_Xylostella'}
LABEL = 'Candidate/Confident/no-mirna'
MEASURES = ['simpson', 'p_max', 'within1', 'H_bits', 'H_mm', 'H_norm', 'mad_nt']
HIGHER_IS_HOMOGENEOUS = {'simpson': 1, 'p_max': 1, 'within1': 1, 'H_bits': -1, 'H_mm': -1, 'H_norm': -1, 'mad_nt': -1}
SEED = 42


def simpson(groups):
    """Unbiased Simpson index: the probability that two reads, drawn without replacement, fall in the same group."""
    n = np.array(list(groups.values()), dtype=float)
    N = n.sum()
    return (n * (n - 1)).sum() / (N * (N - 1)) if N > 1 else np.nan


def measures(starts):
    """starts: {5' start: reads}."""
    pos = np.array(sorted(starts))
    n = np.array([starts[p] for p in pos], dtype=float)
    N, K = n.sum(), len(n)
    if N == 0:
        return {k: np.nan for k in MEASURES}
    p = n / N
    H = -(p * np.log2(p)).sum() + 0.0  # + 0.0 turns -0.0 into 0.0
    mode = pos[np.argmax(n)]
    return {
        'simpson': simpson(starts),                                           # P(two reads share a start)
        'p_max': p.max(),                                                    # share at the commonest start
        'within1': p[np.abs(pos - mode) <= 1].sum(),                          # share within 1 nt of it
        'H_bits': H,                                                          # Shannon entropy
        'H_mm': H + (K - 1) / (2 * N * np.log(2)),                            # with Miller-Madow correction
        'H_norm': H / np.log2(K) if K > 1 else 0.0,                           # entropy / log2(distinct starts)
        'mad_nt': (p * np.abs(pos - mode)).sum(),                             # mean distance from the commonest start
    }


def group(keys, counts):
    out = {}
    for k, c in zip(keys, counts):
        if c:
            out[k] = out.get(k, 0) + c
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos.dropna()), np.asarray(neg.dropna())
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def main(testing):
    print("Toy cases (reads at each 5' start):")
    toys = {'1000 at one start': {0: 1000}, '1000 + 10 stray reads': {0: 1000, **{i: 1 for i in range(3, 13)}},
            '500 + 500, 1 nt apart': {0: 500, 1: 500}, 'the same + 50 stray': {0: 500, 1: 500, 9: 50},
            '90% + 10% at +1 nt': {0: 900, 1: 100}, '90% + 10% scattered': {0: 900, **{i: 10 for i in range(3, 13)}},
            'scattered over 20 starts': {i: 10 for i in range(20)}}
    print(pd.DataFrame({k: measures(v) for k, v in toys.items()}).T[MEASURES].round(3).to_string())

    rng = np.random.default_rng(SEED)
    recs = []
    for species, run in RUNS.items():
        loci = {pid.replace(',', ''): locus for pid, locus in parse_mrd(os.path.join(testing, run, 'output.mrd')).items()}
        for f in sorted(glob.glob(os.path.join(REPO, 'AutoDeep', 'training_data_pre', f'original_data_{species}*.csv'))):
            for pid, label in pd.read_csv(f)[['provisional_id', LABEL]].itertuples(index=False):
                locus = loci[pid]
                mature = [(beg, aln[beg - 1], count) for (arm, beg, count), (aln, _) in zip(reads_by_arm(locus), locus.reads)
                          if arm == 'mature']
                begs = [b for b, _, _ in mature]
                counts = np.array([c for _, _, c in mature])
                rec = {'species': species, 'label': label, 'N': counts.sum(), **measures(group(begs, counts))}
                rec['simpson_with_first_base'] = simpson(group([(b, nt.lower()) for b, nt, _ in mature], counts))
                rec['first_base_mismatched'] = sum(c for _, nt, c in mature if nt.isupper())
                for frac in (0.25, 0.1):
                    draws = [measures(group(begs, rng.binomial(counts, frac))) for _ in range(5)]
                    for k in MEASURES:
                        rec[f'{k}@{frac}'] = np.nanmean([d[k] for d in draws])
                recs.append(rec)
    df = pd.DataFrame(recs)

    print(f"\n{len(df)} training loci; mature-arm reads per locus: median {df.N.median():.0f}, "
          f"fewer than 50: {(df.N < 50).sum()}, fewer than 2: {(df.N < 2).sum()}; "
          f"a single start: {(df.p_max == 1).sum()}")
    print('\nSpearman correlation between the measures:')
    print(df[MEASURES].corr(method='spearman').round(2).to_string())

    print('\nAUC of each measure alone (oriented so that higher = more homogeneous):')
    for k in MEASURES:
        s = df[k] * HIGHER_IS_HOMOGENEOUS[k]
        pairs = [('Confident', 'falsepositive'), ('Candidate', 'falsepositive'), ('Confident', 'Candidate')]
        print(f'  {k:8s} ' + '   '.join(f'{a} vs {b} {auc(s[df.label == a], s[df.label == b]):.3f}' for a, b in pairs))

    print('\nDepth: mean shift when reads are thinned to 25% and 10% (5 draws each), in SDs of the measure '
          'across loci, for all loci and for loci with fewer than 50 mature-arm reads; Spearman between full depth and 10%')
    low = df.N < 50
    for k in MEASURES:
        sd = df[k].std()
        shift = {frac: (df[f'{k}@{frac}'] - df[k]) / sd for frac in (0.25, 0.1)}
        rho = df[[k, f'{k}@0.1']].corr(method='spearman').iloc[0, 1]
        print(f'  {k:8s} 25%: {shift[0.25].mean():+.3f}   10%: {shift[0.1].mean():+.3f}   '
              f'10%, <50 reads: {shift[0.1][low].mean():+.3f}   Spearman {rho:.3f}')

    print("\nFirst nucleotide (output.mrd prints a mismatched base in upper case):")
    print(f"  mature-arm reads with a mismatched first base: {df.first_base_mismatched.sum() / df.N.sum():.2%}")
    diff = df.simpson_with_first_base - df.simpson
    print(f"  Simpson index with the first base as part of the start, minus position only: differs by more than "
          f"0.05 in {(diff.abs() > 0.05).sum()} loci; Spearman "
          f"{df[['simpson', 'simpson_with_first_base']].corr(method='spearman').iloc[0, 1]:.3f}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
