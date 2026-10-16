# Methods text changes

Sentences in the manuscript's Methods that must change because the code changed. One entry per change: item ID, old text, new text.

## B1. Read alignments from output.mrd

- **Old text:** TODO (Tejus): paste the sentence that says read alignments and counts were taken from miRDeep2's per-locus PDF reports.
- **New text:** "For each candidate locus, AutoDeep reads the aligned reads and their counts from miRDeep2's output.mrd file, which lists every read mapped to the precursor in precursor coordinates."
- **Note:** feature values are unchanged. On the C. pomonella run, 5' homogeneity and the number of distinct reads computed from output.mrd equal the PDF-derived training values for all 212 loci (difference < 1e-9).
