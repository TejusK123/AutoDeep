
import re
import pandas as pd
import numpy as np
import sys
import os
from mrd import parse_mrd, five_prime_features, overhangs
from tqdm import tqdm
import warnings


current_dir = os.getcwd()

file = os.path.join(current_dir, "AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt")

#file = "AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt"

folding_energies = []
loci_names = []


print("Reading folding energies of potential miRNAs")

with open(file) as f:
		for i, line in enumerate(f.readlines()):
				if (i+1) % 3 == 1:
						loci_names.append(line[1:].strip())
				if (i+1) % 3 == 0:
						z = re.findall(r'\(\s*-?\d+\.\d+\)', line)[0]
						folding_energies.append(z)


#Note: gotta use RNAfold on entire pri-miRNA to get folding energies not just the pre-RNA: JK

# The 3' overhangs come from output.mrd below, on the structure miRDeep2 used (B3)

data = pd.DataFrame({'provisional_id' : loci_names, 'Folding Energy': [float(item[1:-1]) for item in folding_energies]})

data2 = pd.read_csv(os.path.join(current_dir, "AutoDeepRun/feature_engineered_miRNA_Deep_data_novel_miRNAs.csv"))

intersected_data = pd.merge(data, data2, on='provisional_id') #temporary intersection to make life easier

#print(intersected_data.head())

#---------------------------------------------------------------------
# Entropy-based 5' metrics (normalized_entropy, normalized_entropy_s).
# Not part of the shipped model's feature set, so they are not written to
# fully_formatted_data.csv; they match the newer data in training_data/.

def parse_mrd_file(mrd_filepath):
	"""
	Parse a .mrd file and extract alignment entries with their counts.
	Returns a list of entries.
	"""
	with open(mrd_filepath) as f:
		read_data = f.read().splitlines()
	
	entries = []
	temp = []
	
	for item in read_data:
		if item.startswith('>'):
			if temp:
				entries.append(temp)
			temp = [item]
		else:
			if not item == '':
				temp.append(item)
	
	if temp:
		entries.append(temp)
	
	return entries

def parse_seq_entry(seq_line):
	"""
	Parse a sequence entry line to extract sequence name, count, and alignment.
	Handles formats like: seq_0_x10<tab>ALIGNMENT and obs/exp formats.
	"""
	s = seq_line.strip()
	s = re.sub(r'\t.*$', '', s).strip()  # drop trailing tab/flag
	s = s.strip('\'"')  # remove surrounding quotes
	
	m = re.match(r'^(seq_[0-9]+)_x(\d+)\s+(.*)$', s)
	consensus = re.match(r'(exp|obs)\s(.+)$', s)
	
	if consensus:
		return 'consensus', 1, consensus.group(2).strip()
	if m:
		return m.group(1), int(m.group(2)), m.group(3).strip()
	
	# fallback
	parts = s.split(None, 1)
	if parts:
		m2 = re.match(r'^(seq_[0-9]+)_x(\d+)$', parts[0])
		if m2:
			alignment = parts[1].strip() if len(parts) > 1 else ''
			return m2.group(1), int(m2.group(2)), alignment
	
	return None, None, s

def build_dataframes_from_entries(entries_alignments):
	"""
	Build dataframes from alignment entries, dropping 'exp' consensus if 'obs' exists.
	"""
	dfs = []
	for idx, entry in enumerate(entries_alignments):
		has_obs = any(line.strip().startswith('obs') for line in entry)
		rows = []
		miRNA_id = ''
		for line in entry:
			if line.startswith('>'):
				miRNA_id = line[1:].strip()
			if has_obs and line.strip().startswith('exp'):
				continue  # skip exp if obs exists
			name, count, aln = parse_seq_entry(line)
			if name is not None:
				rows.append((name, count, aln))
		
		df = pd.DataFrame(rows, columns=['sequence_name', 'sequence_count', 'alignment'])
		dfs.append((df, miRNA_id))
	
	return dfs

def extract_five_prime_metrics(df):
	"""
	Extract 5' start position metrics from a dataframe of alignments.
	Returns homogeneity metrics based on 5' nucleotide position.
	"""
	five_prime_data = []
	

	# Skip consensus row (first row if it exists)
	for i, row in df.iloc[1:].iterrows():
		aln = row['alignment']
		count = row['sequence_count']
		
		# Find first nucleotide (A, C, G, T, U, N - case insensitive)
		nucleotide_pattern = r'[ACGTUNacgtun]'
		match = re.search(nucleotide_pattern, aln)
		
		if match:
			idx = match.start()
			nucleotide = match.group().upper()
		else:
			idx = -1
			nucleotide = 'N'
		
		consensus_seq = df.reset_index().loc[0,'alignment']

		location = consensus_seq[idx] if idx != -1 and idx < len(consensus_seq) else "N/A"
		size = consensus_seq.count(location) if location != "N/A" else 0



		five_prime_data.append({
			'five_prime_index': idx,
			'five_prime_nucleotide': nucleotide,
			'sequence_count': count,
			'alignment_location' : location,
			'mature_size' : size
		})
	
	return pd.DataFrame(five_prime_data)

def calculate_homogeneity_score(five_prime_df, miRNA_id=None):
	"""
	Calculate homogeneity metrics from 5' start position data.
	Aggregates counts by position (not by individual sequences).
	
	Returns a dict with:
	- max_prop: Proportion of reads at most abundant position
	- top1/2/3_prop: Cumulative proportion in top N positions
	- normalized_entropy: Shannon entropy normalized by max entropy (over positions)
	- gini_coefficient: Inequality measure (0=uniform, 1=concentrated)
	- n_distinct_positions: Number of different 5' start positions
	- coefficient_variation: Relative spread of position counts
	- total_count: Total reads aggregated
	"""

	if five_prime_df.empty:
		return None
	


	M_idx = five_prime_df.groupby('five_prime_index')['sequence_count'].sum().idxmax()
	alignment_location = five_prime_df.loc[five_prime_df['five_prime_index'] == M_idx, 'alignment_location'].iloc[0]
	if alignment_location == 'M' or alignment_location == 'S':
		miRNA_size = five_prime_df.loc[five_prime_df['five_prime_index'] == M_idx, 'mature_size'].iloc[0]
	else:
		miRNA_size = 22

	#print(f"miRNA size inferred from major position: {miRNA_size} (alignment location: {alignment_location})")
	miRNA_size = 20
	min_idx = max(0, M_idx - miRNA_size)
	max_idx = (M_idx + miRNA_size) #Guarenteed to be within bounds because we're looking at 5' end.
	
	#print(f"Major 5' start position: {M_idx} (alignment location: {alignment_location}), min_idx: {min_idx}, max_idx: {max_idx}")

	# AGGREGATION BY POSITION: Group sequences by 5' start position and sum counts
	position_counts = five_prime_df[(five_prime_df['five_prime_index'] >= min_idx) & (five_prime_df['five_prime_index'] <= max_idx)].groupby('five_prime_index')['sequence_count'].sum().values
	total = position_counts.sum()
	proportions = position_counts / total
	num_locis = five_prime_df.shape[0]
	# Metric 1: Maximum proportion (reads at most abundant position within window)
	max_prop = proportions.max()
	
	# Top N proportions (sorted by position count)
	sorted_position_counts = np.sort(position_counts)[::-1]  # descending
	top1_prop = sorted_position_counts[0] / total if len(sorted_position_counts) > 0 else 0
	top2_prop = np.sum(sorted_position_counts[:min(2, len(sorted_position_counts))]) / total
	top3_prop = np.sum(sorted_position_counts[:min(3, len(sorted_position_counts))]) / total
	
	# Metric 2: Normalized Shannon entropy (over aggregated position counts)
	entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
	n_distinct_positions = len(position_counts)
	max_entropy = np.log2(n_distinct_positions) if n_distinct_positions > 0 else 0
	normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
	
	# Metric 3: Gini coefficient (0=uniform, 1=concentrated) over positions
	sorted_counts = np.sort(position_counts)
	n = len(sorted_counts)
	gini = (2 * np.sum(np.arange(1, n+1) * sorted_counts)) / (n * sorted_counts.sum()) - (n + 1) / n
	
	# Metric 4: Coefficient of variation of position counts
	cv = np.std(position_counts) / np.mean(position_counts) if np.mean(position_counts) > 0 else 0
	
	#STAR homogeneity (temporary fix variables later rn overriding them)


	star_df = five_prime_df[~((five_prime_df['five_prime_index'] > M_idx - miRNA_size) & (five_prime_df['five_prime_index'] < M_idx + miRNA_size))]

	if star_df.empty:
		#print(f"Warning: No star reads found for miRNA {miRNA_id if miRNA_id else 'unknown'}. Skipping 5' homogeneity metrics for star strand.")
		return {
		'max_prop': max_prop,
		'top1_prop': top1_prop,
		'top2_prop': top2_prop,
		'top3_prop': top3_prop,
		'normalized_entropy': normalized_entropy,
		'gini_coefficient': gini,
		'n_distinct_positions': n_distinct_positions,
		'coefficient_variation': cv,
		'total_count': int(total),
		'num_locis': num_locis,
		'max_prop_s': np.nan,
		'top1_prop_s': np.nan,
		'top2_prop_s': np.nan,
		'top3_prop_s': np.nan,
		'normalized_entropy_s': np.nan,
		'gini_coefficient_s': np.nan,
		'coefficient_variation_s': np.nan,
	}

	S_idx = star_df.groupby('five_prime_index')['sequence_count'].sum().idxmax()
	alignment_location = star_df.loc[star_df['five_prime_index'] == S_idx, 'alignment_location'].iloc[0]
	if alignment_location == 'M' or alignment_location == 'S':
		miRNA_size = star_df.loc[star_df['five_prime_index'] == S_idx, 'mature_size'].iloc[0]
	else:
		miRNA_size = 22

	#print(f"miRNA size inferred from major position: {miRNA_size} (alignment location: {alignment_location})")
	miRNA_size = 20
	min_idx = max(0, S_idx - miRNA_size)
	max_idx = (S_idx + miRNA_size) #Guarenteed to be within bounds because we're looking at 5' end.
	
	#print(f"Major 5' start position: {S_idx} (alignment location: {alignment_location}), min_idx: {min_idx}, max_idx: {max_idx}")

	# AGGREGATION BY POSITION: Group sequences by 5' start position and sum counts
	position_counts = star_df[(star_df['five_prime_index'] >= min_idx) & (star_df['five_prime_index'] <= max_idx)].groupby('five_prime_index')['sequence_count'].sum().values
	total = position_counts.sum()
	proportions = position_counts / total
	num_locis = star_df.shape[0]
	# Metric 1: Maximum proportion (reads at most abundant position within window)
	max_prop_s = proportions.max()
	
	# Top N proportions (sorted by position count)
	sorted_position_counts = np.sort(position_counts)[::-1]  # descending
	top1_prop_s = sorted_position_counts[0] / total if len(sorted_position_counts) > 0 else 0
	top2_prop_s = np.sum(sorted_position_counts[:min(2, len(sorted_position_counts))]) / total
	top3_prop_s = np.sum(sorted_position_counts[:min(3, len(sorted_position_counts))]) / total
	
	# Metric 2: Normalized Shannon entropy (over aggregated position counts)
	entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
	n_distinct_positions = len(position_counts)
	max_entropy = np.log2(n_distinct_positions) if n_distinct_positions > 0 else 0
	normalized_entropy_s = entropy / max_entropy if max_entropy > 0 else 0
	
	# Metric 3: Gini coefficient (0=uniform, 1=concentrated) over positions
	sorted_counts = np.sort(position_counts)
	n = len(sorted_counts)
	gini_s = (2 * np.sum(np.arange(1, n+1) * sorted_counts)) / (n * sorted_counts.sum()) - (n + 1) / n
	
	# Metric 4: Coefficient of variation of position counts
	cv_s = np.std(position_counts) / np.mean(position_counts) if np.mean(position_counts) > 0 else 0







	return {
		'max_prop': max_prop,
		'top1_prop': top1_prop,
		'top2_prop': top2_prop,
		'top3_prop': top3_prop,
		'normalized_entropy': normalized_entropy,
		'gini_coefficient': gini,
		'n_distinct_positions': n_distinct_positions,
		'coefficient_variation': cv,
		'total_count': int(total),
		'num_locis': num_locis,
		'max_prop_s': max_prop_s,
		'top1_prop_s': top1_prop_s,
		'top2_prop_s': top2_prop_s,
		'top3_prop_s': top3_prop_s,
		'normalized_entropy_s': normalized_entropy_s,
		'gini_coefficient_s': gini_s,
		'coefficient_variation_s': cv_s,
	}

#---------------------------------------------------------------------
#Beginning of 5' processing inference, from miRDeep2's output.mrd

print("Searching for 5\' processing in potential miRNAs")

directory = sys.argv[1]

# Find output.mrd recursively in the miRDeep2 run directory
mrd_filepath = None
for root, dirs, files in os.walk(directory):
	if 'output.mrd' in files:
		mrd_filepath = os.path.join(root, 'output.mrd')
		break

if mrd_filepath is None:
	print(f"Error: output.mrd not found in {directory}", file=sys.stderr)
	sys.exit(1)

print(f"Found output.mrd at: {mrd_filepath}")
mrd_loci = parse_mrd(mrd_filepath)

loci = list(intersected_data['provisional_id'])
missing = [item for item in loci if item not in mrd_loci]
if missing:
	print(f"Error: {len(missing)} loci from result.csv are not in {mrd_filepath}, e.g. {missing[:3]}", file=sys.stderr)
	sys.exit(1)

five_prime_process_signal_selected_row_count_sum_over_mature_count_sum = []
num_locis = []
drosha_overhangs = []
dicer_overhangs = []
for item in tqdm(loci):
	homogeneity, loci_count = five_prime_features(mrd_loci[item].reads)
	five_prime_process_signal_selected_row_count_sum_over_mature_count_sum.append(homogeneity)
	num_locis.append(loci_count)
	# Signed 3' overhangs; blank for loci without star reads
	drosha, dicer = overhangs(mrd_loci[item])
	drosha_overhangs.append(drosha)
	dicer_overhangs.append(dicer)

signaling_df = pd.DataFrame({'provisional_id': loci, "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum" : five_prime_process_signal_selected_row_count_sum_over_mature_count_sum,  'num_locis' : num_locis})
intersected_data = pd.merge(signaling_df, intersected_data, on='provisional_id')
# "Bot" is the Drosha end (base of the stem), "Top" the Dicer end (loop side). They
# follow Folding Energy, as in the training data (AutoDeep train checks the column order).
overhang_df = pd.DataFrame({'3\' Overhang Bot': drosha_overhangs, '3\' Overhang Top': dicer_overhangs}, index=loci)
position = intersected_data.columns.get_loc('Folding Energy') + 1
for offset, name in enumerate(overhang_df.columns):
	intersected_data.insert(position + offset, name, intersected_data['provisional_id'].map(overhang_df[name]))

intersected_data.to_csv(os.path.join(current_dir, "AutoDeepRun/fully_formatted_data.csv"), index = False)
