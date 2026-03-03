
import re
import pandas as pd
import pypdf  # REMOVED: PyPDF dependency
from pypdf import PdfReader  # REMOVED: PyPDF dependency
import numpy as np
import sys
import os
from tqdm import tqdm
import warnings


current_dir = os.getcwd()

file = os.path.join(current_dir, "AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt")

#file = "AutoDeepRun/RNAfold_novel_precursor_miRNAs.txt"

folding_structure = []
folding_energies = []
three_prime_overhang = []
loci_names = []
precursor_sequence = []


print("Calculating folding energies and identifying 3' 2nt overhangs for potential miRNAs")

with open(file) as f:
		for i, line in enumerate(f.readlines()):
				if (i+1) % 3 == 2:
						precursor_sequence.append(line.strip())
				if (i+1) % 3 == 1:
						loci_names.append(line[1:].strip())
				if (i+1) % 3 == 0:
						z = re.findall(r'\(\s*-?\d+\.\d+\)', line)[0]
						folding_energies.append(z)

						line = (re.sub(r'\(\s*-?\d+\.\d+\)','', line)).strip()

						folding_structure.append(line)
						left = line.index('(')
						right = line[::-1].index(')')
						three_prime_overhang.append(abs(right - left))






data2 = pd.read_csv(os.path.join(current_dir, "AutoDeepRun/formatted_novel_miRNA.csv"))

data2['location'] = data2.apply(lambda row: re.search(row['consensus_mature_sequence'], row['consensus_precursor_sequeunce']).start() if re.search(row['consensus_mature_sequence'], row['consensus_precursor_sequeunce']) else None, axis=1)




#Note: gotta use RNAfold on entire pri-miRNA to get folding energies not just the pre-RNA: JK


structure_lengths = [len(item)%2 for item in folding_structure]



#test new method on a single sequence

#Idea: Find the absolute location of the mature and star sequences in the precursor sequence. Then, find the absolute location of all the unpaired nucleotides in the structure and match them with the mature and star sequences.
#Then, align the mature and star sequences with the unpaired nucleotides and extract the 2nt overhangs from the precursor sequence.

top_overhangs = []
mature_location = []
for i in range(len(folding_structure)):
	
	structure = folding_structure[i]
	precursor_sequence = data2['consensus_precursor_sequeunce'][i]
	mature_sequence = data2['consensus_mature_sequence'][i]
	star_sequence = data2['consensus_star_sequence'][i]


	mature_on_top = False
	if re.search(mature_sequence,precursor_sequence).start() < re.search(star_sequence,precursor_sequence).start():
	
		mature_on_top = True
	else:
	
		mature_on_top = False

	mature_location.append(mature_on_top)
	if mature_on_top:
		pertinent_interval_left = range(0, len(mature_sequence)+1)
		pertinent_interval_right = range(len(precursor_sequence) - len(star_sequence), len(precursor_sequence)+1)
	else:
		pertinent_interval_left = range(0, len(star_sequence)+1)
		pertinent_interval_right = range(len(precursor_sequence) - len(mature_sequence), len(precursor_sequence)+1)

	

	unpaired_nucleotides = (list(re.finditer('\.+', folding_structure[i])))
	
	Left = []
	if structure[0] != '.':
		Left.append('')
	Right = []

	for item in unpaired_nucleotides:
		item_range = list(range(item.span()[0], item.span()[1]))
		left_intersection = ([x for x in item_range if x in pertinent_interval_left])
		right_intersection = ([x for x in item_range if x in pertinent_interval_right])
	
		if len(left_intersection) != 0:
			Left.append(structure[left_intersection[0]: left_intersection[-1]+1])
		if len(right_intersection) != 0:
			Right.append(structure[right_intersection[0]: right_intersection[-1]+1])

	if len(Left) > len(Right):
		Right = [''] * (len(Left) - len(Right)) + Right



	if structure[-1] != '.':
		Right.append('')

	
	bot_3nt_overhang = abs(len(Left[0]) - len(Right[-1]))

	Left = Left[1:]
	Right = Right[:-1]
	top_offset = 0
	bot_offset = 0

	for j in range(len(Left)):
		diff = len(Left[j]) - len(Right[-j - 1])
		if j != len(Left) - 1:
			if diff < 0:
				top_offset += abs(diff)
			else:
				bot_offset += abs(diff)
		else:
			if Left[j] != '' and Right[-j - 1] != '':
				if diff < 0:
					top_offset += abs(diff)
				else:
					bot_offset += abs(diff)

	
	if mature_on_top:
		top = (' ' * (top_offset + bot_3nt_overhang) + mature_sequence)
		bot = (' ' * bot_offset + star_sequence[::-1])
	
		top_overhangs.append(abs(len(top) - len(bot)))
	else:
		top = (' ' * (top_offset + bot_3nt_overhang) + star_sequence)
		bot = (' ' * bot_offset + mature_sequence[::-1])

		top_overhangs.append(abs(len(top) - len(bot)))









#print(len(mature_location), len(top_overhangs), len(three_prime_overhang))
data = pd.DataFrame({'provisional_id' : loci_names, 'Folding Energy': [float(item[1:-1]) for item in folding_energies], '3\' Overhang Bot': three_prime_overhang, '3\' Overhang Top' : top_overhangs})

data2 = pd.read_csv(os.path.join(current_dir, "AutoDeepRun/feature_engineered_miRNA_Deep_data_novel_miRNAs.csv"))

intersected_data = pd.merge(data, data2, on='provisional_id') #temporary intersection to make life easier

#print(intersected_data.head())

#---------------------------------------------------------------------
#Beginning of 5' processing inference

# with warnings.catch_warnings():
# 	warnings.filterwarnings("ignore", category = DeprecationWarning)
# 	five_prime_process_signal_selected_row_count_sum_over_mature_count_sum = []
# 	five_prime_process_signal_top_loci_count_sum_over_mature_count_sum = []
# 	five_prime_process_signal_all_loci_row_count_over_mature_row_count = []
# 	num_locis = []

# 	loci = list(intersected_data['provisional_id'])

# 	print("Searching for 5\' processing in potential miRNAs")

# 	directory = sys.argv[1]
# 	pdf_dirs = []
# 	for root, dirs, files in os.walk(directory):
# 					for dir in dirs:
# 									if dir.startswith('pdf'):
# 													pdf_dirs.append(dir)

# 	target_dir = pdf_dirs[-1]


# 	for item in tqdm(loci):
# 					reader = PdfReader(directory + "/" + target_dir + "/" + item + ".pdf")
# 					num_pages = len(reader.pages)
# 					page = reader.pages[0]
# 					init_page = page.extract_text()
# 					alignments = [np.fromstring(item.split()[0], dtype = np.uint8) for item in init_page.split("\n") if item.count('.')/(len(item) + 0.00000001) > 0.5 and item.count('(') == 0]
# 					counts = [int(item.split()[1]) for item in init_page.split("\n") if item.count('.')/(len(item) + 0.00000001) > 0.5 and item.count('(') == 0]
# 					for i in range(0, len(reader.pages[1:])):
# 									cur_page = reader.pages[1:][i].extract_text().split('\n')
# 									for j, item in enumerate(cur_page):
# 													if item.count('.')/(len(item) + 0.00000001) > 0.5:
# 																	alignments.append(np.fromstring(item, dtype = np.uint8))
# 																	counts.append(int(cur_page[j+1]))
# 					alignments = np.array(alignments)
# 					counts = np.array(counts)
# 					leftmost_nuc = [(alignments[i, np.where(alignments[i,:] != 46)[0][0]], np.where(alignments[i,:] != 46)[0][0]) for i in range(alignments.shape[0])] # 46 is the ASCII code for a period



# 					selection_df = pd.DataFrame({'Leftmost Nucleotide': [item[0] for item in leftmost_nuc], 'Position': [item[1] for item in leftmost_nuc], 'Counts': counts})

# 					max_leftmost_nuc = selection_df.iloc[np.argmax(selection_df['Counts']),:]['Leftmost Nucleotide']

# 					max_position = selection_df.iloc[np.argmax(selection_df['Counts']),:]['Position']

# 					selected_rows = selection_df[(selection_df['Leftmost Nucleotide'] == max_leftmost_nuc) & (selection_df['Position'] == max_position)]



					
# 					if num_pages > selected_rows.shape[0]:
# 						num_pages = selected_rows.shape[0]
					
# 					top_loci = selected_rows.iloc[np.argpartition(selected_rows['Counts'], -num_pages)[-num_pages:],:]
# 					tdf = selection_df[(selection_df['Position'] - max_position) <= 20]


# 					loci_count = selection_df.shape[0]


# 					five_prime_process_signal_selected_row_count_sum_over_mature_count_sum.append(sum(selected_rows['Counts'])/sum(tdf['Counts']))
# 					five_prime_process_signal_top_loci_count_sum_over_mature_count_sum.append(sum(top_loci['Counts'])/sum(tdf['Counts']))
# 					five_prime_process_signal_all_loci_row_count_over_mature_row_count.append(selected_rows.shape[0]/tdf.shape[0])
# 					num_locis.append(loci_count)



# 	signaling_df = pd.DataFrame({'provisional_id': loci, "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum" : five_prime_process_signal_selected_row_count_sum_over_mature_count_sum,  'num_locis' : num_locis})
# 	intersected_data = pd.merge(signaling_df, intersected_data, on='provisional_id')

#---------------------------------------------------------------------
# Beginning of 5' processing inference (NEW: using .mrd file instead of PDF)

# OLD PDF-BASED IMPLEMENTATION (COMMENTED OUT) - Removed to eliminate PyPDF dependency
# See no_pdf.ipynb for the original implementation details

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

print("Searching for 5\' processing in potential miRNAs using .mrd file")

directory = sys.argv[1]

# Find output.mrd file recursively in directory (equivalent to: find <input_dir> -type f -name "output.mrd")
mrd_filepath = None
for root, dirs, files in os.walk(directory):
	if 'output.mrd' in files:
		mrd_filepath = os.path.join(root, 'output.mrd')
		break

if mrd_filepath is None:
	print(f"Error: output.mrd not found in {directory}")
	sys.exit(1)

print(f"Found output.mrd at: {mrd_filepath}")

# Parse entries and build dataframes
entries_alignments = parse_mrd_file(mrd_filepath)
dfs = build_dataframes_from_entries(entries_alignments)

# Extract 5' metrics for each entry
five_prime_metrics = []
loci = list(intersected_data['provisional_id'])

for df, miRNA_id in tqdm(dfs):
	# Extract 5' position metrics
	five_prime_df = extract_five_prime_metrics(df)
	
	# Calculate homogeneity score
	if not five_prime_df.empty:
		metrics = calculate_homogeneity_score(five_prime_df, miRNA_id)
		if metrics:
			metrics['provisional_id'] = miRNA_id if miRNA_id else 'unknown'
			five_prime_metrics.append(metrics)

# Create dataframe from metrics
if five_prime_metrics:
	signaling_df = pd.DataFrame(five_prime_metrics)
	
	# Merge with intersected_data
	# intersected_data = pd.merge(
	# 	signaling_df[['provisional_id', 'max_prop', 'top1_prop', 'top2_prop', 'top3_prop', 
	# 				  'normalized_entropy', 'gini_coefficient', 'n_distinct_positions', 
	# 				  'coefficient_variation', 'total_count']], 
	# 	intersected_data, 
	# 	on='provisional_id', 
	# 	how='left'
	# )
	# intersected_data = pd.merge(
	# 	signaling_df[['provisional_id','normalized_entropy','num_locis']], 
	# 	intersected_data, 
	# 	on='provisional_id', 
	# 	how='left'
	# )
	intersected_data = pd.merge(
		signaling_df[['provisional_id','normalized_entropy','normalized_entropy_s']], 
		intersected_data, 
		on='provisional_id', 
	)
else:
	print("Warning: No 5' metrics extracted from .mrd file")

intersected_data.to_csv(os.path.join(current_dir, "AutoDeepRun/fully_formatted_data.csv"), index = False)

