
import re
import pandas as pd
import pypdf
from pypdf import PdfReader
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
data = pd.DataFrame({'provisional_id' : loci_names, 'Folding Energy': [float(item[1:-1]) for item in folding_energies], '3\' Overhang Bot': three_prime_overhang, '3\' Overhang Top' : top_overhangs, 'mature_seq_on_top' : mature_location})

data2 = pd.read_csv(os.path.join(current_dir, "AutoDeepRun/feature_engineered_miRNA_Deep_data_novel_miRNAs.csv"))

intersected_data = pd.merge(data, data2, on='provisional_id') #temporary intersection to make life easier



#---------------------------------------------------------------------
#Beginning of 5' processing inference

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = DeprecationWarning)
	five_prime_process_signal_selected_row_count_sum_over_mature_count_sum = []
	five_prime_process_signal_top_loci_count_sum_over_mature_count_sum = []
	five_prime_process_signal_all_loci_row_count_over_mature_row_count = []
	num_locis = []

	loci = list(intersected_data['provisional_id'])

	print("Searching for 5\' processing in potential miRNAs")

	directory = sys.argv[1]
	pdf_dirs = []
	for root, dirs, files in os.walk(directory):
					for dir in dirs:
									if dir.startswith('pdf'):
													pdf_dirs.append(dir)

	target_dir = pdf_dirs[-1]


	for item in tqdm(loci):
					reader = PdfReader(directory + "/" + target_dir + "/" + item + ".pdf")
					num_pages = len(reader.pages)
					page = reader.pages[0]
					init_page = page.extract_text()
					alignments = [np.fromstring(item.split()[0], dtype = np.uint8) for item in init_page.split("\n") if item.count('.')/(len(item) + 0.00000001) > 0.5 and item.count('(') == 0]
					counts = [int(item.split()[1]) for item in init_page.split("\n") if item.count('.')/(len(item) + 0.00000001) > 0.5 and item.count('(') == 0]
					for i in range(0, len(reader.pages[1:])):
									cur_page = reader.pages[1:][i].extract_text().split('\n')
									for j, item in enumerate(cur_page):
													if item.count('.')/(len(item) + 0.00000001) > 0.5:
																	alignments.append(np.fromstring(item, dtype = np.uint8))
																	counts.append(int(cur_page[j+1]))
					alignments = np.array(alignments)
					counts = np.array(counts)
					leftmost_nuc = [(alignments[i, np.where(alignments[i,:] != 46)[0][0]], np.where(alignments[i,:] != 46)[0][0]) for i in range(alignments.shape[0])] # 46 is the ASCII code for a period



					selection_df = pd.DataFrame({'Leftmost Nucleotide': [item[0] for item in leftmost_nuc], 'Position': [item[1] for item in leftmost_nuc], 'Counts': counts})

					max_leftmost_nuc = selection_df.iloc[np.argmax(selection_df['Counts']),:]['Leftmost Nucleotide']

					max_position = selection_df.iloc[np.argmax(selection_df['Counts']),:]['Position']

					selected_rows = selection_df[(selection_df['Leftmost Nucleotide'] == max_leftmost_nuc) & (selection_df['Position'] == max_position)]


					if num_pages > len(selected_rows):
						num_pages = len(selected_rows)

					top_loci = selected_rows.iloc[np.argpartition(selected_rows['Counts'], -num_pages)[-num_pages:],:]
					tdf = selection_df[(selection_df['Position'] - max_position) <= 20]


					loci_count = selection_df.shape[0]


					five_prime_process_signal_selected_row_count_sum_over_mature_count_sum.append(sum(selected_rows['Counts'])/sum(tdf['Counts']))
					five_prime_process_signal_top_loci_count_sum_over_mature_count_sum.append(sum(top_loci['Counts'])/sum(tdf['Counts']))
					five_prime_process_signal_all_loci_row_count_over_mature_row_count.append(selected_rows.shape[0]/tdf.shape[0])
					num_locis.append(loci_count)



	signaling_df = pd.DataFrame({'provisional_id': loci, "five_prime_process_signal_selected_row_count_sum_over_mature_count_sum" : five_prime_process_signal_selected_row_count_sum_over_mature_count_sum,  'num_locis' : num_locis})
	intersected_data = pd.merge(signaling_df, intersected_data, on='provisional_id')


	intersected_data.to_csv(os.path.join(current_dir, "AutoDeepRun/fully_formatted_data.csv"), index = False)

