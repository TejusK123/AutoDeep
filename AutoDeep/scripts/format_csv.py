import pandas as pd
import os


def format_csv(csv_path):
	current_dir = os.getcwd()
	output_file = os.path.join(current_dir, "AutoDeepRun/formatted_novel_miRNA.csv")
	data = pd.read_csv(csv_path)
	start_index = data.iloc[:,0][data.iloc[:,0] == 'novel miRNAs predicted by miRDeep2'].index[0]
	try:
		end_index = data.iloc[:,0][data.iloc[:,0] == 'mature miRBase miRNAs detected by miRDeep2'].index[0]
	except:
		end_index = data.shape[0]
	
	novel_data = (data.iloc[start_index:end_index,0]).reset_index(drop=True)
	novel_data = pd.DataFrame(novel_data)
	novel_data.rename(columns=lambda x: novel_data.iloc[0,0], inplace=True)
	novel_data = novel_data.iloc[2:,:]
	novel_data = novel_data.iloc[:, -1].str.split('\t', expand=True)
	print(novel_data)
	novel_data.columns = ['provisional_id', 'miRDeep2_score', 'estimated_probability_miRNA_candidate_is_true_positive', 'rfam_alert','total_read_count','mature_read_count','loop_read_count','star_read_count','significant_randfold_p-value','miRBase_miRNA','example_miRBase_miRNA_with_same_seed','UCSC_browser','NCBI_blastn','consensus_mature_sequence','consensus_star_sequence','consensus_precursor_sequeunce','precursor_coordinate'][:novel_data.shape[1]]
	novel_data.to_csv(output_file, index=False)
	
