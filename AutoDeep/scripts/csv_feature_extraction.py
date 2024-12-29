import pandas as pd
import os

current_dir = os.getcwd()

data_dir = os.path.join(current_dir, "AutoDeepRun/formatted_novel_miRNA.csv")

data = pd.read_csv(data_dir)
data.insert(2, "mirDeep_sensitivity", data['estimated_probability_miRNA_candidate_is_true_positive'].apply(lambda x: float(x.split("+/-")[0])))

data.insert(3, 'sensitivity_uncertainty', data['estimated_probability_miRNA_candidate_is_true_positive'].apply(lambda x: float(x.split("+/-")[1][:-1])))

data = data.drop(columns = ['estimated_probability_miRNA_candidate_is_true_positive'])



data = data.drop(columns = ['rfam_alert'])
#In all training data, rfam_alert is not present so we drop it.
#In general though, rfam alert means that there is high homology to rRNA, tRNA, snRNA, snoRNA, etc. and that the miRNA candidate is likely a false positive.


data = data.drop(columns = ['miRBase_miRNA'])
#In all training data, miRBase_miRNA is not present so we drop it.
#In general though, miRBase_miRNA means that the miRNA candidate is present in miRBase and is likely a true positive.


data.insert(9, 'homologous_seed_in_miRBase', data['example_miRBase_miRNA_with_same_seed'].apply(lambda x: True if x != '-' else False))
data = data.drop(columns = ['example_miRBase_miRNA_with_same_seed'])

#In all training data, UCSC_browser was never invoked in the prediction so we drop it.
data = data.drop(columns = ['UCSC_browser'])

#In all training data, NCBI_blastn was never invoked in the prediction so we drop it.
data = data.drop(columns = ['NCBI_blastn'])

data.insert(10, 'consensus_mature_sequence_length', data['consensus_mature_sequence'].apply(lambda x: len(x)))

data.insert(12, 'consensus_star_sequence_length', data['consensus_star_sequence'].apply(lambda x: len(x)))
data.insert(14, 'consensus_precursor_sequence_length', data['consensus_precursor_sequeunce'].apply(lambda x: len(x)))



#not sure of whether or not to identify 5'u and 3'u or just 5'u for now I'll do both

data.insert(11, 'mature_5\'u_or_3\'u', data['consensus_mature_sequence'].apply(lambda x: x[0] == 'u' or x[-1] == 'u'))

data = data.drop(columns = ['consensus_precursor_sequeunce', 'consensus_star_sequence', 'consensus_mature_sequence', 'precursor_coordinate'])



data_file = os.path.join(current_dir, "AutoDeepRun/feature_engineered_miRNA_Deep_data_novel_miRNAs.csv")
data.to_csv(data_file, index = False)



#For now, the direct sequence data for the mature, star, and precursor sequences are not used in the model. Definetely possible though with a RNN. 
#deepMiRGene one option for this.
