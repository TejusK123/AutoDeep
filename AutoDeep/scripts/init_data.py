import os
import sys
import pandas as pd
from datetime import datetime


def format_csv(csv_path):
    """Format mirDeep2 CSV file into structured miRNA data."""
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
    novel_data.columns = ['provisional_id', 'miRDeep2_score', 'estimated_probability_miRNA_candidate_is_true_positive', 'rfam_alert','total_read_count','mature_read_count','loop_read_count','star_read_count','significant_randfold_p-value','miRBase_miRNA','example_miRBase_miRNA_with_same_seed','UCSC_browser','NCBI_blastn','consensus_mature_sequence','consensus_star_sequence','consensus_precursor_sequeunce','precursor_coordinate'][:novel_data.shape[1]]
    novel_data.to_csv(output_file, index=False)



def find_csv_files(directory):
    # Verify if the provided path is valid
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory.")
        return
    
    # List to store the paths of all found CSV files
    csv_files = []
    pdf_directories = []    
    # Walk through the directory and find all .csv files
    for root, dirs, files in os.walk(directory):

        for dir in dirs:
            if dir.startswith('pdf'):
                dir_path = os.path.abspath(os.path.join(root, dir))
                pdf_directories.append(dir_path)
                print(f"Found 'pdf' directory: {dir_path}")
		
        for file in files:
            if file.endswith('.csv') and file.startswith('result'):
                file_path = os.path.join(root, file)
                csv_files.append(file_path)
                print(f"Found CSV file: {file_path}")
                
    # If no CSV files found, notify the user
    if not csv_files:
        print("No CSV files found in the specified directory. Please run AutoDeep --help to see the expected directory structure.")
        sys.exit(1)
    
    # Example of output - Here you can add any processing you need on each CSV
    for csv_file in csv_files:
        # Replace this print statement with your processing logic
        print(f"Processing file: {csv_file}")
        with open(csv_file, "r") as f:
            content = f.read()
        content = content.replace(",", "")
        with open(csv_file, "w") as f:
            f.write(content)
        format_csv(csv_file)	
        # e.g., load CSV with pandas or process data
# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
    else:
        current_dir = os.getcwd()
        directory = sys.argv[1]
        output_folder = os.path.join(current_dir, "AutoDeepRun")
        os.makedirs(output_folder, exist_ok = True)
        #os.makedirs("AutoDeepRun/miRNA_images", exist_ok = True)
        find_csv_files(directory)
