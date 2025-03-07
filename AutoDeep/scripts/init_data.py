import os
import sys
from format_csv import format_csv
from extract_images import extract_images
from datetime import datetime
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
    for pdfdir in pdf_directories:
        extract_images(pdfdir)
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
