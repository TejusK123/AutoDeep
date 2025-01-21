from pdf2image import convert_from_path
import os
from PIL import Image

# Define the directory with your PDFs
# pdf_dir = r"E:\Biostuff\Research\MirDeep2\reference\pdfs_13_11_2018_t_11_47_25_Bemiscaa"
# output_dir = r"E:\Biostuff\Research\MirDeep2\reference\training_images"



def extract_images(pdf_dir, output_dir = "AutoDeepRun/miRNA_images"):
    # Convert a specific region (top-right) of each page to an image
    def extract_top_right_from_image(page_image, top_start=50, right_margin=200, crop_height=200):
        width, height_image = page_image.size
        # Crop the top-right corner with custom starting position and height
        top_right_corner = page_image.crop((width - right_margin, top_start, width, top_start + crop_height))
        return top_right_corner

    # Function to process each PDF
    def process_pdf(pdf_file):
        pages = convert_from_path(pdf_file, 300, first_page = 1, last_page = 1)  # 300 DPI resolution for high quality
        for i, page_image in enumerate(pages):
            # Extract the top-right corner
            top_right_image = extract_top_right_from_image(page_image, top_start  = 200, crop_height = 600, right_margin = 1500)  # Adjust the margin as needed
            
            # Save the cropped image
            img_filename = f"{os.path.splitext(os.path.basename(pdf_file))[0]}_page{i}_top_right.png"
            img_path = os.path.join(output_dir, img_filename)
            top_right_image.save(img_path)
            print(f"Saved {img_path}")

    # Loop over all PDFs in the directory
    if len(os.listdir(pdf_dir)) == 0:

        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
            
                process_pdf(pdf_path)
    
