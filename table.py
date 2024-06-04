import fitz
import numpy as np
from PIL import Image
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
import csv
from doctr.models import ocr_predictor

# Initialize the OCR model
ocr_model = ocr_predictor(pretrained=True)

# Path to the PDF file
pdf_path = "table.pdf"
pdf_document = fitz.open(pdf_path)

# Initialize the layout model
layout_model = Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

# Output CSV file
output_csv = "extracted_table.csv"

# Function to extract text from image using doctr
def extract_text_from_image(cropped_img):
    result = ocr_model([cropped_img])
    page_text = []
    if result.pages:
        for blk in result.pages[0].blocks:
            for line in blk.lines:
                line_text = " ".join(word.value for word in line.words)
                page_text.append(line_text)
    text = " ".join(page_text)  # Join all the text into a single string
    return text

# Process each page in the PDF
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)

        # Print image dimensions
        print(f"Page {page_number + 1} image dimensions: width={pix.width}, height={pix.height}")

        # Use layoutparser to detect layout
        layout = layout_model.detect(img_np)

        # Print the detected layout blocks for debugging
        print(f"Page {page_number + 1} Layout:")
        for block in layout:
            print(block)

        # Assuming type '0' to '4' correspond to different table cells based on the provided output
        table_blocks = [b for b in layout if b.type in range(5)]

        if not table_blocks:
            print(f"No table blocks found on page {page_number + 1}")
            continue

        # Sort blocks by their position to identify rows and columns
        table_blocks.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))  # Sort by y (top) and then by x (left)

        current_row_y = None
        row_data = []

        for block in table_blocks:
            x_1, y_1, x_2, y_2 = map(int, block.coordinates)
            cropped_img = img_np[y_1:y_2, x_1:x_2]

            # Extract text from the cropped image
            cell_text = extract_text_from_image(cropped_img).replace('\n', ' ')
            print(f"Extracted text: {cell_text}")

            # Group cells by rows based on their y-coordinate
            if current_row_y is None:
                current_row_y = y_1

            if y_1 > current_row_y + 10:  # Threshold to determine a new row
                writer.writerow(row_data)  # Write the collected row data to the CSV file
                row_data = []
                current_row_y = y_1

            row_data.append(cell_text)

        # Write the last row data
        if row_data:
            writer.writerow(row_data)

print("Table extraction and CSV writing complete.")
