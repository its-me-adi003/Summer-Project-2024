# import fitz  
# from PIL import Image
# import pytesseract

# tesseract_path="/usr/bin/tesseract"
# if tesseract_path:
#     print(tesseract_path)
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
#     pdf_path = "special_pages.pdf"
#     pdf_document = fitz.open(pdf_path)
#     for page_number in range(len(pdf_document)):
#         page = pdf_document.load_page(page_number)
#         pix = page.get_pixmap()
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         text = pytesseract.image_to_string(img)
#         with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
#             text_file.write(text)

#     print("OCR processing complete.")

# Doctr

import fitz  
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

pdf_path = "special_pages.pdf"
pdf_document = fitz.open(pdf_path)

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    
    result = model([img_np])
    
    if result.pages:
        page_text = []
        for block in result.pages[0].blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                page_text.append(line_text)
        text = "\n".join(page_text)
    else:
        text = ""

    with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(text)

print("OCR processing complete.")

#easyocr

# import fitz
# from PIL import Image
# import easyocr
# import numpy as np 

# reader = easyocr.Reader(['en'])

# pdf_path = "special_pages.pdf"
# pdf_document = fitz.open(pdf_path)

# for page_number in range(len(pdf_document)):
#     page = pdf_document.load_page(page_number)
#     pix = page.get_pixmap()

#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#     result = reader.readtext(np.array(img))  

#     page_text = [entry[1] for entry in result]

#     text = "\n".join(page_text)

#     with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
#         text_file.write(text)

# print("OCR processing complete.")


