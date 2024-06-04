# import fitz
# import numpy as np
# from PIL import Image
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# import layoutparser as lp
# from layoutparser.models import Detectron2LayoutModel

# # Initialize the OCR model
# model = ocr_predictor(pretrained=True)

# pdf_path = "special_pages.pdf"
# pdf_document = fitz.open(pdf_path)

# # Initialize the layout model
# layout_model = Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

# for page_number in range(len(pdf_document)):
#     page = pdf_document.load_page(page_number)
#     pix = page.get_pixmap()

#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     img_np = np.array(img)

#     # Print image dimensions
#     print(f"Page {page_number + 1} image dimensions: width={pix.width}, height={pix.height}")

#     # Use layoutparser to detect layout
#     layout = layout_model.detect(img_np)

#     # print(layout)
#     # Organize blocks by their types
#     blocks_by_type = {}
#     for block in layout:
#         block_type = block.type
#         if block_type not in blocks_by_type:
#             blocks_by_type[block_type] = []
#         blocks_by_type[block_type].append(block)

#     # Write the blocks to the output file
#     with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
#         for block_type, blocks in blocks_by_type.items():
#             text_file.write(f"Block Type {block_type}:\n")
#             for block in blocks:
#                 x_1, y_1, x_2, y_2 = map(int, block.coordinates)
#                 cropped_img = img_np[y_1:y_2, x_1:x_2]
#                 print(f"Cropping image at coordinates: {x_1}, {y_1}, {x_2}, {y_2}")

#                 result = model([cropped_img])

#                 if result.pages:
#                     page_text = []
#                     for blk in result.pages[0].blocks:
#                         for line in blk.lines:
#                             line_text = " ".join(word.value for word in line.words)
#                             page_text.append(line_text)
#                     text = "\n".join(page_text)
#                 else:
#                     text = ""

#                 text_file.write(text + "\n\n")  # Separate each block's content by a new line
#             text_file.write("\n\n")  # Separate each block type's section by two new lines

# print("OCR processing complete.")

import fitz
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

pdf_path = "special_pages.pdf"
pdf_document = fitz.open(pdf_path)

# Initialize the layout model
layout_model = Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)

    # Print image dimensions
    print(f"Page {page_number + 1} image dimensions: width={pix.width}, height={pix.height}")

    # Use layoutparser to detect layout
    layout = layout_model.detect(img_np)

    # Visualize detected blocks
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    for block in layout:
        x_1, y_1, x_2, y_2 = map(int, block.coordinates)
        rect = patches.Rectangle((x_1, y_1), x_2 - x_1, y_2 - y_1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_1, y_1, f'{block.type}', color='red', fontsize=12)

    plt.title(f'Page {page_number + 1} Layout')
    plt.show()

    # Organize blocks by their types
    blocks_by_type = {}
    for block in layout:
        block_type = block.type
        if block_type not in blocks_by_type:
            blocks_by_type[block_type] = []
        blocks_by_type[block_type].append(block)

    # Write the blocks to the output file
    with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
        for block_type, blocks in blocks_by_type.items():
            text_file.write(f"Block Type {block_type}:\n")
            for block in blocks:
                x_1, y_1, x_2, y_2 = map(int, block.coordinates)
                cropped_img = img_np[y_1:y_2, x_1:x_2]
                print(f"Cropping image at coordinates: {x_1}, {y_1}, {x_2}, {y_2}")

                result = model([cropped_img])

                if result.pages:
                    page_text = []
                    for blk in result.pages[0].blocks:
                        for line in blk.lines:
                            line_text = " ".join(word.value for word in line.words)
                            page_text.append(line_text)
                    text = "\n".join(page_text)
                else:
                    text = ""

                text_file.write(text + "\n\n")  # Separate each block's content by a new line
            text_file.write("\n\n")  # Separate each block type's section by two new lines

print("OCR processing complete.")
