import fitz
import numpy as np
from PIL import Image
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

def get_overlap_area(rect1, rect2):
    """Calculate the overlap area between two rectangles"""
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    overlap_area = x_overlap * y_overlap
    return overlap_area

def filter_blocks(blocks):
    """Filter blocks to remove smaller blocks with significant overlap"""
    filtered_blocks = []
    for i in range(len(blocks)):
        keep = True
        for j in range(len(blocks)):
            if i != j:
                rect1 = blocks[i].coordinates
                rect2 = blocks[j].coordinates
                overlap_area = get_overlap_area(rect1, rect2)
                area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
                area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
                smaller_area = min(area1, area2)
                
                if overlap_area / smaller_area >= 0.8:
                    if area1 < area2:
                        keep = False
                        break
        if keep:
            filtered_blocks.append(blocks[i])
    return filtered_blocks

def insertion_sort1(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        
        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        j = i - 1
        while j >= 0 and arr[j][0] - key[0] > 10:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def insertion_sort2(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        
        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        j = i - 1
        while j >= 0 and arr[j][1] - key[1] > 10 and abs(arr[j][0] - key[0]) <= 10:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)

    # Print image dimensions
    print(f"Page {page_number + 1} image dimensions: width={pix.width}, height={pix.height}")

    # Use layoutparser to detect layout
    layout = layout_model.detect(img_np)

    # Filter blocks to remove overlapping ones
    layout.blocks = filter_blocks(layout._blocks)
    print(layout._blocks)

    # Visualize detected blocks
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    for block in layout.blocks:
        x_1, y_1, x_2, y_2 = map(int, block.coordinates)
        rect = patches.Rectangle((x_1, y_1), x_2 - x_1, y_2 - y_1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_1, y_1, f'{block.type}', color='red', fontsize=12)

    plt.title(f'Page {page_number + 1} Layout')
    plt.show()

    # Organize blocks by their types
    blocks_by_type = {}
    for block in layout.blocks:
        block_type = block.type
        if block_type not in blocks_by_type:
            blocks_by_type[block_type] = []
        blocks_by_type[block_type].append(block)

    # Collect all blocks' coordinates into img_lst
    img_lst = []
    for blocks in blocks_by_type.values():
        for block in blocks:
            x_1, y_1, x_2, y_2 = map(int, block.coordinates)
            img_lst.append([x_1, y_1, x_2, y_2])

    print(img_lst, "here is first image_list")
    insertion_sort1(img_lst)
    insertion_sort2(img_lst)
    print(img_lst, "here is image_list:(")

    # Write the blocks to the output file
    with open(f"page_{page_number + 1}.txt", "w", encoding="utf-8") as text_file:
            for e in img_lst:
                x_1, y_1, x_2, y_2 = e
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
