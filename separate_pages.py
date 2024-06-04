from PyPDF2 import PdfReader, PdfWriter
input_pdf = PdfReader("96564_EXCAVATIONS-AT-KALIBANGAN-THE-HARAPPANS-1960-1969-PART-I-1-2.pdf")
output_pdf = PdfWriter()
list_of_pages=[1,5,6,11,12,23,24,33,34,35,36,39,47,59,62,72,87,97,108,159]
for page in list_of_pages:
    output_pdf.add_page(input_pdf.pages[page-1])
with open("special_pages.pdf", "wb") as output_file:
    output_pdf.write(output_file)
