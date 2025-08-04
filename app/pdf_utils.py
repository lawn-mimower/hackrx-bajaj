import requests
from PyPDF2 import PdfReader
from io import BytesIO

def process_pdf(url):
    response = requests.get(url)
    pdf = PdfReader(BytesIO(response.content))
    texts = []

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            texts.append(content.strip())

    return texts
