# embedder.py

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import torch
import concurrent.futures
from pdf2image import convert_from_path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_TOKENS_PER_CHUNK = 256
CHUNK_OVERLAP = 30

def process_page(page_info):
    """
    Takes a tuple (page_number, pdf_path, page_image_pil), performs visual
    extraction, and returns a list of chunk dictionaries for that page.
    """
    page_num, pdf_path, page_image_pil = page_info
    print(f"[Process {os.getpid()}] Starting page {page_num}...")

    page_image = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num)
        img_height, img_width, _ = page_image.shape
        pdf_width, pdf_height = page.rect.width, page.rect.height
        x_scale = pdf_width / img_width
        y_scale = pdf_height / img_height
        words = page.get_text("words")

    page_chunks = []
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for contour in sorted_contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blob_bbox = fitz.Rect(x * x_scale, y * y_scale, (x + w) * x_scale, (y + h) * y_scale)
        words_in_blob = [word for word in words if fitz.Rect(word[:4]).intersects(blob_bbox)]
        words_in_blob.sort(key=lambda w: (w[1], w[0]))

        if words_in_blob:
            chunk_text = " ".join([word[4] for word in words_in_blob])
            page_chunks.append({"text": chunk_text, "page_number": page_num + 1})

    print(f"[Process {os.getpid()}] Finished page {page_num}, found {len(page_chunks)} chunks.")
    return page_chunks

def process_and_embed_pdf(pdf_path: str, page_nums_to_process: list[int]):
    """
    Main function to process a PDF, embed its content, and upload to Pinecone.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        page_nums_to_process (list[int]): A list of 0-indexed page numbers to process.
    """
    if not all([PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME]):
        print("⚠️ Pinecone environment variables not set. Exiting.")
        return

    print("--- STEP 1: Initializing models and Pinecone ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_model.get_sentence_embedding_dimension(),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone setup complete.")

    print("\n--- STEP 2: Converting PDF pages to images ---")
    pdf_images_pil = convert_from_path(pdf_path, first_page=min(page_nums_to_process)+1, last_page=max(page_nums_to_process)+1)
    page_image_map = {i: img for i, img in zip(page_nums_to_process, pdf_images_pil)}

    print("\n--- STEP 3: Processing pages in parallel to extract chunks ---")
    pages_to_process_info = [(num, pdf_path, page_image_map[num]) for num in page_nums_to_process]
    all_pages_chunks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_page, pages_to_process_info)
        for page_result in results:
            all_pages_chunks.extend(page_result)
    
    print(f"\n✅ Total chunks extracted from all pages: {len(all_pages_chunks)}")

    print("\n--- STEP 4: Splitting oversized chunks ---")
    final_chunks_with_metadata = []
    for chunk_info in all_pages_chunks:
        tokens = tokenizer.encode(chunk_info['text'], add_special_tokens=False)
        if len(tokens) <= MAX_TOKENS_PER_CHUNK:
            final_chunks_with_metadata.append(chunk_info)
            continue
        
        step = MAX_TOKENS_PER_CHUNK - CHUNK_OVERLAP
        for i in range(0, len(tokens), step):
            sub_chunk_tokens = tokens[i : i + MAX_TOKENS_PER_CHUNK]
            sub_chunk_text = tokenizer.decode(sub_chunk_tokens)
            final_chunks_with_metadata.append({"text": sub_chunk_text, "page_number": chunk_info['page_number']})
            
    print(f"Total chunks after splitting: {len(final_chunks_with_metadata)}")

    print("\n--- STEP 5: Generating embeddings and uploading to Pinecone ---")
    texts_to_embed = [chunk['text'] for chunk in final_chunks_with_metadata]
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(final_chunks_with_metadata, embeddings)):
        vector_id = f"vec_{pdf_path}_{i}"
        metadata = {
            "text": chunk['text'],
            "page_number": chunk['page_number'],
            "original_document": pdf_path
        }
        vectors_to_upsert.append((vector_id, embedding.tolist(), metadata))

    # Upsert in batches to avoid overwhelming the connection
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1} to Pinecone.")

    print("\n✅ Embedding and upload process complete!")


if __name__ == '__main__':
    PDF_PATH = "policy.pdf"
    # Define the list of page numbers you want to process (0-indexed)
    PAGE_NUMS_TO_PROCESS = list(range(0, 25))

    process_and_embed_pdf(pdf_path=PDF_PATH, page_nums_to_process=PAGE_NUMS_TO_PROCESS)