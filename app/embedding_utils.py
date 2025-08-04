from google.generativeai import embed_content
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv  # <-- Import dotenv

load_dotenv() 

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def embed_and_store(text_chunks):
    for i, chunk in enumerate(text_chunks):
        embedding = embed_content(chunk, model="models/embedding-001")["embedding"]
        index.upsert([(f"doc-{i}", embedding, {"text": chunk})])
