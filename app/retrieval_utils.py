import os
from pinecone import Pinecone
import google.generativeai as genai

# Load env vars and initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Embed with Gemini
def get_embedding(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return response["embedding"]

# Answer questions using Pinecone search + Gemini
def answer_questions(questions):
    answers = []
    for q in questions:
        query_vector = get_embedding(q)
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {q}"

        response = genai.generate_content(prompt, model="gemini-pro")
        answers.append(response.text.strip())

    return answers
