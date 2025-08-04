from fastapi import FastAPI, Request, Header, HTTPException
from app.models import HackRequest, HackResponse
from app.pdf_utils import process_pdf
from app.embedding_utils import embed_and_store
from app.retrieval_utils import answer_questions
import os
from dotenv import load_dotenv  # <-- Import dotenv

load_dotenv()  # <-- Load variables from .env

app = FastAPI()

@app.post("/hackrx/run", response_model=HackResponse)
async def run_hackrx(payload: HackRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {os.getenv('BEARER_TOKEN')}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        texts = process_pdf(payload.documents)
        embed_and_store(texts)
        answers = answer_questions(payload.questions)
        return {
            "success": True,
            "processing_info": "Answers generated successfully",
            "answers": answers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
