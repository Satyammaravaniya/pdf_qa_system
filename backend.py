from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile
import uvicorn
import os
from pathlib import Path
import time
from pylatexenc.latex2text import LatexNodes2Text
from pdf2image import convert_from_path
from typing import Optional
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai import ImageURLChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

app = FastAPI(title="PDF Math & Text Question Answering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Configure Google Generative AI once
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class QuestionRequest(BaseModel):
    question: str
    processed_text: str

class ProcessResponse(BaseModel):
    status: str
    processed_text: Optional[str] = None
    message: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str

def process_pdf_for_math(pdf_path: str) -> str:
    """Process PDF with mathematical content using Mistral's OCR."""
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    client = Mistral(api_key=MISTRAL_API_KEY)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_file, dpi=300, poppler_path=r"C:\\Users\\HP\\Downloads\\poppler-24.08.0\\Library\\bin")
        all_text = []
        
        for i, image in enumerate(images):
            temp_img_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(temp_img_path, "PNG")
            
            with open(temp_img_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
            
            base64_data_url = f"data:image/png;base64,{encoded}"
            
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )
            
            page_text = image_response.pages[0].markdown
            all_text.append(page_text)
    
    return "\n\n".join(all_text)

def convert_latex_in_text(content: str):
    """Convert LaTeX content to plain text."""
    try:
        plain_text = LatexNodes2Text().latex_to_text(content)
        plain_text = plain_text.replace("sqrt", "√")
        return plain_text
    except Exception:
        return content

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

@app.post("/process_pdf/", response_model=ProcessResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Endpoint to process uploaded PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.file.read())
            pdf_path = tmp_file.name
        
        raw_ocr_text = process_pdf_for_math(pdf_path)
        processed_text = convert_latex_in_text(raw_ocr_text)
        
        return ProcessResponse(
            status="success",
            processed_text=processed_text,
            message="PDF processed successfully"
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message=f"An error occurred: {str(e)}"
        )
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.post("/ask_question/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint to answer questions about processed PDF content."""
    try:
        text_chunks = get_text_chunks(request.processed_text)
        
        get_vector_store(text_chunks)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = vector_db.similarity_search(request.question)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
        If the answer is not in the provided context, just say, "Answer is not available in the context." 
        Don't provide a wrong answer.

        Context:
        {context}

        Question: 
        {request.question}

        Answer:
        """
        
        response = model.generate_content(prompt)
        return AnswerResponse(answer=response.text)
    
    except Exception as e:
        return AnswerResponse(answer=f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)