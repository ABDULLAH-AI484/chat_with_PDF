import os
import time
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables to store the vectorstore and chain
vectorstore = None
rag_chain = None


def configure_gemini():
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    return api_key


def process_pdf(file):
    print("Processing PDF...")
    reader = PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages)
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


def setup_qa_chain(vectorstore, api_key):
    template = """Answer based on context:
    {context}

    Question: {question}

    Answer concisely in 2-3 sentences. If unsure, say so."""

    prompt = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

    retriever = vectorstore.as_retriever()

    return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, rag_chain
    try:
        api_key = configure_gemini()
        contents = await file.read()

        # Save temporarily to process
        with open("temp.pdf", "wb") as f:
            f.write(contents)

        text = process_pdf("temp.pdf")
        chunks = chunk_text(text)
        vectorstore = create_vector_store(chunks, api_key)
        rag_chain = setup_qa_chain(vectorstore, api_key)

        os.remove("temp.pdf")  # Clean up
        return {"status": "success", "message": "PDF processed successfully!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global rag_chain
    if not rag_chain:
        return {"status": "error", "message": "Please upload a PDF first"}

    try:
        start_time = time.time()
        answer = rag_chain.invoke(question)
        elapsed = time.time() - start_time

        # Rate limiting
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)

        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Add this at the bottom of your main.py
if __name__ == "__main__":
    api_key = configure_gemini()
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
