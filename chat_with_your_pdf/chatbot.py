import os
import time
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Configuration
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = api_key

    genai.configure(api_key=api_key)
    return api_key


# PDF Processing
def process_pdf(pdf_path):
    print(f"Processing {pdf_path}...")
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
    return text


# Text Chunking
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# Vector Store Creation
def create_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


# QA Chain Setup
def setup_qa_chain(vectorstore, api_key):
    template = """Answer based on context:
    {context}

    Question: {question}

    Answer concisely in 2-3 sentences. If unsure, say so."""

    prompt = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Current recommended model
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


def main():
    api_key = configure_gemini()
    vectorstore = None
    rag_chain = None

    while True:
        print("\n=== PDF Chatbot ===")
        print("1. Process PDF")
        print("2. Ask a question")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            # Process PDF
            pdf_path = input("Enter PDF file path: ").strip('"')
            if not os.path.exists(pdf_path):
                print("Error: File not found")
                continue

            try:
                text = process_pdf(pdf_path)
                chunks = chunk_text(text)
                vectorstore = create_vector_store(chunks, api_key)
                rag_chain = setup_qa_chain(vectorstore, api_key)
                print("PDF processed successfully!")
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")

        elif choice == "2":
            # Ask question
            if not vectorstore:
                print("Please process a PDF first (Option 1)")
                continue

            question = input("Enter your question: ").strip()
            if question.lower() == 'back':
                continue

            try:
                start_time = time.time()
                answer = rag_chain.invoke(question)
                elapsed = time.time() - start_time

                print(f"\nAnswer: {answer}")
                print(f"(Response time: {elapsed:.2f} seconds)")

                # Rate limiting
                if elapsed < 1.5:
                    time.sleep(1.5 - elapsed)
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again or check your API key")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()