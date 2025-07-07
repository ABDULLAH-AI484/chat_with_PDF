# PDF Q&A Chatbot with Gemini API
A FastAPI web application that leverages Google's Gemini models through LangChain's powerful framework for document processing and intelligent question answering..

## Features
Upload PDF documents for processing

Ask questions about the PDF content

Get concise AI-generated answers

Simple web interface

Rate limiting to prevent API abuse

Google's lightweight yet powerful multimodal model

Optimized for speed and efficiency in Q&A tasks

Temperature setting: 0.3 (balanced creativity/accuracy)

LangChain's integration with Google's fastest Gemini model

## Technologies Used
Backend: FastAPI

AI Integration: Google Gemini API

PDF Processing: PyPDF2

Vector Database: FAISS (via LangChain)

Frontend: Jinja2 templating

## Troubleshooting
If you get API key errors, verify your GEMINI_API_KEY is set correctly

For PDF processing errors, ensure you're uploading valid PDF files

Check the server logs for detailed error messages

🚀 Quick Start:

Set your GEMINI_API_KEY

Run uvicorn main:app --reload

Upload a PDF and start asking questions!
