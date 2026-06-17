# Context-Aware RAG Chatbot

## Project Overview

This project implements a Context-Aware Retrieval-Augmented Generation (RAG) chatbot that answers user questions based on information retrieved from a custom knowledge base. The system combines document retrieval using vector embeddings with Large Language Models (LLMs) to generate accurate and contextually relevant responses.

The chatbot uses ChromaDB as the vector database, Hugging Face embeddings for semantic search, Groq-hosted LLaMA 3.1 for response generation, and Streamlit for the user interface.

---

## Objective

The objective of this project is to build a context-aware chatbot capable of:

* Retrieving relevant information from uploaded documents
* Understanding user queries using semantic search
* Generating responses grounded in retrieved document content
* Maintaining conversational context across multiple interactions
* Demonstrating the complete RAG pipeline used in modern AI applications

---

## Dataset Used

**Custom PDF Knowledge Base**

* Sample document: Artificial Intelligence article (6 pages)
* Document loaded using LangChain PDF Loader
* Text split into overlapping chunks for efficient retrieval
* Chunks converted into vector embeddings and stored in ChromaDB

---

## Technologies Used

### Programming Language

* Python

### Libraries and Frameworks

* Streamlit
* LangChain
* ChromaDB
* Hugging Face Sentence Transformers
* Groq API
* python-dotenv

### AI Models

#### Embedding Model

* sentence-transformers/all-MiniLM-L6-v2

#### Large Language Model

* LLaMA 3.1 8B Instant (via Groq)

---

## Project Architecture

### Document Ingestion Pipeline

1. Load PDF documents
2. Extract text content
3. Split text into chunks
4. Generate vector embeddings
5. Store embeddings in ChromaDB

### Query Pipeline

1. User submits a question
2. Question converted into embedding
3. Similarity search performed in ChromaDB
4. Most relevant chunks retrieved
5. Context passed to LLaMA 3.1
6. Response generated and displayed

---

## Features

* Context-aware question answering
* Retrieval-Augmented Generation (RAG)
* Semantic document search
* Conversational memory
* Source attribution
* Streamlit-based chat interface
* ChromaDB vector storage
* Fast inference using Groq API
* Cached model loading for improved performance

---

## Project Structure

```text
Context-Aware-RAG-Chatbot/
│
├── app.py
├── ingest.py
├── sample.pdf
├── requirements.txt
├── README.md
├── .env
└── vectorstore/
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Context-Aware-RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Running the Project

### Step 1: Create Vector Database

```bash
python ingest.py
```

This will:

* Load the PDF
* Split text into chunks
* Generate embeddings
* Create the ChromaDB vector store

### Step 2: Launch Chatbot

```bash
streamlit run app.py
```

---

## Results and Findings

* Successfully implemented an end-to-end RAG pipeline
* Generated semantic embeddings using MiniLM
* Stored and retrieved document chunks using ChromaDB
* Produced context-aware responses using LLaMA 3.1
* Maintained conversational history for improved interactions
* Achieved accurate retrieval of document-specific information through semantic search

---

## Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embedding Models
* Prompt Engineering
* Semantic Search
* Large Language Models (LLMs)
* LangChain
* Streamlit Development
* API Integration
* AI Application Development

---

## Future Improvements

* Multi-document support
* Document upload through UI
* Hybrid search (keyword + semantic)
* Conversation persistence
* Citation highlighting
* Advanced reranking models

---

## Author

Mohammad Shafay

AI Engineer | Machine Learning Engineer | Computer Vision & Generative AI Enthusiast
