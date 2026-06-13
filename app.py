from groq import Groq
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("Loading vector database...")

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("RAG Chatbot Ready!")

while True:

    question = input("\nAsk a question (type exit to quit): ").strip()

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Use ONLY the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful RAG assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAssistant:")
    print(response.choices[0].message.content)