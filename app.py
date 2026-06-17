import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ──────────────────────────────────────────────
load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"
TOP_K           = 3

# ── Page Setup ──────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("**Knowledge Base:** AI Article (6 pages)")
    st.markdown("**Model:** LLaMA 3.1 8B via Groq")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector Store:** ChromaDB")
    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("DevelopersHub Corporation")
    st.caption("AI/ML Engineering Internship — Task 4")

# ── Load Resources (cached) ─────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

vectorstore = load_vectorstore()
client      = load_groq_client()

# ── Session State ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ───────────────────────────────────────────────
st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Ask me anything about the AI article in the knowledge base.")
st.divider()

# ── Render Chat History ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📄 Sources: Page(s) {msg['sources']}")

# ── Chat Input ───────────────────────────────────────────
question = st.chat_input("Ask a question about AI...")

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Retrieve relevant chunks
            docs = vectorstore.similarity_search(question, k=TOP_K)

            if not docs:
                answer = "⚠️ I couldn't find any relevant information in the knowledge base."
                sources = []
            else:
                # Build context from retrieved chunks
                context = "\n\n".join([doc.page_content for doc in docs])

                # Extract page numbers for attribution
                sources = sorted(set([
                    doc.metadata.get("page", "N/A") for doc in docs
                ]))

                # Build conversation history (last 4 turns)
                history_text = ""
                for turn in st.session_state.chat_history[-4:]:
                    history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

                # Final prompt
                prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not found in the context, say "I don't have enough information in the knowledge base to answer that."
Be concise, clear, and accurate.

--- DOCUMENT CONTEXT ---
{context}

--- CONVERSATION HISTORY ---
{history_text}
--- CURRENT QUESTION ---
{question}

Answer:"""

                try:
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful RAG assistant that answers only from provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=512
                    )
                    answer = response.choices[0].message.content

                    # Save to memory
                    st.session_state.chat_history.append({
                        "user": question,
                        "assistant": answer
                    })

                except Exception as e:
                    answer = f"⚠️ API Error: {str(e)}"
                    sources = []

            # Display answer
            st.markdown(answer)
            if sources:
                st.caption(f"📄 Sources: Page(s) {sources}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if 'sources' in locals() else []
    })