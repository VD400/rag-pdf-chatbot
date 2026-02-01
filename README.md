# 📄 PDF RAG Chatbot (Client-Server Architecture)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to "chat" with their PDF documents. Built with a **FastAPI** backend for processing and a **Streamlit** frontend for the UI.

## 🏗️ Architecture
- **Backend:** FastAPI (handles PDF processing, vector embeddings, and LLM interaction).
- **Frontend:** Streamlit (handles user UI and file uploads).
- **AI Stack:** LangChain, ChromaDB, HuggingFace Embeddings, and OpenRouter LLMs.

## 🚀 How to Run locally

### 1. Clone the repo
```bash
git clone [https://github.com/VD400/rag-pdf-chatbot.git](https://github.com/VD400/rag-pdf-chatbot.git)
cd rag-pdf-chatbot