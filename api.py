import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.config import Settings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
app = FastAPI()

rag_chain = None

class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global rag_chain
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(file_location)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=client 
        )
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model="liquid/lfm-2.5-1.2b-thinking:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"))
        
        system_prompt = "You are a helpful assistant. Answer concisely based on the context provided. If the answer does not exist in the context, just say so.\n\n{context}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return {"status" : "success", "message": "File processed and AI ready."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.post("/chat")
async def chat(request: ChatRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Please upload a document first.")
    response = rag_chain.invoke({"input": request.query})
    return {"answer": response["answer"]}

