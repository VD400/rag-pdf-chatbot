import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.config import Settings 
print(f"🔎 RUNNING CHROMA VERSION: {chromadb.__version__}")
load_dotenv()
app = FastAPI()


CHROMA_CLIENT = None
rag_chain = None

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global rag_chain, CHROMA_CLIENT
    
    print("--- STARTING UPLOAD ---") 
    file_location = f"temp_{file.filename}"
    
    try:
        # 1. Save File
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("1. File saved")

        # 2. Load PDF
        loader = PyPDFLoader(file_location)
        docs = loader.load()
        print(f"2. PDF Loaded: {len(docs)} pages")

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"3. Text Split: {len(splits)} chunks")

        # 4. Initialize Embeddings
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing!")
            
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token, 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("4. Embeddings initialized")

        # 5. Initialize ChromaDB 
        # We reuse the client if it exists to save memory
        if CHROMA_CLIENT is None:
            CHROMA_CLIENT = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        print("5. Chroma Client created")

        # 6. Create Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=CHROMA_CLIENT
        )
        print("6. Vectorstore built")

        # 7. Setup RAG Chain
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(
            model="liquid/lfm-2.5-1.2b-thinking:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        system_prompt = "You are a helpful assistant. Answer based on the context:\n\n{context}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("7. Chain ready")
        
        return {"status": "success", "message": "File processed successfully."}
    
    except Exception as e:
        print(f"CRITICAL ERROR TRACEBACK: {e}") # This will show in Render logs
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
        
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
        print("--- CLEANUP DONE ---")

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="Please upload a document first.")
    response = rag_chain.invoke({"input": request.query})
    return {"answer": response["answer"]}