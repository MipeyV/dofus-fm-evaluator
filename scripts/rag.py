from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# --- LLM ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Base vectorielle ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="rag_dofus", embedding_function=embeddings)

# --- RAG ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
)
