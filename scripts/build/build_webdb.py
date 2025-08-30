# scripts/build/build_webdb.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- User-Agent pour éviter blocage ---
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# --- URLs sources ---
URLS = [
    "https://www.dofuspourlesnoobs.com/guide-forgemagie.html",
    "https://wiki-dofus.eu/w/Equipements",
]

def load_wiki_docs():
    """Charge les documents depuis les sites web et retourne la liste de docs."""
    docs = []
    for url in URLS:
        loader = WebBaseLoader(url, header_template=HEADERS)
        docs.extend(loader.load())
    return docs

if __name__ == "__main__":
    print("[INFO] Chargement des pages...")
    docs = load_wiki_docs()
    print(f"[OK] {len(docs)} documents récupérés.")

    # Split en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)
    print(f"[OK] {len(docs)} chunks générés.")

    # Embeddings HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Construction FAISS
    print("[INFO] Construction de la base vectorielle...")
    db = FAISS.from_documents(docs, embeddings)

    # Sauvegarde
    db.save_local("data/vectorstore")
    print("[OK] Base FAISS sauvegardée dans data/vectorstore/")