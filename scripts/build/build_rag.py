# scripts/build_rag.py
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from scripts.build.build_webdb import load_wiki_docs
from scripts.build.build_dofusdude import fetch_all_equipment, build_docs


def load_lexique_docs(path: str = "docs/dofus_lexique.txt"):
    """
    Charge le fichier de lexique custom et le découpe en chunks.
    Retourne une liste de Documents LangChain.
    """
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Aucun fichier lexique trouvé à {path}")
        return []

    print(f"[INFO] Chargement du lexique: {p}")
    text = p.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    return [Document(page_content=chunk, metadata={"source": "lexique"}) for chunk in chunks]


def build_rag():
    print("[INFO] Chargement Wiki...")
    wiki_docs = load_wiki_docs()

    print("[INFO] Chargement DofusDude (API équipements)...")
    items = fetch_all_equipment()
    dofusdude_docs = build_docs(items)

    print("[INFO] Chargement Lexique...")
    lexique_docs = load_lexique_docs()

    # On fusionne toutes les sources
    all_docs = wiki_docs + dofusdude_docs + lexique_docs

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_docs, embeddings, persist_directory="rag_dofus")

    print(f"[OK] Base vectorielle créée avec {len(all_docs)} documents.")


if __name__ == "__main__":
    build_rag()
