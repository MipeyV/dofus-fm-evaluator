# scripts/cli_chat_with_rag.py
import json
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from src.core.item_models import ItemTemplate, StatDefinition


# --- LLM Ollama ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Chargement de la base FAISS ---
print("[INFO] Chargement de la base vectorielle...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Chargement de la base Chroma ---
print("[INFO] Chargement de la base vectorielle (Chroma)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="rag_dofus", 
    embedding_function=embeddings
)

# --- RAG Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(
        search_kwargs={"k": 10},
        filter={"source": "lexique"}
    ),
    chain_type="stuff",
)

# --- Fonction utilitaire pour tools.py ---
def document_to_itemtemplate(doc):
    stats_defs = {}
    stats_dict = doc.metadata.get("stats", {})

    # ⚠️ si c’est une string JSON
    if isinstance(stats_dict, str):
        try:
            stats_dict = json.loads(stats_dict)
        except Exception:
            stats_dict = {}

    for stat, vals in stats_dict.items():
        stats_defs[stat] = StatDefinition(
            stat,
            vals.get("min", 0),
            vals.get("max", 0),
            vals.get("weight", 1)
        )

    return ItemTemplate(name=doc.metadata.get("name", doc.page_content), stats=stats_defs)

# --- Debug ---
def debug_search(query: str):
    docs = db.similarity_search(query, k=3)
    print("\n[DEBUG] Résultats bruts Chroma:")
    for d in docs:
        print(f"- Source: {d.metadata.get('source', '?')}")
        print(f"  Contenu: {d.page_content[:200]}...\n")

    lex_docs = db.similarity_search("exo", k=5, filter={"source": "lexique"})
    print("[DEBUG] Résultats Lexique:")
    for d in lex_docs:
        print("-", d.page_content[:120])

    return docs

# --- Chat ---
def chat(user_text: str) -> str:
    # Récup standard (wiki/dofusdude)
    docs_chroma = db.similarity_search(user_text, k=3)

    # Récup lexique si question "définition"
    docs_lexique = []
    if any(kw in user_text.lower() for kw in ["c'est quoi", "définition", "signifie", "qu'est-ce que"]):
        docs_lexique = db.similarity_search(user_text, k=5, filter={"source": "lexique"})

    # Debug
    print("\n[DEBUG] Résultats bruts Chroma:")
    for d in docs_chroma:
        print(f"- Source: {d.metadata.get('source','?')}\n  Contenu: {d.page_content[:120]}")

    print("\n[DEBUG] Résultats Lexique:")
    for d in docs_lexique:
        print(f"- {d.page_content[:120]}")

    # Fusionner les deux
    all_docs = docs_lexique + docs_chroma

    # Construire réponse en donnant directement les docs au LLM
    context = "\n\n".join(d.page_content for d in all_docs)
    prompt = f"Voici des documents:\n{context}\n\nQuestion: {user_text}\nRéponse:"
    result = llm.invoke(prompt)

    return getattr(result, "content", str(result))

# --- CLI ---
if __name__ == "__main__":
    print("💬 Chat local avec Ollama + RAG (tape 'quit' pour sortir).")
    while True:
        try:
            msg = input("\n[Toi] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Fermeture du chat.")
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break
        try:
            answer = chat(msg)
            print(f"\n[Bot] {answer}")
        except Exception as e:
            print(f"[Erreur] {e}")