# scripts/cli_chat_with_router_rag.py
import re
import os
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from scripts.tools import score_item  # tool défini dans tools.py

# --- LLM Ollama ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Prompt système ---
SYSTEM_PROMPT = (
    "Tu es un expert du jeu Dofus.\n"
    "Tu aides les joueurs à :\n"
    "- Évaluer la qualité de leurs items (via tools)\n"
    "- Répondre à leurs questions générales (direct LLM)\n"
    "- Donner des explications précises sur la forgemagie et les items (via RAG)\n\n"
    "RÈGLES DE STYLE:\n"
    "- Ton professionnel, clair, courtois.\n"
    "- Phrases courtes, vocabulaire simple, pas d’anglicismes.\n"
    "- Orthographe et grammaire irréprochables.\n"
    "- Utilise des puces si utile.\n"
    "- Si tu ne sais pas, dis-le.\n"
    "LANGUE: Toujours répondre en français."
)

# --- Tools + Agent ---
tools = [score_item]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prefix": SYSTEM_PROMPT},
    verbose=True,
)

# --- Mémoire ---
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent,
    get_history,
    input_messages_key="input",
    history_message_key="chat_history",
)

# --- RAG (VectorDB) ---
print("[INFO] Chargement de la base vectorielle...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "data/vectorstore/dofusdude/faiss_index",  # <-- adapte ce chemin à ton projet
    embeddings,
    allow_dangerous_deserialization=True
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
)

# --- Utils ---
def extract_path(text: str) -> str | None:
    """Extrait un chemin d'image du texte utilisateur."""
    match = re.search(r"([\w\-/\\]+?\.(?:png|jpg|jpeg))", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# --- Router ---
def route_query(user_text: str) -> str:
    # Si on mentionne un fichier image => agent/tool
    if re.search(r"\.(png|jpg|jpeg)", user_text, re.IGNORECASE):
        return "agent"

    # Si vocabulaire Dofus/FM => rag
    keywords = ["exo", "over", "forgemagie", "puits", "jet", "fm", "item", "stats", "dofusbook", "wiki"]
    if any(kw.lower() in user_text.lower() for kw in keywords):
        return "rag"

    # Sinon simple LLM
    return "llm"

# --- Chat function ---
def chat(user_text: str, session_id: str = "default") -> str:
    route = route_query(user_text)
    print(f"[DEBUG] Route choisie: {route}")

    if route == "agent":
        # Extraction du chemin d'image
        path = extract_path(user_text)
        if path and os.path.exists(path):
            try:
                result = score_item.func(path)  # appel direct du tool
                return str(result)
            except Exception as e:
                return f"[Erreur tool] {e}"
        else:
            # fallback sur agent si pas de chemin valide
            result = agent_with_history.invoke(
                {"input": user_text},
                config={"configurable": {"session_id": session_id}},
            )
            return getattr(result, "content", str(result))

    elif route == "rag":
        try:
            result = rag_chain.invoke(user_text)
            return result["result"]
        except Exception as e:
            return f"[Erreur RAG] {e}"

    else:  # LLM direct
        try:
            result = llm.invoke(user_text)
            return getattr(result, "content", str(result))
        except Exception as e:
            return f"[Erreur LLM] {e}"

# --- CLI ---
if __name__ == "__main__":
    print("Chat local avec Ollama + Router + RAG (tape 'quit' pour sortir).")
    sid = "demo"
    while True:
        try:
            msg = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break
        try:
            print(chat(msg, session_id=sid))
        except Exception as e:
            print(f"[Erreur] {e}")