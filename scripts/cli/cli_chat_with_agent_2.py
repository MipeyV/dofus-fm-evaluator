# scripts/cli_chat_with_agent_2.py
import os
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from scripts.rag import rag_chain

# Tools classiques
from scripts.tools import score_item, get_item_stats, get_item_weight, analyze_item, rag_search

# RAG (Chroma + lexique + wiki + dofusdude)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# --- LLM Ollama ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Base vectorielle ---
print("[INFO] Chargement de la base vectorielle (Chroma)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="rag_dofus", embedding_function=embeddings)

# --- Tools ---
TOOLS = {
    "score_item": score_item,
    "get_item_stats": get_item_stats,
    "get_item_weight": get_item_weight,
    "analyze_item": analyze_item,
    "rag_search": rag_search,
}

# --- Prompt système ---
SYSTEM_PROMPT = (
    "Tu es un expert du jeu Dofus.\n"
    "Tu aides les joueurs à :\n"
    "- Évaluer la qualité de leurs items (via tools)\n"
    "- Répondre à leurs questions générales (direct LLM)\n"
    "- Donner des explications précises par le RAG TOUJOURS à travers le tool rag_search "
    "pour les définitions ou quand l'utilisateur demande 'c'est quoi', 'définition', 'signifie', etc.\n\n"
    "STYLE:\n"
    "- Réponses courtes et claires\n"
    "- Français correct\n"
    "- Utilise des puces si utile\n"
    "- Si tu ne sais pas, dis-le\n"
)

# --- Mémoire ---
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent = llm.bind_tools(list(TOOLS.values()))
agent_with_history = RunnableWithMessageHistory(
    agent,
    get_history,
    history_message_key="chat_history",
)

# --- Fonction chat ---
def chat(user_text: str, session_id: str = "default") -> str:
    try:
        history = get_history(session_id)

        # ✅ Forçage RAG pour définitions
        if any(kw in user_text.lower() for kw in ["c'est quoi", "définition", "signifie"]):
            # Cherche uniquement dans le lexique
            docs = db.similarity_search(user_text, k=10, filter={"source": "lexique"})
            print("\n[DEBUG] Résultats Lexique:")
            for d in docs:
                print("-", d.page_content[:200], "...")
            if docs:
                return docs[0].page_content  # prend le meilleur chunk du lexique
            else:
                return "Désolé, aucune définition trouvée dans le lexique."


        # 🔄 Sinon → agent classique (avec tools OCR/FM)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
        result = agent_with_history.invoke(
            messages,
            config={"configurable": {"session_id": session_id}},
        )

        # 🔧 Si tool appelé
        if hasattr(result, "tool_calls") and result.tool_calls:
            outputs = []
            for call in result.tool_calls:
                tool_name = call["name"]
                args = call["args"]
                if tool_name in TOOLS:
                    try:
                        tool_result = TOOLS[tool_name].func(**args)
                        outputs.append(ToolMessage(
                            content=str(tool_result),
                            name=tool_name,
                            tool_call_id=call["id"],
                        ))
                    except Exception as e:
                        outputs.append(ToolMessage(
                            content=f"[Erreur tool {tool_name}] {e}",
                            name=tool_name,
                            tool_call_id=call["id"],
                        ))

            final_result = agent_with_history.invoke(
                [AIMessage(content="", tool_calls=result.tool_calls)] + outputs,
                config={"configurable": {"session_id": session_id}},
            )
            return getattr(final_result, "content", str(final_result))

        return getattr(result, "content", str(result))

    except Exception as e:
        return f"[Erreur agent] {e}"

# --- CLI ---
if __name__ == "__main__":
    print("💬 Chat local avec Ollama + Agent+Tools+RAG (tape 'quit' pour sortir).")
    while True:
        try:
            msg = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Fermeture du chat.")
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break
        print(chat(msg))