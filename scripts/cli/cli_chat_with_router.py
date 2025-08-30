import re
import os
from langchain_ollama import ChatOllama
from scripts.tools import score_item, get_item_stats, get_item_weight, analyze_item

# --- LLM Ollama ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Tools disponibles ---
TOOLS = {
    "score_item": score_item,
    "get_item_stats": get_item_stats,
    "get_item_weight": get_item_weight,
    "analyze_item": analyze_item,
}

# --- Router ---
def route_query(user_text: str) -> str:
    """
    Détermine vers quel tool router la requête, ou si on reste sur le LLM.
    """
    txt = user_text.lower()

    if re.search(r"\.(png|jpg|jpeg)", user_text, re.IGNORECASE):
        # Si c'est une image, on regarde si c'est analyse ou score
        if "analyser" in txt or "analyse" in txt or "commenter" in txt or "commente" in txt:
            return "analyze_item"
        return "score_item"
    if "stats" in txt or "caractéristique" in txt:
        return "get_item_stats"
    if "poids" in txt or "weight" in txt:
        return "get_item_weight"

    return "llm"

def extract_path(text: str) -> str | None:
    match = re.search(r"([\w\-/\\]+?\.(?:png|jpg|jpeg))", text, re.IGNORECASE)
    return match.group(1) if match else None

def chat(user_text: str) -> str:
    route = route_query(user_text)

    if route in TOOLS:
        try:
            if route in {"score_item", "analyze_item"}:
                path = extract_path(user_text)
                if not path or not os.path.exists(path):
                    return f"[Erreur] Impossible de trouver le fichier image dans : {user_text}"
                result = TOOLS[route].func(path)
            else:
                # Pour get_item_stats / get_item_weight -> on passe le nom brut
                item_name = user_text
                result = TOOLS[route].func(item_name)
            return str(result)
        except Exception as e:
            return f"[Erreur tool {route}] {e}"
    else:
        try:
            result = llm.invoke(user_text)
            return getattr(result, "content", str(result))
        except Exception as e:
            return f"[Erreur LLM] {e}"

# --- CLI ---
if __name__ == "__main__":
    print("💬 Chat local avec Ollama + Router (tape 'quit' pour sortir).")
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
        print(chat(msg))