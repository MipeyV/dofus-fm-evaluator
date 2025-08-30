# scripts/cli_chat_with_agent.py
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from scripts.tools import score_item, get_item_stats, get_item_weight, analyze_item

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

# --- Tools ---
TOOLS = {
    "score_item": score_item,
    "get_item_stats": get_item_stats,
    "get_item_weight": get_item_weight,
    "analyze_item": analyze_item,
}

# --- Agent = LLM bindé aux tools ---
agent = llm.bind_tools(list(TOOLS.values()))

# --- Mémoire ---
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent,
    get_history,
    history_message_key="chat_history",
)

# --- Chat ---
def chat(user_text: str, session_id: str = "default") -> str:
    try:
        history = get_history(session_id)

        # 1. Ajout des messages système + humain
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]

        # 2. Premier appel agent
        result = agent_with_history.invoke(
            messages,
            config={"configurable": {"session_id": session_id}},
        )

        # 3. Si l'agent appelle un tool
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
                            content=f"[Erreur lors de l'exécution du tool {tool_name}] {e}",
                            name=tool_name,
                            tool_call_id=call["id"],
                        ))

            # 4. Réinjecter le résultat dans l’agent pour une réponse finale
            final_result = agent_with_history.invoke(
                [AIMessage(content="", tool_calls=result.tool_calls)] + outputs,
                config={"configurable": {"session_id": session_id}},
            )
            return getattr(final_result, "content", str(final_result))

        # 5. Si pas de tool utilisé → réponse directe
        return getattr(result, "content", str(result))

    except Exception as e:
        return f"[Erreur agent] {e}"

# --- CLI ---
if __name__ == "__main__":
    print("💬 Chat local avec Ollama + Agent + Tools (tape 'quit' pour sortir).")
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
