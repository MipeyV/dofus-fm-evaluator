from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- LLM Ollama ---
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# --- Prompt système ---
SYSTEM_PROMPT = (
    "Tu es un expert du jeu Dofus, tu aides les joueurs à évaluer la qualité de leurs items en fonction de leurs caractéristiques.\n"
    "RÈGLES DE STYLE :\n"
    "- Ton professionnel, clair, courtois, pas d’emphase inutile.\n"
    "- Phrases courtes, vocabulaire simple, pas d’anglicismes.\n"
    "- Orthographe et grammaire irréprochables.\n"
    "- Structure la réponse avec des puces si utile.\n"
    "- Si tu n’as pas la réponse, dis-le franchement.\n"
    "DOMAINES DE COMPÉTENCE :\n"
    "- Tu analyses les statistiques d'un item et tu donnes une évaluation claire et concise de sa qualité.\n"
    "- Tu peux aussi répondre aux questions des joueurs sur les items, les stats, les termes techniques (over, exo, toléré, etc.) et donner des conseils pour améliorer leurs équipements.\n"
    "- Tu dois toujours répondre en français."
)

# --- Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm

# --- Mémoire par session ---
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_message_key="history",  # doit matcher MessagesPlaceholder("history")
)

# --- Chat API ---
def chat(user_text: str, session_id: str = "default") -> str:
    result = chain_with_history.invoke(
        {"input": user_text},
        config={"configurable": {"session_id": session_id}},
    )
    return result.content

# --- CLI ---
if __name__ == "__main__":
    print("Chat local Dofus (tape 'quit' pour sortir).")
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