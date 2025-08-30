# scripts/build/build_dofusdude.py
import os
import requests
import json
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ----------- Config ----------
LANG = "fr"
GAME = "dofus3"  # ou "dofus3beta" si tu veux la bêta
SAVE_PATH = "data/vectorstore/dofusdude/"

os.makedirs(SAVE_PATH, exist_ok=True)


def fetch_all_equipment():
    """Récupère tous les équipements Dofus via API JSON brut"""
    url = f"https://api.dofusdu.de/{GAME}/v1/{LANG}/items/equipment/all"
    print(f"→ Requête directe: {url}")

    r = requests.get(url, headers={"Accept": "application/json"})
    r.raise_for_status()
    data = r.json()

    clean_items = []
    for item in data.get("items", []):
        # Supprime les conditions qui font bugger le SDK
        item.pop("conditions", None)
        clean_items.append(item)

    return clean_items


def build_docs(items):
    """Transforme les dicts JSON en textes pour embeddings"""
    docs = []
    for it in items:
        name = it.get("name", "Inconnu")
        level = it.get("level", "?")
        type_name = it.get("type", {}).get("name", "N/A")
        description = it.get("description", "")
        effects = it.get("effects", [])

        text = f"Nom: {name} | Type: {type_name} | Niveau: {level}"
        if description:
            text += f" | Description: {description}"
        if effects:
            effs = ", ".join([e.get("effect", "") for e in effects])
            text += f" | Effets: {effs}"

        # --- Enrichissement pour ItemTemplate ---
        stats_dict = {
            e.get("effect", ""): {
                "min": e.get("from"),
                "max": e.get("to")
            }
            for e in effects if e.get("effect")
        }

        # ⚠️ Ici on convertit stats_dict en string JSON
        docs.append(Document(
            page_content=text,
            metadata={
                "id": str(it.get("ankama_id")),
                "name": name,
                "level": str(level),
                "type": type_name,
                "stats": json.dumps(stats_dict, ensure_ascii=False)  # 👈 fix
            }
        ))
    return docs

if __name__ == "__main__":
    print("[Dofus 3] Récupération des équipements...")
    items = fetch_all_equipment()
    print(f"✅ {len(items)} équipements récupérés")

    print("[Indexation] Transformation en documents...")
    docs = build_docs(items)

    print("[Indexation] Génération des embeddings HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(docs, embeddings)

    # Sauvegarde
    faiss_path = os.path.join(SAVE_PATH, "faiss_index")
    vectordb.save_local(faiss_path)

    print(f"🎉 Base vectorielle sauvegardée dans {faiss_path}")
