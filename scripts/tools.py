# scripts/tools.py
from typing import Optional, Dict
from pathlib import Path
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from transformers import LevitModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

from src.models.predictor import score_image
from src.core.item_models import ItemTemplate, ItemInstance, stat_pool
from src.core.utils import document_to_itemtemplate, build_item_from_ocr

from scripts.rag import db, rag_chain

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "random_forest_model.joblib"

# -------- Tool 0 : score brut depuis image --------
class ScoreItemArgs(BaseModel):
    image_path: str = Field(..., description="Chemin vers l'image de l'item Dofus (PNG/JPG)")
    model_path: Optional[str] = Field(
        default=str(DEFAULT_MODEL),
        description="Chemin vers le modèle .joblib (optionnel)"
    )

def _score_item_impl(image_path: str, model_path: Optional[str] = None) -> str:
    try:
        img = Path(image_path)
        if not img.exists():
            return f"Erreur : le fichier '{img}' est introuvable."

        model = Path(model_path) if model_path else DEFAULT_MODEL
        if not model.exists():
            return f"Erreur : le modèle '{model}' est introuvable."

        res = score_image(str(model), str(img))  # -> dict
        pred = res.get("prediction", None)
        try:
            pred_txt = f"{float(pred):.2f}" if pred is not None else "N/A"
        except Exception:
            pred_txt = str(pred)

        features = res.get("features", {})
        return f"Score: {pred_txt} | Features: {features}"
    except Exception as e:
        return f"Erreur lors de l'évaluation de l'item : {e}"

score_item = StructuredTool.from_function(
    name="score_item",
    description=(
        "Évalue la qualité d'un item Dofus à partir d'une image. "
        "Retourne un score et les caractéristiques extraites. "
        "Utiliser une seule fois par requête. "
        "Après avoir reçu l'Observation, fournir directement la réponse finale."
    ),
    args_schema=ScoreItemArgs,
    func=_score_item_impl,
)

# -------- Tool 1 : récupérer stats théoriques --------
class GetItemStatsArgs(BaseModel):
    name: str = Field(..., description="Nom de l'item Dofus")

def _get_item_stats_impl(name: str) -> Dict:
    docs = db.similarity_search(name, k=1)
    if not docs:
        return {"error": f"Item '{name}' introuvable."}
    template = document_to_itemtemplate(docs[0])
    return {s: {"min": sd.min_value, "max": sd.max_value, "weight": sd.weight}
            for s, sd in template.stats.items()}

get_item_stats = StructuredTool.from_function(
    name="get_item_stats",
    description="Retourne les bornes min/max et valeurs des stats pour un item donné.",
    args_schema=GetItemStatsArgs,
    func=_get_item_stats_impl,
)

# -------- Tool 2 : calcul du poids --------
class GetItemWeightArgs(BaseModel):
    name: str = Field(..., description="Nom de l'item Dofus")
    current_stats: Optional[Dict[str, int]] = Field(
        default=None,
        description="Stats actuelles de l'item si connues, sinon poids max théorique."
    )

def _get_item_weight_impl(name: str, current_stats: Optional[Dict[str, int]] = None) -> float:
    docs = db.similarity_search(name, k=1)
    if not docs:
        return -1
    template = document_to_itemtemplate(docs[0])
    if current_stats:
        item = ItemInstance(template, current_stats)
        return item.get_total_weight()
    return sum(sd.max_value * sd.weight for sd in template.stats.values())

get_item_weight = StructuredTool.from_function(
    name="get_item_weight",
    description="Calcule le poids total d'un item (théorique max ou avec un jet donné).",
    args_schema=GetItemWeightArgs,
    func=_get_item_weight_impl,
)

# -------- Tool 3 : analyse complète (OCR + FM) --------
class AnalyzeItemArgs(BaseModel):
    image_path: str = Field(..., description="Chemin vers l'image de l'item Dofus (PNG/JPG)")

def _analyze_item_impl(image_path: str) -> Dict:
    img = Path(image_path)
    if not img.exists():
        return {"error": f"Fichier '{img}' introuvable."}

    # OCR + prédiction
    res = score_image(str(DEFAULT_MODEL), str(img))  # dict
    pred = res.get("prediction", None)
    current_stats = res.get("current_stats", {})
    item_name = res.get("item_name", None)

    if not item_name:
        return {"error": "Impossible de détecter le nom de l'item dans l'image."}

    # --- Recherche FAISS
    docs = db.similarity_search(item_name, k=1)

    if docs:  
        template = document_to_itemtemplate(docs[0])
        item = ItemInstance(template, current_stats)
        level_raw = docs[0].metadata.get("level")
    else:  
        item = build_item_from_ocr(item_name, current_stats)
        template = item.template
        level_raw = None

    # --- Poids & qualité
    total_weight = item.get_total_weight()
    exo_weight = item.get_exo_weight()
    quality = item.evaluate_quality_algo()

    try:
        level = int(level_raw) if level_raw else "?"
    except (ValueError, TypeError):
        level = "?"

    commentaire = []
    commentaire.append(f"➡️ Score brut: {pred}")
    commentaire.append(f"➡️ Poids théorique de base: {sum(sd.max_value * sd.weight for sd in template.stats.values()):.1f}")
    commentaire.append(f"➡️ Poids actuel: {total_weight:.1f}")
    commentaire.append("➡️ Exo détecté" if quality["exo"] else "➡️ Pas d'exo")
    commentaire.append("➡️ Over détecté" if quality["over"] else "➡️ Pas d'over")
    commentaire.append(f"➡️ Qualité globale du jet: {quality['quality']}")

    return {
        "item": template.name,
        "level": level,
        "stats_detected": current_stats,
        "evaluation": {
            "score": pred,
            "total_weight": total_weight,
            "exo_weight": exo_weight,
            "exo" : quality["exo"],
            "over": quality["over"],
            "pui_category": quality["puit"],
            "quality": quality["quality"]
        },
        "commentaire": "\n".join(commentaire),
    }

analyze_item = StructuredTool.from_function(
    name="analyze_item",
    description=(
        "Analyse complète d'un item Dofus à partir d'une image : "
        "détecte ses stats (OCR), calcule le poids, identifie exos/overs, "
        "et commente la qualité du jet."
    ),
    args_schema=AnalyzeItemArgs,
    func=_analyze_item_impl,
)

class RAGSearchArgs(BaseModel):
    query: str = Field(..., description="Question à poser sur le jeu Dofus.")

def _rag_search_impl(query: str) -> str:
    result = rag_chain.invoke(query)
    return result["result"]

rag_search = StructuredTool.from_function(
    name="rag_search",
    description="Recherche une définition ou explication dans le corpus Dofus (lexique, wiki, dofusdu.de).",
    args_schema=RAGSearchArgs,
    func=_rag_search_impl,
)