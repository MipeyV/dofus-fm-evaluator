# scripts/app_gradio_2.py
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass

import json, csv, time, tempfile, sys
from pathlib import Path
from typing import Dict
import gradio as gr
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.predictor import score_image
from scripts.tools import _analyze_item_impl
from scripts.cli.cli_chat_with_agent import chat  # on réutilise ton agent

DEFAULT_MODEL = str(ROOT / "models" / "random_forest_model.joblib")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_CSV = RESULTS_DIR / "corrections.csv"

EFFETS_ANCHOR = str(ROOT / "tests" / "assets" / "effets.png")
POIDS_ANCHOR = str(ROOT / "tests" / "assets" / "poids.png")

APP_CSS = """
.gradio-container {max-width: 1100px !important}
"""

# ========== Fonctions de prédiction ==========
def make_score_html(pred: float | None) -> str:
    if pred is None:
        return "—"
    return f"### Score prédiction : **{pred:.2f}**"

def auto_crop_item(image_path: str) -> str:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return image_path
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        effets_anchor = cv2.imread(EFFETS_ANCHOR, 0)
        poids_anchor = cv2.imread(POIDS_ANCHOR, 0)
        if effets_anchor is None or poids_anchor is None:
            return image_path

        # Etape 1 : sécurité dimensions
        if (gray.shape[0] < effets_anchor.shape[0] or gray.shape[1] < effets_anchor.shape[1] or
            gray.shape[0] < poids_anchor.shape[0] or gray.shape[1] < poids_anchor.shape[1]):
            return image_path

        # --- Matching global (coarse)
        effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
        poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)
        _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
        _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

        if effets_val < 0.45 or poids_val < 0.45:
            return image_path

        # Définir un gros rectangle autour de la zone stats
        y_start = effets_loc[1]
        y_end = poids_loc[1] + poids_anchor.shape[0]
        x_start = max(0, effets_loc[0] - 50)
        x_end = min(gray.shape[1], poids_loc[0] + poids_anchor.shape[1] + 50)

        coarse_crop = image[y_start:y_end, x_start:x_end].copy()
        if coarse_crop.size == 0:
            return image_path

        # Etape 2 : relancer un matching sur le crop pour raffiner
        gray_crop = cv2.cvtColor(coarse_crop, cv2.COLOR_BGR2GRAY)
        try:
            effets_res2 = cv2.matchTemplate(gray_crop, effets_anchor, cv2.TM_CCOEFF_NORMED)
            poids_res2 = cv2.matchTemplate(gray_crop, poids_anchor, cv2.TM_CCOEFF_NORMED)
            _, _, _, effets_loc2 = cv2.minMaxLoc(effets_res2)
            _, _, _, poids_loc2 = cv2.minMaxLoc(poids_res2)

            y_start2 = effets_loc2[1] + effets_anchor.shape[0]
            y_end2 = poids_loc2[1]
            x_start2 = effets_loc2[0] + 40
            x_end2 = poids_loc2[0] - 10

            fine_crop = coarse_crop[y_start2:y_end2, x_start2:x_end2].copy()
            if fine_crop.size > 0:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                cv2.imwrite(tmp.name, fine_crop)
                return tmp.name
        except Exception as e:
            print(f"[WARN] étape 2 crop fin échouée: {e}")

        # fallback : garder le coarse crop
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp.name, coarse_crop)
        return tmp.name

    except Exception as e:
        print(f"[WARN] auto_crop_item a échoué: {e}")
        return image_path

def predict_png(img_path, model_path):
    if img_path is None:
        return make_score_html(None), None, None, None, None, None, None
    cropped_path = auto_crop_item(img_path)
    model = model_path or DEFAULT_MODEL
    try:
        res = score_image(str(model), cropped_path)  # <<< ici on passe le CROP
        pred = res.get("prediction", None)
        score_html = make_score_html(float(pred) if pred is not None else None)
        pretty = json.dumps(res, ensure_ascii=False, indent=2)
        features = res.get("features", {})
        rows = [["Feature", "Valeur"]] + [[k, v] for k, v in features.items()]
        ocr_lines = res.get("lines", [])
        return (
            score_html,
            "\n".join(f"- {l}" for l in ocr_lines),
            rows,
            pretty,
            float(pred) if pred is not None else None,
            str(img_path),
            str(cropped_path),
        )
    except Exception as e:
        return (
            f"<div style='background:#fecaca;color:#7f1d1d;border:2px solid #ef4444;"
            f"border-radius:14px;padding:18px;font-weight:800;text-align:center;'>❌ Erreur : {e}</div>",
            None, None, None, None, str(img_path), str(cropped_path)
        )

def analyze_png(img_path):
    if not img_path:
        return "⚠️ Aucune image", None, None
    try:
        res = _analyze_item_impl(img_path)
        if "error" in res:
            return f"❌ {res['error']}", None, None
        header = f"### {res['item']} (niv {res['level']})\n"
        eval_md = (
            f"- Score brut : {res['evaluation']['score']:.2f}\n"
            f"- Poids actuel : {res['evaluation']['total_weight']:.1f}\n"
            f"- Exo : {'✅' if res['evaluation'].get('exo') else '❌'}\n"
            f"- Over : {'✅' if res['evaluation'].get('over') else '❌'}\n"
            f"- Puits : {res['evaluation']['pui_category']}\n"
            f"- Qualité : **{res['evaluation']['quality']}**\n"
        )
        stats_table = [["Stat", "Valeur"]] + [[k,v] for k,v in res["stats_detected"].items()]
        return header + "\n" + eval_md + "\n\n" + res['commentaire'], stats_table, json.dumps(res, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ Erreur analyse : {e}", None, None

# Chatbot
def chatbot_fn(message, history, image=None):
    if image:
        message = f"{message} : {image}"
    response = chat(message, session_id="gradio")
    history.append((message, response))
    return "", history

# ========== Interface Gradio ==========
with gr.Blocks(title="Dofus FM Evaluator", css=APP_CSS, theme="default") as demo:
    gr.Markdown("## 🧪 Dofus FM Evaluator")

    state_model_score = gr.State(value=None)
    state_image_path = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem("Solo"):
            with gr.Row():
                image = gr.Image(type="filepath", label="Image", height=360)
                cropped_preview = gr.Image(type="filepath", label="Recadré", height=320)
            model_in = gr.Textbox(value=DEFAULT_MODEL, label="Modèle")
            go = gr.Button("Scorer")
            score_out = gr.Markdown()
            ocr_md = gr.Markdown()
            table_out = gr.Dataframe(headers=["Feature","Valeur"])
            json_out = gr.Code(language="json")
            go.click(
                predict_png,
                inputs=[image, model_in],
                outputs=[score_out, ocr_md, table_out, json_out, state_model_score, state_image_path, cropped_preview],
            )

        with gr.TabItem("Analyse complète"):
            image_a = gr.Image(type="filepath", label="Image", height=360)
            go_a = gr.Button("Analyser")
            result_md = gr.Markdown()
            stats_out = gr.Dataframe(headers=["Stat","Valeur"])
            json_out_a = gr.Code(language="json")
            go_a.click(analyze_png, inputs=[image_a], outputs=[result_md, stats_out, json_out_a])

        with gr.TabItem("Chatbot"):
            chatbot = gr.Chatbot(label="Assistant Dofus")
            msg = gr.Textbox(label="Message")
            img_in = gr.Image(type="filepath", label="Optionnel: Image")
            send = gr.Button("Envoyer")
            send.click(chatbot_fn, inputs=[msg, chatbot, img_in], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
