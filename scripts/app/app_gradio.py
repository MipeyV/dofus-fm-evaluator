# scripts/app_gradio.py
import os
os.environ["MPLBACKEND"] = "Agg"

import requests

import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass

try:
    import gradio.utils as gutils
    class _SafeMatplotlibBackendMananger:
        def __enter__(self):
            try:
                import matplotlib
                self._original_backend = matplotlib.get_backend()
                matplotlib.use("Agg")
            except Exception:
                self._original_backend = None
            return self
        def __exit__(self, exc_type, exc, tb):
            try:
                if getattr(self, "_original_backend", None):
                    import matplotlib
                    matplotlib.use(self._original_backend)
            except Exception:
                pass
    gutils.MatplotlibBackendMananger = _SafeMatplotlibBackendMananger
except Exception:
    pass

import json
import csv
import time
import tempfile
from pathlib import Path
import sys

import gradio as gr
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.models.predictor import score_image

DEFAULT_MODEL = str(ROOT / "models" / "random_forest_model.joblib")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CORRECTIONS_CSV = RESULTS_DIR / "corrections.csv"

EFFETS_ANCHOR = str(ROOT / "tests" / "assets" / "effets.png")
POIDS_ANCHOR = str(ROOT / "tests" / "assets" / "poids.png")

APP_CSS = """
.gradio-container {max-width: 1100px !important}
.hero{
  margin:16px 0;padding:20px;border-radius:14px;
  border:1px solid rgba(148,163,184,.25);
  background:linear-gradient(180deg, rgba(15,23,42,.85), rgba(15,23,42,.65));
  box-shadow:0 6px 18px rgba(0,0,0,.25);
}
.hero-title{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.5rem;letter-spacing:.2px;color:#e2e8f0;}
.hero-title .emoji{font-size:1.6rem;}
.hero-badges{margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;}
.pill{padding:8px 12px;border-radius:999px;background:#0ea5e9;color:#0b1020;font-weight:700;}
.dark .pill{background:#22d3ee;color:#082f49;}
.hero-sub{margin-top:10px;color:#cbd5e1;font-size:.95rem;opacity:.9;}
.hero-sub code{background:rgba(148,163,184,.2);color:#f1f5f9;padding:2px 6px;border-radius:6px;}
"""

def make_score_html(pred: float | None) -> str:
    if pred is None:
        return (
            '<div style="background:#334155;color:#e2e8f0;border:2px solid #94a3b8;'
            'border-radius:14px;padding:18px;font-size:1.2em;font-weight:800;'
            'text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.18)">—</div>'
        )
    return (
        '<div style="background:#0ea5e9;color:#0b1020;border:2px solid #38bdf8;'
        'border-radius:14px;padding:18px;font-size:1.35em;font-weight:800;'
        'text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.18)">'
        f'Score prédiction : {pred:.2f}'
        '</div>'
    )

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

        if (
            effets_anchor.shape[0] > gray.shape[0] or effets_anchor.shape[1] > gray.shape[1]
            or poids_anchor.shape[0] > gray.shape[0] or poids_anchor.shape[1] > gray.shape[1]
        ):
            return image_path

        effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
        poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)
        _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
        _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

        if effets_val < 0.45 or poids_val < 0.45:
            return image_path

        padding_left = 50
        roi_max_width = 330
        x_start = effets_loc[0] + padding_left
        x_end = min(x_start + roi_max_width, gray.shape[1] - 10)
        right_limit_by_anchor = poids_loc[0] - 10
        if right_limit_by_anchor > x_start + 40:
            x_end = min(x_end, right_limit_by_anchor)

        y_start = effets_loc[1] + effets_anchor.shape[0]
        y_end = poids_loc[1]

        if x_end <= x_start or y_end <= y_start:
            return image_path

        cropped = image[y_start:y_end, x_start:x_end].copy()
        if cropped.size == 0:
            return image_path

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp.name, cropped)
        return tmp.name
    except Exception:
        return image_path

def predict_png(img_path, model_path):
    if img_path is None:
        return make_score_html(None), None, None, None, None, None, None
    cropped_path = auto_crop_item(img_path)
    model = model_path or DEFAULT_MODEL
    try:
        res = score_image(model, img_path)
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
            f"<div style='background:#fecaca;color:#7f1d1d;border:2px solid #ef4444;border-radius:14px;padding:18px;font-weight:800;text-align:center;'>❌ Erreur : {e}</div>",
            None, None, None, None, str(img_path), str(cropped_path)
        )

def predict_batch(filepaths, model_path):
    if not filepaths:
        return "Aucun fichier fourni.", None, None, None
    model = model_path or DEFAULT_MODEL
    results = []
    table = [["Image", "Score"]]
    for p in filepaths:
        path = p if isinstance(p, str) else getattr(p, "name", None)
        if not path:
            continue
        try:
            cropped = auto_crop_item(path)
            res = score_image(model, cropped)
            score = float(res.get("prediction", 0.0))
            results.append({"image": Path(path).name, "cropped": Path(cropped).name, **res})
            table.append([Path(path).name, f"{score:.2f}"])
        except Exception as e:
            table.append([Path(path).name, f"ERREUR: {e}"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"predictions_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "prediction"])
        for row in table[1:]:
            w.writerow(row)
    summary = f"### ✅ Traitement terminé : **{len(table)-1}** images\n**CSV** sauvegardé : `{out_csv}`"
    json_dump = json.dumps(results, ensure_ascii=False, indent=2)
    return summary, json_dump, table, str(out_csv)

def save_correction(img_path: str, model_score: float, manual_score):
    if not img_path:
        return "⚠️ Pas d’image sélectionnée."
    if manual_score is None:
        return "⚠️ Saisis une valeur."
    try:
        need_header = not CORRECTIONS_CSV.exists()
        with CORRECTIONS_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["image", "score_modele", "score_humain", "timestamp"])
            w.writerow([Path(img_path).name, model_score, float(manual_score), time.strftime("%Y-%m-%d %H:%M:%S")])
        return f"✅ Correction sauvegardée pour `{Path(img_path).name}`"
    except Exception as e:
        return f"❌ Erreur : {e}"

with gr.Blocks(title="Dofus FM Evaluator", css=APP_CSS, theme="default") as demo:
    gr.HTML(
        """
        <div class="hero">
          <div class="hero-title">
            <span class="emoji">🧪</span>
            <span>Dofus FM Evaluator</span>
          </div>
          <div class="hero-badges">
            <span class="pill">OCR → Parsing → Features → ML</span>
            <span class="pill">Drag & Drop PNG</span>
          </div>
          <div class="hero-sub">Modèle par défaut : <code>models/random_forest_model.joblib</code></div>
        </div>
        """
    )

    state_model_score = gr.State(value=None)
    state_image_path = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem("Solo (une image)"):
            with gr.Row():
                with gr.Column(scale=5):
                    image = gr.Image(type="filepath", label="Image PNG/JPG (brut ou déjà rogné)", height=360)
                    cropped_preview = gr.Image(type="filepath", label="Item recadré (auto)", height=320)
                with gr.Column(scale=5):
                    model_in = gr.Textbox(value=DEFAULT_MODEL, label="Chemin du modèle (.joblib)", interactive=True)
                    go = gr.Button("Scorer l’image", variant="primary")
                    score_out = gr.HTML()
                    with gr.Accordion("Lignes OCR (diagnostic)", open=False):
                        ocr_md = gr.Markdown()
                    with gr.Accordion("Features extraites", open=True):
                        table_out = gr.Dataframe(headers=["Feature", "Valeur"], row_count=(0, "dynamic"))
                    with gr.Accordion("Résultat complet (JSON)", open=False):
                        json_out = gr.Code(language="json")
                    manual_in = gr.Number(label="Corriger le score manuellement", precision=2)
                    save_btn = gr.Button("Sauvegarder correction", variant="secondary")
                    save_msg = gr.Markdown()

            go.click(
                fn=predict_png,
                inputs=[image, model_in],
                outputs=[score_out, ocr_md, table_out, json_out, state_model_score, state_image_path, cropped_preview],
            )
            save_btn.click(fn=save_correction, inputs=[state_image_path, state_model_score, manual_in], outputs=[save_msg])

        with gr.TabItem("Batch (plusieurs images)"):
            with gr.Row():
                files = gr.File(label="Dépose plusieurs images (PNG/JPG)", type="filepath", file_count="multiple")
                with gr.Column():
                    model_in_b = gr.Textbox(value=DEFAULT_MODEL, label="Chemin du modèle (.joblib)", interactive=True)
                    go_b = gr.Button("Scorer le lot", variant="primary")

            summary_out = gr.Markdown()
            with gr.Accordion("Résultats (liste JSON)", open=False):
                json_batch_out = gr.Code(language="json")
            table_batch_out = gr.Dataframe(headers=["Image", "Score"], row_count=(0, "dynamic"))
            csv_path_out = gr.Textbox(label="CSV généré", interactive=False)
            download_btn = gr.DownloadButton(label="Télécharger le CSV", value=None)

            def _batch_and_prepare_download(filepaths, model_path):
                summary, json_text, table, csv_path = predict_batch(filepaths, model_path)
                return summary, json_text, table, csv_path, csv_path

            go_b.click(
                fn=_batch_and_prepare_download,
                inputs=[files, model_in_b],
                outputs=[summary_out, json_batch_out, table_batch_out, csv_path_out, download_btn],
            )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        share=False
    )