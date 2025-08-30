# Dofus Forgemagie Evaluator

Projet de machine learning pour évaluer la qualité des jets d'items dans le jeu **Dofus**, à partir de captures d'écran et d'un modèle de classification supervisée.

## 🎯 Objectifs

* Extraire automatiquement les statistiques des items à partir de captures d'écran (OCR).
* Évaluer la qualité d’un jet : parfait, très bon, bon, jet craft, jet nul.
* Identifier les **exos** (caractéristiques supplémentaires) et les **overs** (valeurs supérieures aux bornes).
* Calculer les poids associés à chaque ligne de stats.
* Entraîner et tester un modèle de prédiction supervisée (RandomForest).
* Fournir une interface utilisateur et un **chatbot Dofus** pour interpréter les résultats et répondre aux questions.

---

## 🛠️ Technologies utilisées

* **Python 3.11**
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* **scikit-learn** (modèle de classification RandomForest)
* **pandas**, **numpy** (manipulation de données)
* **opencv-python**, **pytesseract** (vision & OCR)
* **matplotlib** (exploration et visualisation)
* **FastAPI / Starlette / Uvicorn** (API backend)
* **Gradio** (interfaces web interactives)
* **LangChain + Chroma + FAISS** (RAG et chatbot spécialisé Dofus)

---

## 🗂️ Structure du projet

dofus-fm-evaluator/

```
├── src/
│   ├── ocr/
│   │   └── reader.py              # OCR & détection des lignes de stats
│   │   └── reader_enhanced.py     # OCR avancé avec exo/over
│   ├── core/
│   │   └── item_models.py         # Classes ItemTemplate / ItemInstance
│   │   └── utils.py               # Fonctions utilitaires (FAISS → ItemTemplate)
│   ├── models/
│   │   └── predictor.py           # Entraînement & prédiction (RandomForest)
├── data/
│   ├── raw/                       # Captures d'écran brutes
│   └── processed/                 # Jeux de données CSV générés
├── models/
│   └── random_forest_model.joblib # Modèle entraîné sauvegardé
├── scripts/
│   ├── app_gradio.py              # Interface Gradio OCR + Scoring
│   ├── app_gradio_2.py            # Interface Gradio OCR + Analyse complète + Chatbot
│   ├── cli/
│   │   └── cli_chat_with_agent.py # Agent LLM + Tools + RAG en CLI
│   │   └── cli_chat_with_rag.py   # Recherche sémantique dans la base Dofus
│   └── tools.py                   # Définition des Tools LangChain (OCR, analyse...)
├── tests/
│   └── test_reader.py             # Tests unitaires OCR
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
└── .venv/                         # (Optionnel) Environnement virtuel
```

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-utilisateur/dofus-fm-evaluator.git
cd dofus-fm-evaluator
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Activation :

```bash
# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer et configurer **Tesseract OCR**

* Télécharger et installer Tesseract : [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Par défaut : `C:\Program Files\Tesseract-OCR`
* Ajouter le chemin à la variable d’environnement `PATH`
* Vérifier l’installation :

```bash
tesseract --version
```

* Télécharger le fichier `fra.traineddata` pour la langue française :
  [https://github.com/tesseract-ocr/tessdata/blob/main/fra.traineddata](https://github.com/tesseract-ocr/tessdata/blob/main/fra.traineddata)
* Le placer dans : `C:\Program Files\Tesseract-OCR\tessdata\`

---

## ▶️ Utilisation

### Interface simple (OCR + Score)

```bash
python -m scripts.app.app_gradio
```

Accessible sur : [http://127.0.0.1:7860](http://127.0.0.1:7860)

### Interface avancée (Analyse complète + Chatbot)

```bash
python -m scripts.app.app_gradio_2
```

Accessible sur : [http://127.0.0.1:7860](http://127.0.0.1:7860)

### CLI Agent (LLM + Tools + RAG)

```bash
python -m scripts.cli.cli_chat_with_agent
```

---

## ✅ Fonctionnalités principales

* OCR avancé avec détection :

  * **Exo** (caractéristiques supplémentaires, souvent en haut de l’item)
  * **Over** (valeurs au-dessus des bornes)
* Poids calculés automatiquement en fonction des stats × poids de référence.
* Algorithme d’évaluation qualitatif : `bon`, `très bon`, `parfait`, `jet craft`, `jet nul`.
* Chatbot contextuel avec recherche sémantique (RAG) :

  * Lexique Dofus
  * Wiki Dofus
  * Données de dofusdu.de
* Génération de datasets simulés pour l’entraînement du modèle.

---

## 🔧 Améliorations à venir

* Corriger certaines erreurs de calculs de poids (matching OCR ↔ stat\_pool).
* Améliorer la robustesse de la détection des exos et overs.
* Étendre la base de connaissances du chatbot (plus de documents Dofus).
* Optimiser l’interface Gradio pour une utilisation fluide.

---

## 👤 Auteur

**MipeyV**
Projet personnel Data & IA appliqué à **Dofus**.