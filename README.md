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
├── src/
│   ├── ocr/
│   │   └── reader.py           # OCR & détection des lignes dans une image
│   └── model/
│       └── predictor.py        # Entraînement & prédiction avec RandomForest
├── data/
│   ├── raw/                    # Captures d'écran brutes
│   └── processed/              # Jeux de données CSV générés
├── models/
│   └── saved_model.joblib      # Modèle entraîné sauvegardé
├── tests/
│   └── test_reader.py          # Tests unitaires OCR
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
└── .venv/                      # (Optionnel) Environnement virtuel

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
### 3. Activer son environnement virtuel

```Activer son environnement virtuel
# Windows :
.\.venv\Scripts\activate
# Linux/macOS :
source .venv/bin/activate
```

### 3. Installer les dépendances de python
```
pip install -r requirements.txt
```
### 4. Installer et configurer Tesseract OCR

Ce projet utilise Tesseract pour extraire le texte des captures d’écran.

Étapes :
Télécharger Tesseract :
https://github.com/tesseract-ocr/tesseract

Installer Tesseract (par défaut dans C:\Program Files\Tesseract-OCR sur Windows)

Ajouter le chemin d’installation à ta variable d’environnement PATH :

Ouvrir les Variables d’environnement

Modifier la variable Path

Ajouter : C:\Program Files\Tesseract-OCR

Vérifier dans un terminal :

```bash
tesseract --version
```
(Optionnel mais recommandé) Télécharger le fichier fra.traineddata si tu veux utiliser la langue française :
https://github.com/tesseract-ocr/tessdata/blob/main/fra.traineddata
Place-le dans :
C:\Program Files\Tesseract-OCR\tessdata\

## A faire / améliorer
Pour l'instant mon OCR ne permet pas encore de lire toutes les informations de mes screens de tests, je dois réduire le bruit du signal et pouvoir parser correctement les valeurs.

## Auteur
MipeyV