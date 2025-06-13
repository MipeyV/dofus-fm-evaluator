# Dofus Forgemagie Evaluator

Projet de machine learning pour évaluer la qualité de jets d'items dans le jeu **Dofus**, à partir de captures d'écran et d'un modèle de classification supervisée.

## Objectif

- Extraire les statistiques des items à partir de captures d'écran via OCR.
- Évaluer automatiquement la qualité d’un jet (parfait, très bon, moyen, mauvais).
- Identifier les over/exo/stats tolérées.
- Entraîner ou tester un modèle de prédiction à partir de jeux de données synthétiques.

---

## Technologies

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- scikit-learn
- pandas
- opencv-python
- pytesseract
- numpy
- matplotlib (exploration)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-utilisateur/dofus-fm-evaluator.git
cd dofus-fm-evaluator
```

### 2. Créer un environnement virtuel

```
python -m venv .venv
```

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

```
tesseract --version
```
(Optionnel mais recommandé) Télécharger le fichier fra.traineddata pour utiliser la langue française :
https://github.com/tesseract-ocr/tessdata/blob/main/fra.traineddata
Placer-le dans :
C:\Program Files\Tesseract-OCR\tessdata\

## A faire / améliorer
Pour l'instant mon OCR ne permet pas encore de lire toutes les informations de mes screens de tests, je dois réduire le bruit du signal et pouvoir parser correctement les valeurs.

## Auteur
MipeyV
