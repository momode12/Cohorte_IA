# Lab — Algorithmes d'Ensemble

Bagging, Boosting, Stacking et Voting appliqués à un problème de classification binaire synthétique.

# Cloner le projet depuis mon GitHub
git clone <URL_DU_REPOSITORY>

# Se déplacer dans le dossier du projet
cd Lab_algo

## Setup (avec uv)

```bash
uv sync
```
## Structure des fichiers générés par uv
Lab_algo/
├── .venv/                # Environnement virtuel (géré par uv)
├── pyproject.toml        # Fichier de configuration principal
├── uv.lock               # Fichier de verrouillage des dépendances
├── requirements.txt      # Dépendances au format classique
├── src/                  # Code source du projet
│   ├── data.py
│   ├── partie1_bagging.py
│   ├── partie2_boosting.py
│   ├── partie3_stacking.py
│   └── utils.py
└── README.md


## Installer les bibliothèques
uv pip install -r requirements.txt

Puis pour lancer un script :

# uv run src/main.py



## Partie 4 — Bilan des performances

| Modèle | Précision | Temps | À retenir |
|--------|-----------|-------|-----------|
| Arbre seul | 0.8400 | 0.04s |  Apprend par cœur, rate sur du nouveau |
| Bagging manuel | 0.9140 | 1.61s |  Corrige l'arbre seul |
| Random Forest | 0.9460 | 0.30s |  Meilleur que Bagging |
| AdaBoost | 0.8260 | 0.97s |  Le moins bon |
| XGBoost | 0.9540 | 1.00s |  Excellent, mais risque de surapprentissage |
| Hard Voting | 0.9540 | 1.05s |  Vote des modèles |
| Soft Voting | 0.9680 | 1.02s |  Moyenne des probabilités, le meilleur |
| Stacking | 0.9620 | 4.36s |  Très bon, mais lent |

## Question de réflexion (Partie 1.3)
> Pourquoi la limitation des variables à chaque nœud (max_features) améliore-t-elle la diversité des arbres par rapport à un Bagging classique ?
 ## Dans le Bagging, tous les arbres utilisent les mêmes variables, ce qui peut les rendre similaires. Avec Random Forest, max_features sélectionne aléatoirement certaines variables à chaque nœud, ce qui diversifie les arbres, réduit la corrélation de leurs erreurs et améliore le modèle final.
