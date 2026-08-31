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

| Modèle | Accuracy (Test) | Temps d'entraînement (s) | Observations / Surapprentissage |
|--------|:---------------:|:------------------------:|----------------------------------|
| **Arbre de décision seul** | 0.8400 | 0.0360 | Fort surapprentissage : Train = 1.0000 contre Test = 0.8400. L'arbre apprend parfaitement les données d'entraînement mais généralise mal. |
| **Bagging manuel** | 0.9140 | 1.6138 | Réduction significative de la variance grâce au vote de 50 arbres. Amélioration de +0.074 par rapport à l'arbre seul. |
| **Random Forest** | **0.9460** | 0.2985 | `max_features=sqrt` et `log2` donnent les meilleurs résultats (0.9460). `max_features=None` (0.9160) est moins performant car les arbres sont trop corrélés. |
| **AdaBoost** | 0.8260 | 0.9743 | Meilleur résultat avec `learning_rate=1.0`. Performance inférieure aux autres méthodes d'ensemble sur ce jeu de données. |
| **XGBoost** | **0.9540** | 0.9956 | Très bonne performance. Surapprentissage observé vers 25-30 itérations (augmentation du logloss sur l'ensemble de validation). |
| **Hard Voting** | 0.9540 | — | Vote majoritaire sur les classes prédites par les 4 modèles. Performance équivalente à XGBoost seul. |
| **Soft Voting** | **0.9680** | — | Meilleur score global. Moyenne des probabilités prédites par les 4 modèles. Légèrement supérieur au Stacking. |
| **Stacking** | **0.9620** | 4.3601 | Meilleure performance globale. Combine 4 modèles (Logistic Regression, SVM, Random Forest, XGBoost) avec un méta-modèle (Régression Logistique). |


## Question de réflexion (Partie 1.3)
> Pourquoi la limitation des variables à chaque nœud (max_features) améliore-t-elle la diversité des arbres par rapport à un Bagging classique ?
 ## Dans le Bagging, tous les arbres utilisent les mêmes variables, ce qui peut les rendre similaires. Avec Random Forest, max_features sélectionne aléatoirement certaines variables à chaque nœud, ce qui diversifie les arbres, réduit la corrélation de leurs erreurs et améliore le modèle final.
