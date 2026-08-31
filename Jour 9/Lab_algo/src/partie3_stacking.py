# Partie 3 : Métamodèles — Voting & Stacking (35 min)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
import time  # Ajouter pour mesurer le temps
from data import get_data

warnings.filterwarnings("ignore", category=FutureWarning)

def get_base_models():
    # Retourne la liste des modèles de base hétérogènes utilisés partout
    return [
        ("logreg", LogisticRegression(max_iter=1000)),
        ("svm", SVC(probability=True)),  # probability=True nécessaire pour le soft voting
        ("rf", RandomForestClassifier(random_state=42)),
        ("xgb", XGBClassifier(eval_metric="logloss")),
    ]

def partie_3_voting(X_train, X_test, y_train, y_test):
    # Hard Voting vs Soft Voting
    start = time.time()
    hard_voting = VotingClassifier(estimators=get_base_models(), voting="hard")
    hard_voting.fit(X_train, y_train)
    acc_hard = accuracy_score(y_test, hard_voting.predict(X_test))
    hard_time = time.time() - start
    print(f"Hard Voting — Accuracy test : {acc_hard:.4f}")

    start = time.time()
    soft_voting = VotingClassifier(estimators=get_base_models(), voting="soft")
    soft_voting.fit(X_train, y_train)
    acc_soft = accuracy_score(y_test, soft_voting.predict(X_test))
    soft_time = time.time() - start
    print(f"Soft Voting — Accuracy test : {acc_soft:.4f}")

    # Retourner un dictionnaire comme attendu par main.py
    return {
        "hard": {
            "accuracy_test": acc_hard,
            "training_time_s": hard_time
        },
        "soft": {
            "accuracy_test": acc_soft,
            "training_time_s": soft_time
        }
    }


def partie_3_stacking(X_train, X_test, y_train, y_test):
    # Stacking avec méta-modèle = régression logistique
    start = time.time()
    stacking = StackingClassifier(
        estimators=get_base_models(),
        final_estimator=LogisticRegression(),
        cv=3,  # au lieu de 5
    )
    stacking.fit(X_train, y_train)
    acc_test = accuracy_score(y_test, stacking.predict(X_test))
    elapsed = time.time() - start
    print(f"Stacking — Accuracy test : {acc_test:.4f}")

    # Retourner un dictionnaire comme attendu par main.py
    return {
        "accuracy_test": acc_test,
        "training_time_s": elapsed
    }

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    voting = partie_3_voting(X_train, X_test, y_train, y_test)
    stacking = partie_3_stacking(X_train, X_test, y_train, y_test)
    
    print("\nRésultats Partie 3:")
    print(f"Hard Voting: {voting['hard']['accuracy_test']:.4f}")
    print(f"Soft Voting: {voting['soft']['accuracy_test']:.4f}")
    print(f"Stacking: {stacking['accuracy_test']:.4f}")