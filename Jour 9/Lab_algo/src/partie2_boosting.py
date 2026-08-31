# Partie 2 : Boosting
import time
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from data import get_data


def partie_2_1_adaboost(X_train, X_test, y_train, y_test):
    # 2.1 — AdaBoost avec des stumps
    print("\n--- 2.1 AdaBoost ---")
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    results = {}

    for lr in learning_rates:

        model = AdaBoostClassifier(
            estimator=stump, n_estimators=100, learning_rate=lr, random_state=42
        )

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        acc_test = accuracy_score(y_test, model.predict(X_test))
        results[lr] = {"accuracy_test": acc_test, "training_time_s": elapsed}
        
        print(
            f"learning_rate={lr} "
            f"— Accuracy : {acc_test:.4f} "
            f"— Temps : {elapsed:.4f}s"
        )

    return results


def partie_2_2_xgboost(X_train, X_test, y_train, y_test):
    # 2.2 — XGBoost avec suivi de la Log-Loss

    print("\n--- 2.2 XGBoost ---")
    model = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)
    start = time.time()
    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False
    )
    elapsed = time.time() - start
    acc_test = accuracy_score(y_test, model.predict(X_test))
    
    print(f"Accuracy test est: {acc_test:.4f}")
    print(f"Temps d'entraînement est: " f"{elapsed:.4f}s")

    # Résultats de la courbe d'apprentissage
    results = model.evals_result()
    train_logloss = results["validation_0"]["logloss"]
    test_logloss = results["validation_1"]["logloss"]
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_logloss, label="Train")
    plt.plot(test_logloss, label="Test")
    plt.xlabel("Nombre d'itérations des arbres")
    plt.ylabel("Log-Loss")
    plt.title("Courbe d'apprentissage de XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("xgboost_logloss.png", dpi=150)
    plt.close()

    print("Graphique sauvegardé : " "xgboost_logloss.png")

    return {"accuracy_test": acc_test, "training_time_s": elapsed}


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data()

    partie_2_1_adaboost(X_train, X_test, y_train, y_test)
    partie_2_2_xgboost(X_train, X_test, y_train, y_test)
