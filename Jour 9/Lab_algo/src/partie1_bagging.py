# Partie 1 : Bagging & Random Forest

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import get_data


def partie_1_1_baseline(X_train, X_test, y_train, y_test):
    # 1.1 — Arbre de décision sans limite de profondeur.

    print("\n--- 1.1 Arbre de décision ---")
    model = DecisionTreeClassifier(random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    print(f"Accuracy train : {acc_train:.4f}")
    print(f"Accuracy test  : {acc_test:.4f}")
    print(f"Temps          : {elapsed:.4f}s")

    return {
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "training_time_s": elapsed,
    }


def partie_1_2_bagging_manuel(X_train, X_test, y_train, y_test, B=50):
    # 1.2 — Bagging implémenté manuellement

    print("\n--- 1.2 Bagging manuel ---")
    start = time.time()
    n = len(X_train)
    predictions = np.zeros((B, len(X_test)))

    for i in range(B):
        # Tirage bootstrap avec remise
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        tree = DecisionTreeClassifier(random_state=i)
        tree.fit(X_boot, y_boot)
        predictions[i] = tree.predict(X_test)

    # Vote majoritaire
    final_pred = (predictions.mean(axis=0) >= 0.5).astype(int)
    acc_test = accuracy_score(y_test, final_pred)
    elapsed = time.time() - start

    print(f"Nombre d'arbres est : {B}")
    print(f"Accuracy test est : {acc_test:.4f}")
    print(f"Temps est : {elapsed:.4f}s")

    return {
        "accuracy_test": acc_test,
        "training_time_s": elapsed,
    }


def partie_1_3_random_forest(X_train, X_test, y_train, y_test):
    # 1.3 — Random Forest avec plusieurs valeurs de max_features

    print("\n--- 1.3 Random Forest ---")
    max_features_options = ["sqrt", "log2", None]
    results = {}
    best_model = None
    best_accuracy = -1
    best_mf = None

    for mf in max_features_options:
        
        rf = RandomForestClassifier(
            n_estimators=100, max_features=mf, random_state=42, n_jobs=-1
        )

        start = time.time()
        rf.fit(X_train, y_train)
        elapsed = time.time() - start
        acc_test = accuracy_score(y_test, rf.predict(X_test))
        results[str(mf)] = {"accuracy_test": acc_test, "training_time_s": elapsed}

        print(
            f"max_features={mf} "
            f"— Accuracy : {acc_test:.4f} "
            f"— Temps : {elapsed:.4f}s"
        )

        if acc_test > best_accuracy:

            best_accuracy = acc_test
            best_model = rf
            best_mf = mf

    # Importance des variables
    importances = best_model.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.title(f"Importance des variables " f"(max_features={best_mf})")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.close()

    print("\n Graphique bien sauvegardé : " "feature_importances.png")

    return {
        "results": results,
        "best_accuracy": best_accuracy,
        "best_mf": best_mf,
        "training_time_s": results[str(best_mf)]["training_time_s"],
    }


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_data()

    partie_1_1_baseline(X_train, X_test, y_train, y_test)
    partie_1_2_bagging_manuel(X_train, X_test, y_train, y_test)
    partie_1_3_random_forest(X_train, X_test, y_train, y_test)
