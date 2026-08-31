# Main — Exécution complète du Lab Algorithmes d'Ensemble
from data import get_data
from partie1_bagging import (
    partie_1_1_baseline,
    partie_1_2_bagging_manuel,
    partie_1_3_random_forest,
)
from partie2_boosting import partie_2_1_adaboost, partie_2_2_xgboost
from partie3_stacking import partie_3_voting, partie_3_stacking

def main():
    
    print("\n--- LAB — ALGORITHMES D'ENSEMBLE")


    print("\n")
    print("\n--- PARTIE 0 — CONFIGURATION ET DONNÉES")

    X_train, X_test, y_train, y_test = get_data()

    print(f"Nombre total d'échantillons : " f"{len(X_train) + len(X_test)}")
    print(f"Nombre de features sont: " f"{X_train.shape[1]}")
    print(f"Taille X_train est: " f"{X_train.shape}")
    print(f"Taille X_test est : " f"{X_test.shape}")
    print(f"Taille y_train est : " f"{y_train.shape}")
    print(f"Taille y_test est: " f"{y_test.shape}")

    print("\n")
    print("\n--- PARTIE 1 — BAGGING & RANDOM FOREST")

    baseline = partie_1_1_baseline(X_train, X_test, y_train, y_test)
    bagging = partie_1_2_bagging_manuel(X_train, X_test, y_train, y_test)
    random_forest = partie_1_3_random_forest(X_train, X_test, y_train, y_test)

    print("\n")
    print("\n--- PARTIE 2 — BOOSTING")

    adaboost = partie_2_1_adaboost(X_train, X_test, y_train, y_test)
    xgboost = partie_2_2_xgboost(X_train, X_test, y_train, y_test)

    print("\n")
    print("\n--- PARTIE 3 — VOTING & STACKING")

    voting = partie_3_voting(X_train, X_test, y_train, y_test)
    stacking = partie_3_stacking(X_train, X_test, y_train, y_test)

    print("\n")
    print("\n--- PARTIE 4 — BILAN DES PERFORMANCES")

    # Meilleur AdaBoost
    best_lr = max(adaboost, key=lambda lr: adaboost[lr]["accuracy_test"])
    best_adaboost = adaboost[best_lr]

    print(f"\n{'Modèle':<25}" f"{'Accuracy Test':<18}" f"{'Temps (s)':<15}")
    print(
        f"{'Arbre seul':<25}"
        f"{baseline['accuracy_test']:<18.4f}"
        f"{baseline['training_time_s']:<15.4f}"
    )
    print(
        f"{'Bagging manuel':<25}"
        f"{bagging['accuracy_test']:<18.4f}"
        f"{bagging['training_time_s']:<15.4f}"
    )
    print(
        f"{'Random Forest':<25}"
        f"{random_forest['best_accuracy']:<18.4f}"
        f"{random_forest['training_time_s']:<15.4f}"
    )
    print(
        f"{'AdaBoost':<25}"
        f"{best_adaboost['accuracy_test']:<18.4f}"
        f"{best_adaboost['training_time_s']:<15.4f}"
    )
    print(
        f"{'XGBoost':<25}"
        f"{xgboost['accuracy_test']:<18.4f}"
        f"{xgboost['training_time_s']:<15.4f}"
    )
    print(
        f"{'Stacking':<25}"
        f"{stacking['accuracy_test']:<18.4f}"
        f"{stacking['training_time_s']:<15.4f}"
    )
    print("\n")
    print("\n--- VOTING")
    print(f"Hard Voting est : " f"{voting['hard']['accuracy_test']:.4f}")
    print(f"Soft Voting est : " f"{voting['soft']['accuracy_test']:.4f}")
    print("\n---")
    print("\n--- FIN DU LAB")


if __name__ == "__main__":
    main()
