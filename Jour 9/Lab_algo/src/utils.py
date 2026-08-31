# Utilitaires pour évaluer les modèles

import time

from sklearn.metrics import accuracy_score


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Entraîne un modèle et retourne les accuracy

    start = time.time()

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    return {
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "training_time_s": elapsed,
    }