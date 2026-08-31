# Partie 0 : Configuration et Données

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_data(random_state: int = 42):
    # Génère un jeu de données de classification binaire
    X, y = make_classification(
        n_samples=2500,
        n_features=20,
        n_informative=12,
        n_classes=2,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")