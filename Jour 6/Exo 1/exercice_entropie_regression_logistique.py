import math
import matplotlib.pyplot as plt

# 1. entropie(proportions)
def entropie(proportions):
    h = 0.0
    for p in proportions:
        if p > 0:
            h -= p * math.log2(p)
    return h

# 2. gain_information(entropie_parent, sous_ensembles)
def gain_information(entropie_parent, sous_ensembles):
    taille_totale = sum(taille for taille, _ in sous_ensembles)
    entropie_ponderee = 0.0
    for taille, h_sv in sous_ensembles:
        entropie_ponderee += (taille / taille_totale) * h_sv
    return entropie_parent - entropie_ponderee

# 3. sigmoide(z)
def sigmoide(z):
    return 1 / (1 + math.exp(-z))

# 4. log_loss(y_true, y_pred_proba)
def log_loss(y_true, y_pred_proba):
    return -(y_true * math.log(y_pred_proba) + (1 - y_true) * math.log(1 - y_pred_proba))

# Test
if __name__ == "__main__":

    # Entropie
    print(f"Entropie pièce équilibrée : {entropie([0.5, 0.5])}")
    print(f"Entropie pièce biaisée 0.8/0.2 : {entropie([0.8, 0.2])}")

    # Gain d'information 
    entropie_parent = 1.0
    h_soleil = entropie([1 / 4, 3 / 4])     
    h_couvert = entropie([2 / 2, 0 / 2])    
    h_pluie = entropie([2 / 4, 2 / 4])     

    sous_ensembles = [
        (4, h_soleil),
        (2, h_couvert),
        (4, h_pluie),
    ]

    gain = gain_information(entropie_parent, sous_ensembles)
    print(f"Gain pour Temps : {gain}")

    # Sigmoïde 
    print(f"sigmoide(-2) = {sigmoide(-2):.4f}")
    print(f"sigmoide(0) = {sigmoide(0):.4f}")
    print(f"sigmoide(2) = {sigmoide(2):.4f}")

    # Affichage de la sigmoïde 
    x_values = [i / 10 for i in range(-100, 101)]
    y_values = [sigmoide(x) for x in x_values]

    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, color="royalblue")
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Seuil 0.5")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1, label="z = 0")
    plt.title("La fonction sigmoïde")
    plt.xlabel("Score z")
    plt.ylabel("Probabilité")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Log loss
    print(f"log_loss(1, 0.9) = {log_loss(1, 0.9)}")
    print(f"log_loss(1, 0.1) = {log_loss(1, 0.1)}")

    # Mini régression logistique  
    beta0, beta1 = 0.5, 0.3
    x = 1
    z = beta0 + beta1 * x
    p = sigmoide(z)
    perte = log_loss(1, p)

    print(f"z = {z}")
    print(f"probabilite = {p:.4f}")
    print(f"log_loss = {perte:.4f}")
