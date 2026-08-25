# importation des bibliotheques
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset (Client, Age, Revenu, Achat)
data = [
    (1, "Jeune", "Faible", "Non"),
    (2, "Jeune", "Faible", "Non"),
    (3, "Vieux", "Faible", "Non"),
    (4, "Vieux", "Élevé", "Oui"),
    (5, "Vieux", "Élevé", "Oui"),
    (6, "Jeune", "Élevé", "Oui"),
    (7, "Jeune", "Faible", "Non"),
    (8, "Vieux", "Faible", "Non"),
    (9, "Jeune", "Élevé", "Oui"),
    (10, "Vieux", "Élevé", "Oui"),
]


# Fonction entropie détaillée : affiche chaque étape du calcul
def entropie_detaillee(nb_oui, nb_non, nom="H"):
    total = nb_oui + nb_non
    proportions = []
    if nb_oui > 0:
        proportions.append(nb_oui / total)
    if nb_non > 0:
        proportions.append(nb_non / total)

    print(f"--- Calcul de {nom} ---")
    somme = 0.0
    for p in proportions:
        log2p = math.log2(p)
        terme = p * log2p
        somme += terme
        print(f"  p = {p:.4f}  |  log2(p) = {log2p:.4f}  |  p * log2(p) = {terme:.4f}")

    h = -somme
    print(f"  Somme des termes = {somme:.4f}")
    print(f"  {nom} = - (somme) = -({somme:.4f}) = {h:.4f} bit")
    print()
    return h

# Fonction gain (avec sous-ensembles imprimés)
def gain_caracteristique(nom_caracteristique, h_parent, sous_groupes, total):

    print("#" * 50)
    print(f"Étape : Gain pour la caractéristique '{nom_caracteristique}'")
    print("#" * 50)
    print()

    entropie_ponderee = 0.0
    for modalite, (oui, non) in sous_groupes.items():
        n = oui + non
        print(f"--- Sous-ensemble '{modalite}' ({n} clients) ---")
        print(f"Oui : {oui}, Non : {non}")
        h_sv = entropie_detaillee(oui, non, nom=f"H({modalite})")
        entropie_ponderee += (n / total) * h_sv

    gain = h_parent - entropie_ponderee
    print(f"Entropie pondérée après split sur '{nom_caracteristique}' = {entropie_ponderee:.4f}")
    print(f"Gain('{nom_caracteristique}') = {h_parent:.4f} - {entropie_ponderee:.4f} = {gain:.4f} bit")
    print()
    return gain

# PROGRAMME PRINCIPAL
if __name__ == "__main__":

    total = len(data)
    nb_oui = sum(1 for c in data if c[3] == "Oui")
    nb_non = sum(1 for c in data if c[3] == "Non")
    
    # Étape 1 : Entropie initiale H(S)
    print("#" * 50)
    print("Étape 1 : Entropie initiale H(S)")
    print("#" * 50)
    print(f"Nombre total de clients : {total}")
    print(f"Nombre de Oui : {nb_oui}, Nombre de Non : {nb_non}")
    print(f"Proportions : Oui = {nb_oui/total:.4f}, Non = {nb_non/total:.4f}")
    print()

    h_s = entropie_detaillee(nb_oui, nb_non, nom="H(S)")

    # Étape 2 : Gain pour 'Âge'
    groupes_age = {}
    for modalite in ["Jeune", "Vieux"]:
        oui = sum(1 for c in data if c[1] == modalite and c[3] == "Oui")
        non = sum(1 for c in data if c[1] == modalite and c[3] == "Non")
        groupes_age[modalite] = (oui, non)

    gain_age = gain_caracteristique("Âge", h_s, groupes_age, total)

    # Étape 3 : Gain pour 'Revenu'
    groupes_revenu = {}
    for modalite in ["Faible", "Élevé"]:
        oui = sum(1 for c in data if c[2] == modalite and c[3] == "Oui")
        non = sum(1 for c in data if c[2] == modalite and c[3] == "Non")
        groupes_revenu[modalite] = (oui, non)

    gain_revenu = gain_caracteristique("Revenu", h_s, groupes_revenu, total)

    # Étape 4 : Conclusion - première question de l'arbre
    print("#" * 50)
    print("Étape 4 : Comparaison des gains")
    print("#" * 50)
    print(f"Gain(Âge)    = {gain_age:.4f} bit")
    print(f"Gain(Revenu) = {gain_revenu:.4f} bit")

    meilleure_carac = "Revenu" if gain_revenu > gain_age else "Âge"
    print(f"=> La première question de l'arbre porte sur : '{meilleure_carac}'")
    print("   (c'est la caractéristique qui apporte le plus de gain d'information)")
    print()

    # Étape 5 : Construction de l'arbre complet
    print("#" * 50)
    print("Étape 5 : Arbre complet")
    print("#" * 50)
    h_faible = entropie_detaillee(*groupes_revenu["Faible"], nom="H(Revenu=Faible)")
    h_eleve = entropie_detaillee(*groupes_revenu["Élevé"], nom="H(Revenu=Élevé)")

    if h_faible == 0 and h_eleve == 0:
        print("Les deux sous-ensembles issus du split sur 'Revenu' sont purs")
        print("(entropie = 0), donc l'arbre s'arrête après cette unique division.")
    print()
    print("Arbre final :")
    print("  Revenu ?")
    print("  ├── Faible -> Achat = Non  (H = 0.00 bit)")
    print("  └── Élevé  -> Achat = Oui  (H = 0.00 bit)")

    # Visualisation de l'arbre (comme Figure 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Arbre de décision construit à partir du mini-dataset clients")

    # Noeud racine : Revenu
    root = patches.FancyBboxPatch((3.5, 5.8), 3, 1.4, boxstyle="round,pad=0.1",
                                   facecolor="#7fc7e8", edgecolor="black")
    ax.add_patch(root)
    ax.text(5, 6.5, f"Revenu ?\nH = {h_s:.2f} bit", ha="center", va="center", fontsize=10)

    # Noeud gauche : Faible -> Non
    left = patches.FancyBboxPatch((0.5, 2.5), 3, 1.4, boxstyle="round,pad=0.1",
                                   facecolor="#f4a3a3", edgecolor="black")
    ax.add_patch(left)
    ax.text(2, 3.2, f"Achat = Non\nH = {h_faible:.2f} bit", ha="center", va="center", fontsize=10)

    # Noeud droit : Élevé -> Oui
    right = patches.FancyBboxPatch((6.5, 2.5), 3, 1.4, boxstyle="round,pad=0.1",
                                    facecolor="#a8e6a1", edgecolor="black")
    ax.add_patch(right)
    ax.text(8, 3.2, f"Achat = Oui\nH = {h_eleve:.2f} bit", ha="center", va="center", fontsize=10)

    # Flèches
    ax.annotate("", xy=(2, 3.9), xytext=(4.3, 5.8),
                arrowprops=dict(arrowstyle="->"))
    ax.text(3, 5.0, "Faible", fontsize=9)

    ax.annotate("", xy=(8, 3.9), xytext=(5.7, 5.8),
                arrowprops=dict(arrowstyle="->"))
    ax.text(6.8, 5.0, "Élevé", fontsize=9)

    plt.tight_layout()
    plt.show()