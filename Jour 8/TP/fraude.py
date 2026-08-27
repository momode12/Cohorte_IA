import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef
)


# ============================================================
# Q1 : CHARGEMENT DU DATASET ET PRÉDICTION AU SEUIL 0.5
# ============================================================

df = pd.read_csv("dataset_fraude.csv")

seuil = 0.5

df["y_pred"] = (df["y_proba"] >= seuil).astype(int)

print("\n" + "=" * 60)
print("Q1 : CHARGEMENT DU DATASET")
print("=" * 60)

print(f"Nombre total de transactions : {len(df)}")
print(f"Seuil utilisé : {seuil}")
print(f"Nombre de fraudes prédites : {df['y_pred'].sum()}")

print("\nAperçu du dataset :")
print(df.head(10))


# ============================================================
# Q2 : MATRICE DE CONFUSION
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    df["y_true"],
    df["y_pred"],
    labels=[0, 1]
).ravel()

print("\n" + "=" * 60)
print("Q2 : MATRICE DE CONFUSION")
print("=" * 60)

print(f"VP (Vrais Positifs)  : {tp}")
print(f"VN (Vrais Négatifs)  : {tn}")
print(f"FP (Faux Positifs)   : {fp}")
print(f"FN (Faux Négatifs)   : {fn}")


# ============================================================
# Q3 : ACCURACY, PRÉCISION, RAPPEL, F1
# ============================================================

accuracy_manuel = (tp + tn) / (tp + tn + fp + fn)

# Protection contre division par zéro
if tp + fp > 0:
    precision_manuel = tp / (tp + fp)
else:
    precision_manuel = 0

if tp + fn > 0:
    rappel_manuel = tp / (tp + fn)
else:
    rappel_manuel = 0

if precision_manuel + rappel_manuel > 0:
    f1_manuel = (
        2 * precision_manuel * rappel_manuel
        / (precision_manuel + rappel_manuel)
    )
else:
    f1_manuel = 0


print("\n" + "=" * 60)
print("Q3 : MÉTRIQUES - CALCUL MANUEL")
print("=" * 60)

print(f"Accuracy  : {accuracy_manuel:.3f}")
print(f"Précision : {precision_manuel:.3f}")
print(f"Rappel    : {rappel_manuel:.3f}")
print(f"F1-Score  : {f1_manuel:.3f}")


print("\n--- Vérification avec Scikit-Learn ---")

accuracy_sklearn = accuracy_score(
    df["y_true"],
    df["y_pred"]
)

precision_sklearn = precision_score(
    df["y_true"],
    df["y_pred"],
    zero_division=0
)

rappel_sklearn = recall_score(
    df["y_true"],
    df["y_pred"],
    zero_division=0
)

f1_sklearn = f1_score(
    df["y_true"],
    df["y_pred"],
    zero_division=0
)

print(f"Accuracy  : {accuracy_sklearn:.3f}")
print(f"Précision : {precision_sklearn:.3f}")
print(f"Rappel    : {rappel_sklearn:.3f}")
print(f"F1-Score  : {f1_sklearn:.3f}")


# ============================================================
# Q4 : COMPARAISON AVEC UN MODÈLE NAÏF
# ============================================================

y_pred_naif = pd.Series([0] * len(df))

accuracy_naif = accuracy_score(
    df["y_true"],
    y_pred_naif
)

proportion_fraudes = df["y_true"].mean()


print("\n" + "=" * 60)
print("Q4 : MODÈLE NAÏF VS MODÈLE RÉEL")
print("=" * 60)

print(
    f"Accuracy du modèle naïf (toujours 0) : "
    f"{accuracy_naif:.3f}"
)

print(
    f"Accuracy du modèle réel              : "
    f"{accuracy_sklearn:.3f}"
)

print(
    f"\nProportion de fraudes dans le dataset : "
    f"{proportion_fraudes:.3%}"
)


# ============================================================
# Q5 : IMPACT DU SEUIL SUR PRÉCISION / RAPPEL / F1
# ============================================================

print("\n" + "=" * 60)
print("Q5 : IMPACT DU SEUIL SUR PRÉCISION / RAPPEL")
print("=" * 60)

for seuil_test in [0.3, 0.5, 0.7]:

    y_pred_test = (
        df["y_proba"] >= seuil_test
    ).astype(int)

    prec = precision_score(
        df["y_true"],
        y_pred_test,
        zero_division=0
    )

    rap = recall_score(
        df["y_true"],
        y_pred_test,
        zero_division=0
    )

    f1 = f1_score(
        df["y_true"],
        y_pred_test,
        zero_division=0
    )

    n_alertes = y_pred_test.sum()

    print(f"\nSeuil = {seuil_test}")

    print(
        f"  Nombre d'alertes déclenchées : "
        f"{n_alertes}"
    )

    print(f"  Précision : {prec:.4f}")
    print(f"  Rappel    : {rap:.4f}")
    print(f"  F1-Score  : {f1:.4f}")


# ============================================================
# Q6 : COURBE ROC ET AUC-ROC
# ============================================================

fpr, tpr, seuils_roc = roc_curve(
    df["y_true"],
    df["y_proba"]
)

auc_roc = roc_auc_score(
    df["y_true"],
    df["y_proba"]
)

print("\n" + "=" * 60)
print("Q6 : AUC-ROC")
print("=" * 60)

print(f"AUC-ROC : {auc_roc:.3f}")


# ============================================================
# Q7 : COURBE PRECISION-RECALL ET PR-AUC
# ============================================================

precision_vals, recall_vals, seuils_pr = precision_recall_curve(
    df["y_true"],
    df["y_proba"]
)

pr_auc = average_precision_score(
    df["y_true"],
    df["y_proba"]
)

print("\n" + "=" * 60)
print("Q7 : PR-AUC")
print("=" * 60)

print(f"PR-AUC : {pr_auc:.3f}")


# ============================================================
# Q8 : MCC
# ============================================================

mcc = matthews_corrcoef(
    df["y_true"],
    df["y_pred"]
)

print("\n" + "=" * 60)
print("Q8 : MCC")
print("=" * 60)

print(f"MCC : {mcc:.3f}")


# ============================================================
# Q9 : MATRICE DE COÛTS ET SEUIL OPTIMAL
# ============================================================

cout_fp = 15
cout_fn = 500

print("\n" + "=" * 60)
print("Q9 : COÛTS PAR SEUIL")
print("=" * 60)

meilleur_seuil = None
meilleur_cout = None

for seuil_test in [
    round(i / 100, 2)
    for i in range(5, 100, 5)
]:

    y_pred_test = (
        df["y_proba"] >= seuil_test
    ).astype(int)

    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
        df["y_true"],
        y_pred_test,
        labels=[0, 1]
    ).ravel()

    cout_total = (
        fp_t * cout_fp
        + fn_t * cout_fn
    )

    print(
        f"Seuil = {seuil_test:.2f} | "
        f"FP = {fp_t} | "
        f"FN = {fn_t} | "
        f"Coût total = {cout_total} €"
    )

    if meilleur_cout is None or cout_total < meilleur_cout:

        meilleur_cout = cout_total
        meilleur_seuil = seuil_test


cout_seuil_05 = (
    fp * cout_fp
    + fn * cout_fn
)

print("\n--- Résultat de l'optimisation ---")

print(
    f"Coût total au seuil 0.5 : "
    f"{cout_seuil_05} €"
)

print(
    f"Meilleur seuil trouvé    : "
    f"{meilleur_seuil}"
)

print(
    f"Coût minimal             : "
    f"{meilleur_cout} €"
)


# ============================================================
# Q10 : RAPPEL PAR RÉGION
# ============================================================

print("\n" + "=" * 60)
print("Q10 : RAPPEL PAR RÉGION")
print("=" * 60)

for region in sorted(df["region"].unique()):

    df_region = df[
        df["region"] == region
    ]

    rappel_region = recall_score(
        df_region["y_true"],
        df_region["y_pred"],
        zero_division=0
    )

    n_fraudes_region = df_region["y_true"].sum()

    print(
        f"Région {region} : "
        f"Rappel = {rappel_region:.3f} "
        f"(sur {n_fraudes_region} fraudes réelles)"
    )


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL DU MODÈLE")
print("=" * 60)

print(f"Transactions totales : {len(df)}")
print(f"Fraudes réelles      : {df['y_true'].sum()}")
print(f"Fraudes prédites     : {df['y_pred'].sum()}")

print(f"\nAccuracy  : {accuracy_sklearn:.3f}")
print(f"Précision : {precision_sklearn:.3f}")
print(f"Rappel    : {rappel_sklearn:.3f}")
print(f"F1-Score  : {f1_sklearn:.3f}")
print(f"MCC       : {mcc:.3f}")
print(f"AUC-ROC   : {auc_roc:.3f}")
print(f"PR-AUC    : {pr_auc:.3f}")

print(f"\nSeuil optimal selon les coûts : {meilleur_seuil}")
print(f"Coût minimal                  : {meilleur_cout} €")


# ============================================================
# FIGURES
# Les figures sont affichées UNIQUEMENT APRÈS tous les résultats
# ============================================================


# ------------------------------------------------------------
# FIGURE 1 : MATRICE DE CONFUSION
# ------------------------------------------------------------

cm_perso = [
    [tp, fp],
    [tn, fn]
]

labels_cm = [
    ["VP", "FP"],
    ["VN", "FN"]
]

fig_cm, ax_cm = plt.subplots(figsize=(5, 5))

ax_cm.imshow(
    cm_perso,
    cmap="Blues"
)

valeur_max = max(
    tp,
    fp,
    tn,
    fn
)

for i in range(2):

    for j in range(2):

        valeur = cm_perso[i][j]
        label = labels_cm[i][j]

        couleur_texte = (
            "white"
            if valeur > valeur_max / 2
            else "black"
        )

        ax_cm.text(
            j,
            i,
            f"{label}\n{valeur}",
            ha="center",
            va="center",
            color=couleur_texte,
            fontsize=14,
            fontweight="bold"
        )

ax_cm.set_xticks([])
ax_cm.set_yticks([])

ax_cm.set_title(
    "Matrice de confusion (seuil = 0.5)"
)

plt.tight_layout()

plt.savefig(
    "matrice_confusion.png",
    dpi=300
)


# ------------------------------------------------------------
# FIGURE 2 : COURBE ROC
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Modèle (AUC = {auc_roc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Hasard (AUC = 0.5)"
)

plt.xlabel(
    "Taux de Faux Positifs (FPR)"
)

plt.ylabel(
    "Taux de Vrais Positifs (Rappel)"
)

plt.title("Courbe ROC")

plt.legend()

plt.tight_layout()

plt.savefig(
    "courbe_roc.png",
    dpi=300
)


# ------------------------------------------------------------
# FIGURE 3 : COURBE PRECISION-RECALL
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.plot(
    recall_vals,
    precision_vals,
    label=f"Modèle (PR-AUC = {pr_auc:.3f})"
)

baseline = df["y_true"].mean()

plt.axhline(
    y=baseline,
    linestyle="--",
    label=f"Hasard (baseline = {baseline:.3f})"
)

plt.xlabel("Rappel")
plt.ylabel("Précision")

plt.title(
    "Courbe Precision-Recall"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "courbe_pr.png",
    dpi=300
)


# ============================================================
# AFFICHAGE DES 3 FIGURES À LA FIN
# ============================================================

print("\n" + "=" * 60)
print("FIGURES")
print("=" * 60)

print("✓ matrice_confusion.png")
print("✓ courbe_roc.png")
print("✓ courbe_pr.png")

print("\nAffichage des graphiques...")

plt.show()