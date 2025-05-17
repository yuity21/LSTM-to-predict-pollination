import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Chemin vers les fichiers générés par la préparation LSTM
input_dir = "data/splits/lstm_datasets"
# Dossier de sauvegarde des graphiques
output_dir = "models/graphs"
os.makedirs(output_dir, exist_ok=True)

# Liste des cultures
cultures = ["colza", "tournesol", "lavande", "pommiers"]

for culture in cultures:
    file_path = os.path.join(input_dir, f"y_{culture}.npy")
    try:
        y = np.load(file_path)
        print(f"[INFO] {culture}: Fichier chargé avec {len(y)} échantillons.")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger {file_path}: {e}")
        continue

    # Calcul de la distribution des classes
    counts = Counter(y)
    total = len(y)
    print(f"\n📊 Distribution pour {culture} (total: {total} échantillons) :")
    for cls in sorted(counts):
        pct = (counts[cls] / total) * 100
        print(f"  Classe {cls} : {counts[cls]} ({pct:.1f}%)")
    
    # Création du graphique en barres
    plt.figure(figsize=(6, 4))
    labels = [str(k) for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]
    plt.bar(labels, values, color='blue', alpha=0.7)
    plt.title(f"Répartition des classes — {culture}")
    plt.xlabel("Classe (0=basse, 1=moyenne, 2=haute)")
    plt.ylabel("Nombre de séquences")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{culture}_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Graphique sauvegardé pour {culture} dans {plot_path}")
