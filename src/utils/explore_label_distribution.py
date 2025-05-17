import numpy as np

def show_distribution(file_path):
    # Charge le fichier et calcule la distribution des labels
    y = np.load(file_path)
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    return dist, len(y)

files = {
    "colza": "data/splits/lstm_datasets/y_colza.npy",
    "lavande": "data/splits/lstm_datasets/y_lavande.npy",
    "pommiers": "data/splits/lstm_datasets/y_pommiers.npy",
    "tournesol": "data/splits/lstm_datasets/y_tournesol.npy",
}

for culture, file_path in files.items():
    try:
        dist, total = show_distribution(file_path)
        print(f"\nDistribution pour {culture} (total: {total} échantillons) :")
        for label in sorted(dist.keys()):
            pct = (dist[label] / total) * 100
            print(f"  Label {label} : {dist[label]} ({pct:.1f}%)")
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path} : {e}")
