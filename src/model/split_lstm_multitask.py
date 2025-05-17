import os
import numpy as np

# Dossiers d'entrée et de sortie
input_dir = "data/splits/lstm_datasets"
output_dir = "data/splits/lstm_splits_multitask"
os.makedirs(output_dir, exist_ok=True)

# Liste des cultures (ordre important pour les labels multi-sorties)
cultures = ["colza", "tournesol", "lavande", "pommiers"]

X_list, y_list = [], []

# Chargement des données pour chaque culture avec vérification
for culture in cultures:
    try:
        X = np.load(os.path.join(input_dir, f"X_{culture}.npy"))
        y = np.load(os.path.join(input_dir, f"y_{culture}.npy"))
    except Exception as e:
        print(f"[ERREUR] Chargement des fichiers pour {culture} : {e}")
        continue
    print(f"[INFO] {culture}: X shape = {X.shape}, y shape = {y.shape}")
    X_list.append(X)
    y_list.append(y)

# Vérification que toutes les séquences X et y ont la même longueur
if not all(len(X_list[0]) == len(x) for x in X_list):
    raise ValueError("Les fichiers X n'ont pas la même longueur")
if not all(len(y_list[0]) == len(y) for y in y_list):
    raise ValueError("Les fichiers y n'ont pas la même longueur")

# Utiliser X du premier fichier (NB : les séquences doivent être identiques pour chaque culture)
X = X_list[0]
# Empiler les labels de chaque culture pour obtenir une sortie multi-sortie de shape (n, 4)
y_multi = np.stack(y_list, axis=1)

print(f"[INFO] Nombre total de séquences après fusion: {X.shape[0]}")
print(f"[INFO] Forme finale de X: {X.shape}, forme finale de y multi: {y_multi.shape}")

# Splitting temporel en respectant l'ordre chronologique : 70%/15%/15%
n = len(X)
i_train = int(n * 0.7)
i_val = int(n * 0.85)

X_train, y_train = X[:i_train], y_multi[:i_train]
X_val, y_val = X[i_train:i_val], y_multi[i_train:i_val]
X_test, y_test = X[i_val:], y_multi[i_val:]

# Sauvegarde des splits
np.save(os.path.join(output_dir, "X_train_multi.npy"), X_train)
np.save(os.path.join(output_dir, "y_train_multi.npy"), y_train)
np.save(os.path.join(output_dir, "X_val_multi.npy"), X_val)
np.save(os.path.join(output_dir, "y_val_multi.npy"), y_val)
np.save(os.path.join(output_dir, "X_test_multi.npy"), X_test)
np.save(os.path.join(output_dir, "y_test_multi.npy"), y_test)

print(f"[✅] Split multi-tâche effectué :")
print(f"  Train: {X_train.shape[0]} séquences")
print(f"  Validation: {X_val.shape[0]} séquences")
print(f"  Test: {X_test.shape[0]} séquences")
print(f"[→] Forme X: {X.shape}, Forme y: {y_multi.shape}")
