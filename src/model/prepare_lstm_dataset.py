import os
import pandas as pd
import numpy as np

# Chemins : source des fichiers labellisés et destination des datasets LSTM
input_dir = "data/processed/bee_weather_data_labeled"
output_dir = "data/splits/lstm_datasets"

os.makedirs(output_dir, exist_ok=True)

# Paramètres
cultures = ["colza", "tournesol", "lavande", "pommiers"]
sequence_length = 10  # Nombre de jours d'entrée
features = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
    "sunshine_duration",
]

def prepare_lstm_data(df, culture, seq_len):
    # Conversion robuste de la colonne "time"
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    initial_count = len(df)
    df = df.dropna(subset=["time"])
    if len(df) < initial_count:
        print(f"[INFO] {initial_count - len(df)} lignes supprimées à cause de dates invalides pour {culture}.")

    # Trier par date
    df = df.sort_values("time")

    # Colonnes à utiliser pour les features
    f_cols = features + [f"is_flowering_{c}" for c in cultures]
    target_col = f"activity_score_{culture}"

    # Copie de df pour la normalisation
    df_norm = df.copy()

    # Normalisation min-max de chaque colonne de features
    for col in f_cols:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        if col_max == col_min:
            print(f"[WARN] La colonne {col} a une variance nulle. Elle sera remplacée par 0.")
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
    
    # Construction des séquences X et des labels y
    X = []
    y = []
    for i in range(len(df_norm) - seq_len):
        X_seq = df_norm[f_cols].iloc[i:i + seq_len].values
        y_label = df_norm[target_col].iloc[i + seq_len]
        X.append(X_seq)
        y.append(y_label)
    
    X = np.array(X)
    y = np.array(y)
    return X, y

# Traitement pour chaque culture
for culture in cultures:
    all_X, all_y = [], []
    print(f"[INFO] Préparation des données LSTM pour {culture}")
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"[ERREUR] Lecture de {file} pour {culture} : {e}")
                continue

            if f"activity_score_{culture}" not in df.columns:
                print(f"[INFO] {file} n'a pas la colonne activity_score_{culture}. Passage au fichier suivant.")
                continue

            X, y = prepare_lstm_data(df, culture, sequence_length)
            all_X.append(X)
            all_y.append(y)

    if all_X:
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        np.save(os.path.join(output_dir, f"X_{culture}.npy"), X_final)
        np.save(os.path.join(output_dir, f"y_{culture}.npy"), y_final)
        print(f"[✅] Données LSTM générées pour {culture} : {X_final.shape[0]} séquences")
    else:
        print(f"[ERREUR] Aucune donnée valide pour {culture}.")
