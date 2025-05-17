import pandas as pd
import os

# Dossier contenant les fichiers météo
input_dir = "data/raw/bee_weather_data"
output_dir = "data/processed/bee_weather_data_with_flowering"

os.makedirs(output_dir, exist_ok=True)

# Règles de floraison par culture (mois)
flowering_periods = {
    "colza": [3, 4, 5],       # mars à mai
    "tournesol": [6, 7, 8],   # juin à août
    "lavande": [6, 7],        # juin à juillet
    "pommiers": [4, 5],       # avril à mai
}

# Traitement de chaque fichier météo
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[ERREUR] Lecture du fichier {file} : {e}")
            continue
        
        # Vérifier la présence de la colonne 'time'
        if "time" not in df.columns:
            print(f"[SKIP] Pas de colonne 'time' dans {file}")
            continue

        # Conversion robuste de la colonne 'time'
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # Supprimer les lignes où la conversion a échoué
        initial_count = len(df)
        df = df.dropna(subset=["time"])
        removed = initial_count - len(df)
        if removed > 0:
            print(f"[INFO] {removed} lignes supprimées dans {file} à cause d'une date invalide.")

        # Extraire le mois à partir de la colonne 'time'
        df["month"] = df["time"].dt.month

        # Ajouter pour chaque culture une colonne indiquant la floraison
        for crop, months in flowering_periods.items():
            df[f"is_flowering_{crop}"] = df["month"].apply(lambda m: 1 if m in months else 0)

        # Sauvegarder dans le nouveau dossier
        output_file = os.path.join(output_dir, file)
        try:
            df.to_csv(output_file, index=False)
            print(f"[✅] Fichier enrichi enregistré : {output_file}")
        except Exception as e:
            print(f"[ERREUR] Enregistrement du fichier {output_file} : {e}")
