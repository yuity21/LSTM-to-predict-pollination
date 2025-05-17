import pandas as pd
import os

# Répertoires d'entrée et de sortie
input_dir = "data/processed/bee_weather_data_with_flowering"
output_dir = "data/processed/bee_weather_data_labeled"

os.makedirs(output_dir, exist_ok=True)

# 🌸 Règles de floraison par culture (mois)
flowering_periods = {
    "colza": [3, 4, 5],       # mars à mai
    "tournesol": [6, 7, 8],   # juin à août
    "lavande": [6, 7],        # juin à juillet
    "pommiers": [4, 5],       # avril à mai
}

# Fonction de calcul du score d'activité synthétique
def compute_score(row, crop):
    # Si la plante n'est pas en floraison, le score est de base 0
    if row[f"is_flowering_{crop}"] == 0:
        return 0

    favorable = 0

    # Définition de seuils personnalisés selon la culture
    if crop == "lavande":
        temp_lower, temp_upper = 14, 30    # Seuils ajustés pour la lavande
        sunshine_threshold = 30            # Seuil: 30 minutes d'ensoleillement
    elif crop == "tournesol":
        temp_lower, temp_upper = 15, 32    # Seuils pour le tournesol
        sunshine_threshold = 40            # Seuil: 40 minutes d'ensoleillement
    else:
        temp_lower, temp_upper = 15, 32
        sunshine_threshold = 60            # Seuil: 60 minutes d'ensoleillement

    # Vérification de la température
    if temp_lower <= row["temperature_2m_max"] <= temp_upper:
        favorable += 1

    # Vérification des précipitations (aucune pluie)
    if row["precipitation_sum"] == 0:
        favorable += 1

    # Vérification de la vitesse du vent (inférieure à 25 km/h)
    if row["wind_speed_10m_max"] < 25:
        favorable += 1

    # Conversion de l'ensoleillement : de secondes en minutes
    sunshine_minutes = row["sunshine_duration"] / 60
    if sunshine_minutes >= sunshine_threshold:
        favorable += 1

    # Log interne pour visualiser le nombre de conditions favorables sur un échantillon
    # print(f"[DEBUG] {crop} - Temp: {row['temperature_2m_max']}, Precip: {row['precipitation_sum']}, Vent: {row['wind_speed_10m_max']}, Sunshine (min): {sunshine_minutes}, Conditions favorables: {favorable}")

    # Attribution du score en fonction du nombre de conditions satisfaites
    if favorable == 4:
        return 2  # Haute activité
    elif favorable >= 2:
        return 1  # Moyenne activité
    else:
        return 0  # Basse activité

# Traitement de chaque fichier
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[ERREUR] Lecture de {file} : {e}")
            continue

        # Traitement pour chaque culture
        for crop in ["colza", "tournesol", "lavande", "pommiers"]:
            score_col = f"activity_score_{crop}"
            df[score_col] = df.apply(lambda row: compute_score(row, crop), axis=1)
            print(f"[INFO] Labels générés pour {crop} dans {file}")

        # Sauvegarde du fichier labellisé
        output_file = os.path.join(output_dir, file)
        try:
            df.to_csv(output_file, index=False)
            print(f"[✅] Données synthétiques enregistrées : {output_file}")
        except Exception as e:
            print(f"[ERREUR] Enregistrement de {output_file} : {e}")
