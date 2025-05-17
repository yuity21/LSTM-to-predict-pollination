import requests
import pandas as pd
import os
import json

# Dossier de sortie
output_dir = "data/raw/bee_weather_data"
os.makedirs(output_dir, exist_ok=True)

# Coordonnées des villes
cities = {
    "Pau": {"lat": 43.2951, "lon": -0.3708},
    "Orleans": {"lat": 47.9025, "lon": 1.9090},
    "Avignon": {"lat": 43.9493, "lon": 4.8055},
}

# Variables météo à récupérer
variables = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
    "sunshine_duration"
]

# Période d'étude
start_year = 2017
end_year = 2023
start_month = 1
end_month = 12

# Fonction principale
def fetch_weather_data(city, lat, lon):
    all_data = []
    all_json = []

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-{start_month:02d}-01"
        end_date = f"{year}-{end_month:02d}-30"
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily={','.join(variables)}"
            f"&timezone=auto"
        )
        print(f"[INFO] Téléchargement de {city} pour l'année {year}...")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Vérifie si le code de statut est 200
        except Exception as e:
            print(f"[ERREUR] {city} {year} : {e}")
            continue

        data_json = response.json()
        df = pd.DataFrame(data_json["daily"])
        df["city"] = city
        all_data.append(df)
        all_json.append(data_json)
        print(f"[OK] {city} {year} téléchargé.")

    # Fusionner toutes les années si on a des données
    if all_data:
        df_total = pd.concat(all_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f"{city}_2019_2023.csv")
        df_total.to_csv(csv_path, index=False)

        json_path = os.path.join(output_dir, f"{city}_2019_2023.json")
        with open(json_path, "w") as f:
            json.dump(all_json, f, indent=2)

        print(f"[OK] {city} — Données enregistrées dans {csv_path} et {json_path}")
    else:
        print(f"[ERREUR] Aucune donnée n'a été récupérée pour {city}.")

# Lancer pour toutes les villes
for city, coords in cities.items():
    fetch_weather_data(city, coords["lat"], coords["lon"])
