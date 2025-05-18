import os
import pandas as pd
import matplotlib.pyplot as plt

# Dossier contenant les fichiers complets avec labels
data_dir = "bee_weather_data_labeled"
cultures = ["colza", "tournesol", "lavande", "pommiers"]

# Plot par culture
for culture in cultures:
    plt.figure(figsize=(15, 6))
    first_file = True

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file))
            if f"activity_score_{culture}" not in df.columns:
                continue

            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")
            df = df.dropna(subset=["temperature_2m_max", "sunshine_duration", f"activity_score_{culture}"])

            # Affichage sur le premier fichier uniquement pour éviter surcharge
            if first_file:
                plt.plot(df["time"], df["temperature_2m_max"], label="Température max (°C)", color="orange", alpha=0.6)
                plt.plot(df["time"], df["sunshine_duration"], label="Ensoleillement (min)", color="gold", alpha=0.5)

                # Floraison = bande colorée
                plt.fill_between(df["time"],
                                 0,
                                 1,
                                 where=df[f"is_flowering_{culture}"] == 1,
                                 color="lightgreen",
                                 alpha=0.3,
                                 label="Floraison")
                
                # 🐝 Score d’activité
                plt.plot(df["time"], df[f"activity_score_{culture}"], label="Score d'activité", color="purple")

                first_file = False

    plt.title(f"🌿 Activité pollinisatrice – {culture.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Valeurs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graph_{culture}.png")  # Export PNG
    plt.show()
