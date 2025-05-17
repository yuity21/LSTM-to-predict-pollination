# Interface de Prédiction de Pollinisation

Cette interface permet de visualiser les prédictions d'activité de pollinisation pour l'année 2025 basées sur des données météorologiques d'Open Meteo et un modèle LSTM multi-tâches préalablement entraîné.

## Fonctionnalités

- Sélection de la ville (Pau, Orléans, Avignon)
- Sélection du mois pour visualiser les prédictions
- Affichage des prédictions sous forme de calendrier pour 4 cultures :
  - Colza
  - Tournesol
  - Lavande
  - Pommiers
- Code couleur pour les niveaux d'activité de pollinisation (basse, moyenne, haute)

## Installation et lancement

1. Assurez-vous d'avoir installé toutes les dépendances requises :

```bash
pip install -r requirements.txt
```

2. Lancez l'application Streamlit :

```bash
streamlit run interface/app.py
```

## Comment ça fonctionne

1. L'application charge le modèle LSTM entraîné (situé dans `models/lstm_models_multitask/best_multitask_lstm.keras`)
2. Elle télécharge les données météorologiques via l'API Open Meteo (en utilisant 2023 comme proxy pour 2025)
3. Les données sont préparées et transformées en séquences pour le modèle LSTM
4. Le modèle prédit l'activité de pollinisation pour chaque jour et chaque culture
5. Les résultats sont affichés dans un calendrier mensuel interactif

## Notes importantes

- Pour la démonstration, les données de 2025 sont simulées en utilisant les données de 2023 avec de légères variations
- Les prédictions sont calculées à la volée et ne sont pas stockées
- Le modèle est chargé avec `compile=False` car nous n'avons besoin que de la prédiction, pas de l'entraînement 