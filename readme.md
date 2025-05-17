# Système de Prédiction d'Activité des Pollinisateurs

Ce projet développe un système d'intelligence artificielle pour prédire l'activité des pollinisateurs basé sur des données météorologiques et des périodes de floraison.

## Table des matières

1. [Structure du projet](#structure-du-projet)
2. [Installation](#installation)
3. [Exécution du pipeline complet](#exécution-du-pipeline-complet)
4. [Exécution étape par étape](#exécution-étape-par-étape)
5. [Interface utilisateur](#interface-utilisateur)
6. [Description des données](#description-des-données)
7. [Architecture des modèles](#architecture-des-modèles)

## Structure du projet

```
├── data/                      # Stockage des données
│   ├── raw/                   # Données brutes téléchargées
│   ├── processed/             # Données traitées
│   ├── splits/                # Ensembles d'entraînement/validation/test
│   └── figures/               # Graphiques et visualisations
├── interface/                 # Interface utilisateur Streamlit
│   ├── app.py                 # Application Streamlit
│   └── README.md              # Documentation de l'interface
├── models/                    # Modèles entraînés
│   ├── lstm_models_multitask/ # Modèles LSTM multi-tâches
│   └── lstm_analysis/         # Analyses de performance des modèles
├── notebooks/                 # Notebooks Jupyter pour l'exploration
├── src/                       # Code source
│   ├── data/                  # Scripts de préparation des données
│   │   ├── download_weather_data.py
│   │   ├── add_flowering_season.py
│   │   ├── generate_synthetic_activity_labels.py
│   │   └── explore_activity_distribution.py
│   ├── model/                 # Scripts d'entraînement des modèles
│   │   ├── prepare_lstm_dataset.py
│   │   ├── split_lstm_multitask.py
│   │   └── train_multitask_lstm.py
│   └── utils/                 # Fonctions utilitaires
├── venv/                      # Environnement virtuel Python de l'utilisateur qu'il devra créer
├── requirements.txt           # Dépendances Python
└── readme.md                  # Ce fichier
```

## Installation

Pour installer les dépendances nécessaires, suivez ces étapes :

```bash
# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows:
venv\Scripts\activate
# Sur macOS/Linux:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Exécution du pipeline complet

Pour exécuter l'ensemble du pipeline de A à Z, utilisez le script suivant:

```bash
# Sur Windows (PowerShell):
python src/data/download_weather_data.py
python src/data/add_flowering_season.py
python src/data/generate_synthetic_activity_labels.py
python src/data/explore_activity_distribution.py
python src/model/prepare_lstm_dataset.py
python src/model/split_lstm_multitask.py
python src/model/train_multitask_lstm.py

# Sur Linux/macOS:
bash readme.md
```

## Exécution étape par étape

### 1. Téléchargement des données météorologiques

```bash
python src/data/download_weather_data.py
```
Ce script récupère les données météorologiques historiques via l'API Open Meteo pour différentes villes françaises et les sauvegarde dans `data/raw/`.

### 2. Ajout des indicateurs de floraison

```bash
python src/data/add_flowering_season.py
```
Ce script ajoute des indicateurs binaires pour les périodes de floraison de différentes cultures (colza, tournesol, lavande, pommiers) aux données météorologiques.

### 3. Génération des étiquettes d'activité synthétiques

```bash
python src/data/generate_synthetic_activity_labels.py
```
Ce script génère des étiquettes synthétiques d'activité des pollinisateurs basées sur des règles prédéfinies liées aux conditions météorologiques et aux périodes de floraison.

### 4. Exploration de la distribution des activités

```bash
python src/data/explore_activity_distribution.py
```
Ce script crée des visualisations de la distribution des étiquettes d'activité pour comprendre la répartition des données.

### 5. Préparation des données pour le modèle LSTM

```bash
python src/model/prepare_lstm_dataset.py
```
Ce script transforme les données en séquences temporelles adaptées à l'entraînement d'un modèle LSTM.

### 6. Fractionnement des données pour l'entraînement multi-tâche

```bash
python src/model/split_lstm_multitask.py
```
Ce script divise les données en ensembles d'entraînement, de validation et de test pour l'apprentissage multi-tâche.

### 7. Entraînement du modèle LSTM multi-tâche

```bash
python src/model/train_multitask_lstm.py
```
Ce script entraîne un modèle LSTM multi-tâche pour prédire simultanément l'activité des pollinisateurs pour différentes cultures.

## Interface utilisateur

Pour lancer l'interface utilisateur Streamlit qui permet de visualiser les prédictions:

```bash
streamlit run interface/app.py
```

L'interface permet de:
- Sélectionner une ville (Pau, Orléans, Avignon)
- Visualiser les prédictions d'activité des pollinisateurs par mois
- Comparer l'activité pour différentes cultures (colza, tournesol, lavande, pommiers)

Pour plus de détails, consultez `interface/README.md`.

## Description des données

- **Données météorologiques**: Température, précipitations, vitesse du vent, humidité, etc.
- **Périodes de floraison**: Indicateurs binaires pour 4 cultures différentes
- **Étiquettes d'activité**: Niveaux d'activité des pollinisateurs (bas, moyen, élevé) pour chaque culture

## Architecture des modèles

Le projet utilise un modèle LSTM multi-tâche qui:
- Prend en entrée des séquences de données météorologiques et indicateurs de floraison
- Prédit simultanément l'activité des pollinisateurs pour 4 cultures différentes
- Utilise une couche partagée pour capturer les motifs communs et des couches spécifiques pour chaque culture

Les modèles entraînés sont sauvegardés dans le dossier `models/lstm_models_multitask/`.
