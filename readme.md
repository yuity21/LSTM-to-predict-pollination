#!/bin/bash
# Ce script exécute tout le pipeline du projet de A à Z.
# Assurez-vous d'avoir activé votre environnement virtuel avant de lancer ce script.

echo "==== Début du pipeline de prédiction des pollinisateurs ===="

echo "[1/7] Téléchargement des données météo..."
python src/data/download_weather_data.py
if [ $? -ne 0 ]; then
    echo "Erreur lors du téléchargement des données météo."
    exit 1
fi

echo "[2/7] Ajout des indicateurs de floraison aux données météo..."
python src/data/add_flowering_season.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'ajout des indicateurs de floraison."
    exit 1
fi

echo "[3/7] Génération des labels d'activité synthétiques..."
python src/data/generate_synthetic_activity_labels.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de la génération des labels d'activité."
    exit 1
fi

echo "[4/7] Exploration des distributions d'activité (facultatif)..."
python src/data/explore_activity_distribution.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exploration de la distribution des labels."
    exit 1
fi

echo "[5/7] Préparation des datasets pour le modèle LSTM..."
python src/model/prepare_lstm_dataset.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de la préparation du dataset LSTM."
    exit 1
fi

echo "[6/7] Création des splits train/val/test multi-tâches..."
python src/model/split_lstm_multitask.py
if [ $? -ne 0 ]; then
    echo "Erreur lors du fractionnement du dataset multi-tâches."
    exit 1
fi

echo "[7/7] Entraînement du modèle LSTM multi-tâches..."
python src/model/train_multitask_lstm.py
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'entraînement du modèle."
    exit 1
fi

echo "==== Pipeline terminé avec succès ===="
