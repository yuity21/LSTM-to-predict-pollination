import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import calendar
from PIL import Image
import base64


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction de Pollinisation 2025", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    /* Styles généraux */
    .main {
        background-color: #f0f5fa;
        padding: 20px;
    }
    h1 {
        color: #1A5276;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        padding: 20px 0 10px 0;
        text-align: center;
        border-bottom: 2px solid #3498DB;
        margin-bottom: 20px;
    }
    h2, h3 {
        color: #2874A6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 12px 20px;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1A5276;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Carte de culture */
    .culture-card {
        background-color: #E8F6FC;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 5px solid #3498DB;
    }
    
    .calendar-day {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Légende */
    .legend-box {
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info Box */
    .info-box {
        background-color: #D6EAF8;
        border-left: 5px solid #3498DB;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1wrcr25 {
        background-color: #D6EAF8;
    }
    
    /* Custom calendar styling */
    .calendar-header {
        font-weight: bold;
        text-align: center;
        padding: 8px 5px;
        background-color: #2874A6;
        color: white;
        border-radius: 5px;
        margin-bottom: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .activity-high {
        background-color: #7fff7f;
        color: #1e441e;
        border: 2px solid #2ECC71;
    }
    
    .activity-medium {
        background-color: #ffff7f;
        color: #4d4d00;
        border: 2px solid #F1C40F;
    }
    
    .activity-low {
        background-color: #ff7f7f;
        color: #661a1a;
        border: 2px solid #E74C3C;
    }
    
    .inactive-day {
        background-color: #ECF0F1;
        color: #7F8C8D;
        border: 1px solid #BDC3C7;
    }
    
    .page-header {
        background: linear-gradient(135deg, #2980B9, #3498DB);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .page-header h1 {
        color: white;
        border-bottom: none;
        margin-bottom: 10px;
        padding-bottom: 5px;
    }
    
    .page-header p {
        color: #E8F6FC;
        font-size: 18px;
        margin-bottom: 0;
    }
    
    .category-header {
        background-color: #2C3E50;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 20px 0 15px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .tab-container {
        border: 1px solid #BDC3C7;
        border-radius: 10px;
        padding: 15px;
        background-color: black;
    }
    
    .status-active {
        background-color: #D5F5E3;
        color: #186A3B;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: 500;
        display: inline-block;
        border: 1px solid #2ECC71;
    }
    
    .status-inactive {
        background-color: #FADBD8;
        color: #943126;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: 500;
        display: inline-block;
        border: 1px solid #E74C3C;
    }
    
    @media (max-width: 768px) {
        .calendar-day {
            width: 30px;
            height: 30px;
            font-size: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Variables globales
cultures = ["colza", "tournesol", "lavande", "pommiers"]
cultures_fr = {
    "colza": "Colza",
    "tournesol": "Tournesol",
    "lavande": "Lavande", 
    "pommiers": "Pommiers"
}
mois_floraison = {
    "colza": [3, 4, 5],       # mars à mai
    "tournesol": [6, 7, 8],   # juin à août
    "lavande": [6, 7],        # juin à juillet
    "pommiers": [4, 5],       # avril à mai
}
villes = {
    "Pau": {"lat": 43.2951, "lon": -0.3708},
    "Orleans": {"lat": 47.9025, "lon": 1.9090},
    "Avignon": {"lat": 43.9493, "lon": 4.8055},
}
variables_meteo = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
    "sunshine_duration"
]
activite_labels = ["Basse", "Moyenne", "Haute"]
couleurs_activite = {
    "Basse": "#FADBD8",    # Rouge pastel
    "Moyenne": "#FCF3CF",  # Jaune pastel
    "Haute": "#D5F5E3",    # Vert pastel
}
icones_cultures = {
    "colza": "🌱",
    "tournesol": "🌻",
    "lavande": "💜",
    "pommiers": "🍎"
}
couleurs_cultures = {
    "colza": "#82E0AA",      # Vert
    "tournesol": "#F4D03F",  # Jaune
    "lavande": "#BB8FCE",    # Violet
    "pommiers": "#EC7063"    # Rouge-rose
}

# En-tête de l'application
st.markdown("""
<div class="page-header">
    <h1>Prédiction de l'Activité de Pollinisation</h1>
    <p>
        Visualisez l'activité des pollinisateurs basée sur les données météorologiques et un modèle LSTM multi-tâches
    </p>
</div>
""", unsafe_allow_html=True)

# Fonction pour déterminer si les données sont disponibles pour une date donnée
def donnees_disponibles(annee, mois):
    # Obtenir la date actuelle réelle
    date_actuelle = datetime.now()
    
    # Les données ne sont pas disponibles pour les dates futures (mois et années)
    if annee > date_actuelle.year:
        return False
    elif annee == date_actuelle.year and mois > date_actuelle.month:
        return False
    # Les données sont disponibles pour les dates passées et le mois courant
    return True

# Fonction pour charger le modèle LSTM
@st.cache_resource
def charger_modele():
    try:
        modele_path = "models/lstm_models_multitask/best_multitask_lstm.keras"
        if not os.path.exists(modele_path):
            st.error(f"Le modèle n'existe pas à l'emplacement {modele_path}")
            return None
        model = tf.keras.models.load_model(
            modele_path,
            compile=False  # Pas besoin de recompiler pour la prédiction
        )
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Fonction pour télécharger les données météo pour l'année sélectionnée
def telecharger_donnees_meteo(ville, lat, lon, annee):
    """
    Récupère :
      - l'historique via l'API archive jusqu'à hier inclus
      - les prévisions via l'API forecast pour aujourd'hui + 6 jours
    Puis fait un forward/back-fill des NaN météo.
    """
    date_actuelle = datetime.now()

    # Pas de données réelles pour les années futures
    if annee > date_actuelle.year:
        st.error(f"Aucune donnée météo n'est disponible pour {annee}.")
        return None

    mets = ",".join(variables_meteo)

    # Archive jusqu'à hier
    start_hist = f"{annee}-01-01"
    end_hist   = (date_actuelle - timedelta(days=1)).strftime("%Y-%m-%d")
    url_hist = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_hist}&end_date={end_hist}"
        f"&daily={mets}&timezone=auto"
    )
    resp_h = requests.get(url_hist, timeout=10)
    resp_h.raise_for_status()
    df_hist = pd.DataFrame(resp_h.json()["daily"])
    df_hist["time"] = pd.to_datetime(df_hist["time"])

    # 2) Forecast pour aujourd'hui et J+6
    fc_start = date_actuelle.strftime("%Y-%m-%d")
    fc_end   = (date_actuelle + timedelta(days=6)).strftime("%Y-%m-%d")
    url_fc = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={fc_start}&end_date={fc_end}"
        f"&daily={mets}&timezone=auto"
    )
    resp_f = requests.get(url_fc, timeout=10)
    resp_f.raise_for_status()
    df_fc = pd.DataFrame(resp_f.json()["daily"])
    df_fc["time"] = pd.to_datetime(df_fc["time"])

    # Concaténer historiques + prévisions
    df = pd.concat([df_hist, df_fc], ignore_index=True)

    # 3) Forward/back-fill pour combler tous les NaN météo
    df = df.sort_values("time").reset_index(drop=True)
    for col in variables_meteo:
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

    # Colonnes communes et indicateurs de floraison
    df["city"]  = ville
    df["month"] = df["time"].dt.month
    for culture, mois in mois_floraison.items():
        df[f"is_flowering_{culture}"] = df["month"].isin(mois).astype(int)
    df["donnees_disponibles"] = True

    return df

# Fonction pour préparer les données pour le modèle LSTM
def preparer_donnees_lstm(df, sequence_length=10):
    """
    Prépare les séquences d'entrée pour le modèle LSTM multi-tâches.
    Chaque séquence a shape (sequence_length, 9) : 
      - 5 variables météo
      - 4 indicateurs de floraison (colza, tournesol, lavande, pommiers)
    Retourne :
      - all_sequences : liste de 4 arrays de shape (n_seq, sequence_length, 9),
                        un array par culture
      - dates         : liste des datetime associées à chaque séquence
      - disponibles   : liste de bool (True si données réelles dispo pour la date)
    """
    if df is None or len(df) < sequence_length:
        return None, None, None

    # 1) Définir les colonnes de features (5 météo + 4 floraisons)
    feature_cols = variables_meteo + [f"is_flowering_{c}" for c in cultures]

    # 2) Normalisation min-max de chaque colonne de features
    df_norm = df.copy()
    for col in feature_cols:
        if col in df_norm.columns:
            vmin, vmax = df_norm[col].min(), df_norm[col].max()
            df_norm[col] = 0.0 if vmax == vmin else (df_norm[col] - vmin) / (vmax - vmin)

    # 3) Construction des séquences glissantes
    X_seqs = []
    dates   = []
    disponibles = []
    last_idx = len(df_norm) - sequence_length + 1
    for i in range(last_idx):
        window = df_norm.iloc[i : i + sequence_length]
        X_seqs.append(window[feature_cols].values)
        # date associée = dernier jour de la fenêtre
        dates.append(window["time"].iloc[-1])
        disponibles.append(True)

    X_array = np.array(X_seqs)  # shape (n_seq, sequence_length, 9)

    # Réplication de la même entrée pour chaque culture
    all_sequences = [X_array for _ in cultures]

    return all_sequences, dates, disponibles


# Fonction pour prédire l'activité de pollinisation
def predire_activite(model, all_sequences, disponibles):
    if model is None:
        return None
    
    # Faire les prédictions pour chaque culture séparément
    all_predictions = []
    for i, X_sequences in enumerate(all_sequences):
        with st.spinner(f"🧠 Prédiction pour {cultures_fr[cultures[i]]}..."):
            predictions_culture = model.predict(X_sequences, verbose=0)  # Désactiver la verbosité
            if isinstance(predictions_culture, list):
                # Si le modèle renvoie une liste de prédictions, prendre celle correspondant à la culture actuelle
                pred = predictions_culture[i]
            else:
                # Sinon, utiliser directement les prédictions
                pred = predictions_culture
            all_predictions.append(pred)
    
    # Convertir les prédictions en classes
    resultats = {}
    for i, culture in enumerate(cultures):
        # Prendre l'index de la classe avec la plus haute probabilité
        pred_class = np.argmax(all_predictions[i], axis=1)
        resultats[culture] = pred_class
    
    return resultats, disponibles

# Fonction pour afficher le calendrier avec les prédictions
def afficher_calendrier(dates, predictions, disponibles, ville, mois_selectionne, annee_selectionnee):
    if predictions is None or len(predictions) == 0:
        st.warning(f"Aucune prédiction disponible pour {calendar.month_name[mois_selectionne]} {annee_selectionnee}.")
        return
    
    # Créer un DataFrame avec les dates et les prédictions
    df_pred = pd.DataFrame({
        "date": dates,
        "disponible": disponibles,
        **{culture: predictions[culture] for culture in cultures}
    })
    
    # Convertir les dates en objets datetime
    df_pred["date"] = pd.to_datetime(df_pred["date"])
    
    # Filtrer par mois
    df_mois = df_pred[df_pred["date"].dt.month == mois_selectionne]
    
    if df_mois.empty:
        # Vérifier si c'est parce que le mois est dans le futur
        date_actuelle = datetime.now()
        if (annee_selectionnee > date_actuelle.year or 
            (annee_selectionnee == date_actuelle.year and mois_selectionne > date_actuelle.month)):
            st.warning(f"""
            ⚠️ **Les données météorologiques pour {calendar.month_name[mois_selectionne]} {annee_selectionnee} ne sont pas encore disponibles.**
            
            Open Meteo ne dispose pas de données pour cette période future. Les prédictions ne peuvent pas être générées.
            Pour obtenir des prédictions, veuillez sélectionner une période passée ou actuelle.
            """)
        else:
            st.warning(f"Aucune donnée disponible pour le mois {mois_selectionne}.")
        return
    
    # Obtenir le premier jour du mois et le nombre de jours
    annee = df_mois["date"].dt.year.iloc[0]
    premier_jour_semaine = datetime(annee, mois_selectionne, 1).weekday()
    nb_jours = calendar.monthrange(annee, mois_selectionne)[1]
    
    # Créer un DataFrame pour le calendrier (6 semaines x 7 jours)
    jours_calendrier = []
    
    jour_actuel = 1
    for semaine in range(6):
        for jour_semaine in range(7):
            if semaine == 0 and jour_semaine < premier_jour_semaine:
                # Cases vides avant le premier jour du mois
                jours_calendrier.append({
                    "semaine": semaine,
                    "jour_semaine": jour_semaine,
                    "jour": None,
                    "date": None
                })
            elif jour_actuel <= nb_jours:
                date_jour = datetime(annee, mois_selectionne, jour_actuel)
                jours_calendrier.append({
                    "semaine": semaine,
                    "jour_semaine": jour_semaine,
                    "jour": jour_actuel,
                    "date": date_jour
                })
                jour_actuel += 1
            else:
                # Cases vides après le dernier jour du mois
                jours_calendrier.append({
                    "semaine": semaine,
                    "jour_semaine": jour_semaine,
                    "jour": None,
                    "date": None
                })
    
    df_calendrier = pd.DataFrame(jours_calendrier)
    
    # Fusionner les données de prédiction avec le calendrier
    df_calendrier_merge = pd.merge(
        df_calendrier, 
        df_mois, 
        left_on="date", 
        right_on="date", 
        how="left"
    )
    
    # Nom complet du mois sélectionné
    nom_mois = calendar.month_name[mois_selectionne]
    
    # En-tête avec ville et mois
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #3498DB, #2874A6); padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
        <h2 style='margin-bottom: 10px; color: white;'>🗓️ Calendrier de Pollinisation - {nom_mois} {annee_selectionnee}</h2>
        <h3 style='color: #E8F6FC; margin: 0;'>📍 {ville}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Information sur la prédiction
    st.markdown("""
    <div class='info-box'>
        <h4 style='margin-top: 0; color: #1A5276;'>ℹ️ À propos des prédictions</h4>
        <p style='color: #2C3E50; margin-bottom: 0;'>
            Les prédictions sont basées sur des données météorologiques réelles d'Open Meteo et montrent la probabilité 
            d'activité des pollinisateurs pour chaque culture. Les zones de floraison spécifiques à chaque culture 
            sont prises en compte dans le modèle.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Légende commune pour tous les calendriers
    st.markdown("<div class='category-header'>🔍 Légende</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='legend-box' style='background-color: {couleurs_activite["Basse"]}; color: #943126; border: 2px solid #E74C3C;'>
            ⬇️ Activité Basse
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='legend-box' style='background-color: {couleurs_activite["Moyenne"]}; color: #7D6608; border: 2px solid #F1C40F;'>
            ⚡ Activité Moyenne
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='legend-box' style='background-color: {couleurs_activite["Haute"]}; color: #186A3B; border: 2px solid #2ECC71;'>
            ⬆️ Activité Haute
        </div>
        """, unsafe_allow_html=True)
    
    # Créer une visualisation pour chaque culture avec fond blanc
    st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
    tabs = st.tabs([f"{icones_cultures[culture]} {cultures_fr[culture]}" for culture in cultures])
    
    for i, culture in enumerate(cultures):
        with tabs[i]:
            st.markdown(f"""
            <div class='culture-card' style='border-left: 5px solid {couleurs_cultures[culture]};'>
                <h3 style='color: #1A5276; margin-top: 0;'>{icones_cultures[culture]} {cultures_fr[culture]}</h3>
                <p style='color: #2C3E50;'>Période de floraison: {', '.join([calendar.month_name[m] for m in mois_floraison[culture]])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Affichage du calendrier
            jours_semaine = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
            
            # Afficher les entêtes des jours
            cols_header = st.columns(7)
            for j, jour in enumerate(jours_semaine):
                with cols_header[j]:
                    st.markdown(f"""
                    <div class='calendar-header'>
                        {jour}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Afficher le calendrier
            for semaine in range(6):
                cols = st.columns(7)
                for jour_semaine in range(7):
                    with cols[jour_semaine]:
                        idx = df_calendrier_merge[
                            (df_calendrier_merge["semaine"] == semaine) & 
                            (df_calendrier_merge["jour_semaine"] == jour_semaine)
                        ].index
                        
                        if len(idx) > 0:
                            jour = df_calendrier_merge.loc[idx[0], "jour"]
                            if jour is not None and not pd.isna(jour):
                                # Vérifier si on a une prédiction pour ce jour
                                if idx[0] < len(df_calendrier_merge) and culture in df_calendrier_merge.columns:
                                    pred = df_calendrier_merge.loc[idx[0], culture]
                                    if not pd.isna(pred):
                                        activity_class = ["activity-low", "activity-medium", "activity-high"][int(pred)]
                                        st.markdown(f"""
                                        <div class='calendar-day {activity_class}'>
                                            {int(jour)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Aucune prédiction disponible pour ce jour
                                        st.markdown(f"""
                                        <div class='calendar-day inactive-day'>
                                            {int(jour)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    # Aucune prédiction disponible pour ce jour
                                    st.markdown(f"""
                                    <div class='calendar-day inactive-day'>
                                        {int(jour)}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("")
                        else:
                            st.markdown("")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Fonction principale
def main():
    # Barre latérale pour les contrôles
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0; background: linear-gradient(135deg, #2980B9, #3498DB); border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>🐝 Contrôles</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Section "À propos" dans la barre latérale
    with st.sidebar.expander("ℹ️ À propos de l'application"):
        st.markdown("""
        <div style='color: #fff0fe;'>
        Cette application permet de visualiser les prédictions d'activité de pollinisation 
        pour différentes cultures, basées sur des données météorologiques d'Open Meteo et un modèle LSTM multi-tâches.
        
        <b>Comment ça marche :</b>
        <ol>
            <li>Sélectionnez une ville</li>
            <li>Choisissez une année</li>
            <li>Choisissez un mois</li>
            <li>Cliquez sur "Générer les prédictions"</li>
        </ol>
        
        Le modèle utilise les données météorologiques réelles et des périodes de floraison
        pour prédire l'activité des pollinisateurs.
        
        <b>Note :</b>  Pour les jours futurs (jusqu’à 6 jours), les données météo de prévision sont utilisées.
        </div>
        """, unsafe_allow_html=True)
    
    # Sélection de la ville
    ville_selectionnee = st.sidebar.selectbox(
        "🏙️ Choisir une ville",
        list(villes.keys())
    )
    
    # Sélection de l'année
    date_actuelle = datetime.now()
    annee_actuelle = date_actuelle.year
    annee_selectionnee = st.sidebar.select_slider(
        "📆 Choisir une année",
        options=list(range(2017, annee_actuelle + 1)),
        value=annee_actuelle
    )
    
    # Sélection du mois avec un slider
    mois_selectionne = st.sidebar.select_slider(
        "📅 Choisir un mois",
        options=list(range(1, 13)),
        format_func=lambda m: calendar.month_name[m],
        value=min(date_actuelle.month, 5)  # Valeur par défaut: mois actuel ou mai
    )
    
    # Vérification si les données sont disponibles pour la période sélectionnée
    donnees_dispo = donnees_disponibles(annee_selectionnee, mois_selectionne)
    if not donnees_dispo:
        st.sidebar.warning(f"""
        ⚠️ **Attention** : Les données météorologiques pour {calendar.month_name[mois_selectionne]} {annee_selectionnee} 
        ne sont pas encore disponibles dans Open Meteo. Aucune prédiction ne pourra être générée pour cette période.
        """)
    
    # Affichage des cultures actives pour le mois sélectionné
    st.sidebar.markdown("<div class='category-header'>🌱 Cultures en floraison ce mois-ci</div>", unsafe_allow_html=True)
    for culture in cultures:
        est_en_floraison = mois_selectionne in mois_floraison[culture]
        status_class = "status-active" if est_en_floraison else "status-inactive"
        status_text = "En floraison" if est_en_floraison else "Pas en floraison"
        
        st.sidebar.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 10px; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div style='margin-right: 10px; font-size: 24px;'>{icones_cultures[culture]}</div>
            <div style='flex-grow: 1;'>
                <div style='font-weight: 500; color: #2C3E50;'>{cultures_fr[culture]}</div>
                <div class='{status_class}'>{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton pour lancer la prédiction
    if st.sidebar.button("🚀 Générer les prédictions"):
        # Vérifier d'abord si les données sont disponibles
        if not donnees_disponibles(annee_selectionnee, mois_selectionne):
            st.error(f"""
            ⚠️ **Les données météorologiques pour {calendar.month_name[mois_selectionne]} {annee_selectionnee} ne sont pas disponibles.**
            
            Open Meteo ne dispose pas de données pour cette période future. Les prédictions ne peuvent pas être générées.
            Pour obtenir des prédictions, veuillez sélectionner une période passée ou actuelle.
            """)
        else:
            with st.spinner("Chargement du modèle..."):
                # Charger le modèle LSTM
                model = charger_modele()
                
                if model:
                    # Coordonnées de la ville sélectionnée
                    lat = villes[ville_selectionnee]["lat"]
                    lon = villes[ville_selectionnee]["lon"]
                    
                    # Télécharger les données météo
                    df_meteo = telecharger_donnees_meteo(ville_selectionnee, lat, lon, annee_selectionnee)
                    
                    if df_meteo is not None:
                        # Préparer les données pour le LSTM
                        with st.spinner("⚙️ Préparation des données..."):
                            all_sequences, dates, disponibles = preparer_donnees_lstm(df_meteo)
                        
                        if all_sequences is not None:
                            # Faire les prédictions
                            predictions, disponibles_final = predire_activite(model, all_sequences, disponibles)
                            
                            # Afficher le calendrier avec les prédictions
                            afficher_calendrier(dates, predictions, disponibles_final, ville_selectionnee, mois_selectionne, annee_selectionnee)
                        else:
                            st.error("Impossible de préparer les données pour le modèle. Veuillez vérifier les dates sélectionnées.")
                    else:
                        st.error("Impossible de télécharger les données météo. Veuillez réessayer.")
                else:
                    st.error("Impossible de charger le modèle LSTM. Veuillez vérifier que le fichier existe.")
    
    # Pied de page
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='background-color: #2C3E50; color: white; padding: 15px; border-radius: 10px; text-align: center;'>
        <p style='margin-bottom: 5px;'>Développé avec Streamlit et TensorFlow par Ryan Shams Mouktar Houssen et Hayat Meghlat M1 TI UPPA 2025</p>
        <p style='margin: 0;'>Données météo: Open Meteo API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 