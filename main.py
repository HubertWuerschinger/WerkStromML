import streamlit as st
import pandas as pd
import h5py
import json
from datetime import datetime

# Laden des gespeicherten Regressionsmodells
model_file_path = "regression_model.h5"
loaded_model = None

def load_model():
    global loaded_model
    with h5py.File(model_file_path, 'r') as file:
        model_weights = file['model_weights'][:]
        model_intercept = file['model_intercept'][()]
        loaded_model = {'weights': model_weights, 'intercept': model_intercept}

# Funktion zur Durchführung der Vorhersage
def predict(df):
    if loaded_model is not None:
        X = df[['Area Under Curve', 'Standard Deviation (Frequency)']]
        y_pred = X.dot(loaded_model['weights']) + loaded_model['intercept']
        return y_pred
    else:
        st.error("Fehler: Modell nicht geladen")

# Farbskala für die Prognosewerte
def color_scale(value):
    if value <= 50:
        return 'green'
    elif value <= 100:
        return 'lightgreen'
    elif value <= 150:
        return 'yellow'
    elif value <= 200:
        return 'orange'
    elif value <= 250:
        return 'red'
    else:
        return 'darkred'

# Farbskala für den Balken basierend auf dem Prognosewert
def slider_color_scale(value):
    if value <= 50:
        return '#00FF00'  # Grün
    elif value <= 100:
        return '#7FFF00'  # Grasgrün
    elif value <= 150:
        return '#FFFF00'  # Gelb
    elif value <= 200:
        return '#FFA500'  # Orange
    elif value <= 250:
        return '#FF0000'  # Rot
    else:
        return '#8B0000'  # Dunkelrot

# Streamlit-Anwendung
def main():
    st.title("WerkStromML")
    st.sidebar.title("CSV hochladen und Arbeitsdaten eingeben")

    # CSV-Datei hochladen
    uploaded_file = st.sidebar.file_uploader("CSV hochladen", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Laden des Modells
        load_model()

        # Benutzereingaben für Arbeitsdaten
        werkzeugtyp = st.sidebar.text_input("Werkzeugtyp", "Typ XYZ")
        einsatzdauer_min = st.sidebar.number_input("Einsatzdauer (Minuten)", value=60)
        material = st.sidebar.text_input("Bearbeitetes Material", "Material ABC")

        # Vorhersage durchführen
        y_pred = predict(df[['Area Under Curve', 'Standard Deviation (Frequency)']])
        
        # Anzeige der Vorhersagen mit Farbskala und Balken
        st.subheader("Werkzeugverschleißmessung:")
        for pred in y_pred:
            st.write(f"Modellprognose: {int(pred)} µm", unsafe_allow_html=True, key=str(int(pred)))

            st.write(
                f"<div style='background-color: {color_scale(pred)}; padding: 8px; border-radius: 5px;'></div>",
                unsafe_allow_html=True
            )
            
            # Anzeige des Balkens mit variabler Länge basierend auf dem Verschleißgrad
            #st.subheader("Werkzeugverschleiß:")
            if pred <= 0:
                tool_capacity = 100
            elif pred >= 300:
                tool_capacity = 0
            else:
                tool_capacity = 100 - int(100 * (pred / 300))  # Umgekehrter Verschleißgrad
            #st.progress(tool_capacity)

            # Anzeige der Prognose in Prozent
            #st.write(f"Werkzeugvebrauch: {tool_capacity}%")

            st.subheader("Verschleißgrad:")
            progress_bar_value = int(100 * (pred / 300))  # Umrechnung von µm in Prozent
            st.progress(progress_bar_value)

            # Anzeige der Prognose in Prozent
            st.write(f"Werkzeugvebrauch: {progress_bar_value}%")



        # Aktuelles Datum und Uhrzeit
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Arbeitsdaten hinzufügen
        new_data = pd.DataFrame({
            'Aktuelles Datum': [current_date],
            'Uhrzeit': [current_time],
            'Werkzeugtyp': [werkzeugtyp],
            'Einsatzdauer': [einsatzdauer_min],
            'Bearbeitetes Material': [material]
        })

        df = pd.concat([df, new_data], ignore_index=True)


        # Speichern der Daten in einer JSON-Datei
        if st.sidebar.button("Daten speichern"):
            df.to_json('Arbeitsdaten.txt', orient='records', mode='a')
            st.sidebar.success("Daten erfolgreich gespeichert!")

if __name__ == "__main__":
    main()
