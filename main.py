import streamlit as st
import pandas as pd
import h5py
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

# Streamlit-Anwendung
def main():
    st.title("Regression Prediction App")
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
        
        # Anzeige der Vorhersagen
        st.subheader("Vorhersagen:")
        st.write(y_pred)

        # Aktuelles Datum und Uhrzeit
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Arbeitsdaten hinzufügen
        df['Aktuelles Datum'] = current_date
        df['Uhrzeit'] = current_time
        df['Werkzeugtyp'] = werkzeugtyp
        df['Einsatzdauer'] = einsatzdauer_min
        df['Bearbeitetes Material'] = material

        # Speichern der Daten in einer CSV-Datei
        df.to_csv('Arbeitsdaten.csv', index=False)

if __name__ == "__main__":
    main()
