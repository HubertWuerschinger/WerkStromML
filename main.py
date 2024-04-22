import streamlit as st
import pandas as pd
import h5py
from datetime import datetime
import matplotlib.pyplot as plt

# Laden des gespeicherten Regressionsmodells
model_file_path = "regression_model.h5"
loaded_model = None

def load_model():
    global loaded_model
    if loaded_model is None:
        with h5py.File(model_file_path, 'r') as file:
            model_weights = file['model_weights'][:]
            model_intercept = file['model_intercept'][()]
            loaded_model = {'weights': model_weights, 'intercept': model_intercept}

def predict(df):
    load_model()
    if loaded_model is not None:
        X = df[['Area Under Curve', 'Standard Deviation (Frequency)']]
        y_pred = X.dot(loaded_model['weights']) + loaded_model['intercept']
        return y_pred
    else:
        st.error("Fehler: Modell nicht geladen")

def create_chart(df):
    if not df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(df['Uhrzeit']), df['Werkzeugverschleiß'], marker='o', linestyle='-', color='b')
        plt.title('Werkzeugverschleiß')
        plt.xlabel('Zeit')
        plt.ylabel('Werkzeugverschleiß (µm)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

def display_predictions(df):
    st.subheader("Werkzeugverschleißmessung:")
    for _, row in df.iterrows():
        pred = row['Werkzeugverschleiß']
        color = color_scale(pred)
        st.markdown(f"<div style='background-color: {color}; padding: 8px; border-radius: 5px;'></div>", unsafe_allow_html=True)
        progress_bar_value = 100 - int(100 * (pred / 300))
        st.progress(progress_bar_value)
        st.write(f"Werkzeugkapazität: {progress_bar_value}%")

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

def main():
    st.title("WerkStromML")
    st.sidebar.title("CSV hochladen und Arbeitsdaten eingeben")

    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()

    werkzeugtyp = st.sidebar.text_input("Werkzeugtyp", "Typ XYZ")
    einsatzdauer_min = st.sidebar.number_input("Einsatzdauer (Minuten)", value=60)
    material = st.sidebar.text_input("Bearbeitetes Material", "Material ABC")

    uploaded_file = st.sidebar.file_uploader("CSV hochladen", type=["csv"])
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        if not new_data.empty:
            y_pred = predict(new_data[['Area Under Curve', 'Standard Deviation (Frequency)']])
            new_data['Uhrzeit'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_data['Werkzeugtyp'] = werkzeugtyp
            new_data['Einsatzdauer'] = einsatzdauer_min
            new_data['Bearbeitetes Material'] = material
            new_data['Werkzeugverschleiß'] = y_pred
            new_data['Werkzeugkapazität'] = [100 - int(100 * (pred / 300)) for pred in y_pred]
            display_predictions(new_data)

            if st.sidebar.button("Daten speichern"):
                try:
                    existing_data = pd.read_csv('Arbeitsdaten.csv')
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    existing_data = pd.DataFrame()

                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data.to_csv('Arbeitsdaten.csv', index=False)
                st.session_state['data'] = combined_data
                st.sidebar.success("Daten erfolgreich gespeichert!")

    # Diagramm immer anzeigen, wenn Daten vorhanden sind
    if not st.session_state['data'].empty:
        st.subheader("Werkzeugverschleiß-Diagramm")
        create_chart(st.session_state['data'])

if __name__ == "__main__":
    main()
