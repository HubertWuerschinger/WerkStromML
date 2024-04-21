import streamlit as st
import pandas as pd
import h5py

# Laden des gespeicherten Regressionsmodells
model_file_path = "regression_model.h5"
loaded_model = None

def load_model():
    global loaded_model
    with h5py.File(model_file_path, 'r') as file:
        loaded_model = file['model'][:]

# Funktion zur Durchführung der Vorhersage
def predict(df):
    if loaded_model is not None:
        X = df[['Area Under Curve', 'Standard Deviation (Frequency)']]
        y_pred = loaded_model.predict(X)
        return y_pred
    else:
        st.error("Error: Model not loaded")

# Streamlit-Anwendung
def main():
    st.title("Regression Prediction App")
    st.sidebar.title("Upload CSV")

    # CSV-Datei hochladen
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Laden des Modells
        load_model()

        # Vorhersage durchführen
        y_pred = predict(df[['Area Under Curve', 'Standard Deviation (Frequency)']])
        
        # Anzeige der Vorhersagen
        st.subheader("Predictions:")
        st.write(y_pred)

if __name__ == "__main__":
    main()
