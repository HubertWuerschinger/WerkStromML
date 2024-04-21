import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Laden des gespeicherten Regressionsmodells
model_file_path = "regression_model.pkl"
loaded_model = joblib.load(model_file_path)

# Funktion zur Durchführung der Vorhersage
def predict(df):
    X = df[['Mean Amplitude', 'Standard Deviation (Amplitude)']]
    y_pred = loaded_model.predict(X)
    return y_pred

# Streamlit-Anwendung
def main():
    st.title("Regression Prediction App")
    st.sidebar.title("Upload CSV")

    # CSV-Datei hochladen
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Vorhersage durchführen
        y_pred = predict(df)
        
        # Anzeige der Vorhersagen
        st.subheader("Predictions:")
        st.write(y_pred)

if __name__ == "__main__":
    main()
