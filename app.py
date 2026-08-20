
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os

# --- Configuration ---
# Ensure Google Drive is mounted and the folder path is correct
folder_path = "/content/drive/MyDrive/SmartEnergy" # This should match where you saved your model/scalers
scaler_path = "/content/scaler.pkl" # This is where the overall scaler was saved
target_scaler_inverse_path = "/content/target_scaler_for_inverse.pkl" # This is where the inverse target scaler was saved

# --- Load Model and Scalers ---
loaded_lstm_model = None
loaded_scaler = None
target_scaler_for_inverse = None

try:
    # Load the saved LSTM model from Drive
    loaded_lstm_model = tf.keras.models.load_model(os.path.join(folder_path, "energy_lstm_model.keras"))
    # Load the feature scaler from /content
    loaded_scaler = joblib.load(scaler_path)
    # Load the target scaler for inverse transformation from /content
    target_scaler_for_inverse = joblib.load(target_scaler_inverse_path)
    print("Model and scalers loaded successfully!")
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    # Handle the case where loading fails, maybe disable prediction endpoint or return error

# Define input features and n_steps (must match training)
# Make sure these are consistent with how the model was trained
features_for_prediction = [
    'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows',
    'lag_1', 'lag_6', 'lag_144', 'lag_1008', 'TotalConsumption' # TotalConsumption is the target used during scaling
]
lstm_n_steps = 10 # This should match the 'n_steps' used during model training

# --- FastAPI App ---
app = FastAPI(title="Smart Energy Prediction API")

# Request body for prediction
class PredictionRequest(BaseModel):
    sequence: list[dict] # A list of dictionaries, each dict represents an observation

@app.get("/")
def home():
    return {"message": "Smart Energy Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": loaded_lstm_model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if loaded_lstm_model is None or loaded_scaler is None or target_scaler_for_inverse is None:
        raise HTTPException(status_code=500, detail="Model or scalers not loaded. Check server logs.")

    try:
        # Convert input sequence to DataFrame
        input_df = pd.DataFrame(request.sequence)

        # Ensure column order matches the training data for scaling
        # It's crucial that `features_for_prediction` includes the target column
        # as it was part of the original `df_scaled` for `scaler`.
        input_df = input_df[features_for_prediction]

        # Scale the input sequence
        scaled_input = loaded_scaler.transform(input_df)

        # Reshape for LSTM: (1, n_steps, n_features)
        # The input should have 'lstm_n_steps' (10) observations, each with 'len(features_for_prediction)' (10) features.
        if scaled_input.shape[0] != lstm_n_steps:
             raise ValueError(f"Input sequence must contain {lstm_n_steps} observations, but got {scaled_input.shape[0]}.")

        prediction_input = scaled_input.reshape(1, lstm_n_steps, len(features_for_prediction))

        # Make prediction
        scaled_prediction = loaded_lstm_model.predict(prediction_input)

        # Inverse transform the prediction
        # The target_scaler_for_inverse expects a 2D array, e.g., [[value]]
        predicted_total_consumption = target_scaler_for_inverse.inverse_transform(scaled_prediction)[0][0]

        return {"predicted_total_consumption": predicted_total_consumption}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
