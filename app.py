import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "energy_lstm_model.keras"
DATA_PATH = BASE / "processed_energy_data.csv"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

st.title("⚡ Smart Energy Consumption Predictor")
st.caption("LSTM-based electricity consumption forecasting")

if not MODEL_PATH.exists():
    st.error("energy_lstm_model.keras not found.")
    st.stop()

model = load_model()
df = load_data()

col1, col2, col3 = st.columns(3)

col1.metric("Model", "LSTM")
col2.metric("Input Shape", str(model.input_shape))
col3.metric("Output Shape", str(model.output_shape))

st.divider()

st.subheader("Energy Prediction")

st.info(
    "The current LSTM expects 10 time steps × 10 features. "
    "Enter the prepared/scaled 10×10 sequence used during training."
)

values = st.data_editor(
    pd.DataFrame(
        np.zeros((10, 10)),
        columns=[f"Feature {i+1}" for i in range(10)]
    ),
    use_container_width=True,
    num_rows="fixed"
)

if st.button("🔮 Predict Energy Consumption", type="primary"):

    x = values.to_numpy(dtype=np.float32)

    if x.shape != (10, 10):
        st.error(f"Expected (10, 10), received {x.shape}")
        st.stop()

    prediction = model.predict(
        x.reshape(1, 10, 10),
        verbose=0
    )

    result = float(prediction[0][0])

    st.success("Prediction completed!")

    st.metric(
        "Predicted Consumption",
        f"{result:.4f}"
    )

    st.warning(
        "This is currently the model's scaled output. "
        "The final version should apply your exact scaler and "
        "inverse-transform the prediction."
    )

if df is not None:

    st.divider()

    st.subheader("Historical Energy Data")

    st.write(
        f"Rows: {len(df):,} | Columns: {len(df.columns)}"
    )

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )
