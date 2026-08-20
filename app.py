import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

BASE = Path(__file__).parent

MODEL_PATH = BASE / "energy_lstm_model.keras"
SCALER_PATH = BASE / "scaler.pkl"
DATA_PATH = BASE / "processed_energy_data.csv"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


st.title("⚡ Smart Energy Consumption Predictor")
st.caption("LSTM-based electricity consumption forecasting")

if not MODEL_PATH.exists():
    st.error("❌ energy_lstm_model.keras not found.")
    st.stop()

model = load_model()
scaler = load_scaler()
df = load_data()


# -----------------------------
# Model Information
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Model", "LSTM")
col2.metric("Input Shape", str(model.input_shape))
col3.metric("Output Shape", str(model.output_shape))

st.divider()


# -----------------------------
# Prediction
# -----------------------------

st.subheader("🔮 Energy Prediction")

st.info(
    "The trained LSTM expects 10 time steps × 10 features."
)

default_data = pd.DataFrame(
    np.zeros((10, 10)),
    columns=[f"Feature {i+1}" for i in range(10)]
)

values = st.data_editor(
    default_data,
    use_container_width=True,
    num_rows="fixed"
)


if st.button("🔮 Predict Energy Consumption", type="primary"):

    try:

        x = values.to_numpy(dtype=np.float32)

        if x.shape != (10, 10):
            st.error(
                f"Expected input shape (10, 10), "
                f"but received {x.shape}"
            )
            st.stop()

        x = x.reshape(1, 10, 10)

        prediction = model.predict(
            x,
            verbose=0
        )

        result = float(prediction[0][0])

        st.success("Prediction completed successfully!")

        st.metric(
            "Model Output",
            f"{result:.4f}"
        )

        if scaler is None:
            st.warning(
                "scaler.pkl was not found. "
                "The value above is the model output and "
                "has not been converted back to the original "
                "energy-consumption unit."
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )


# -----------------------------
# Historical Data
# -----------------------------

if df is not None:

    st.divider()

    st.subheader("📊 Historical Energy Data")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Features",
        len(df.columns)
    )

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )
