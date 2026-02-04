import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "AbdramaneB/used-cars-price-prediction"
MODEL_FILENAME = "best_price_model_v2.joblib"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    MODEL_REPO_ID = "AbdramaneB/used-cars-price-model"
    MODEL_FILENAME = "best_price_model_v2.joblib"

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        filename=MODEL_FILENAME,
    )

    return joblib.load(model_path)

def main():
    st.set_page_config(page_title="Used Cars Price Prediction", layout="centered")
    st.title("Used Cars Price Prediction App")

    segment = st.selectbox("Segment", ["non-luxury segment", "luxury segment"], index=1)
    kilometers_driven = st.number_input("Kilometers_Driven", min_value=0.0, value=120000.0, step=5000.0)
    mileage = st.number_input("Mileage [km/l]", min_value=0.0, value=15.0, step=1.0)
    engine = st.number_input("Engine [cc]", min_value=0.0, value=1200.0, step=50.0)
    power = st.number_input("Power [BHP]", min_value=0.0, value=90.0, step=5.0)
    seats = st.number_input("Seats", min_value=1, max_value=10, value=5, step=1)

    input_df = pd.DataFrame([{
        "Segment": 0 if segment == "luxury segment" else 1,
        "Kilometers_Driven": kilometers_driven,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
    }])

    if st.button("Predict Price"):
        model = load_model()
        pred = model.predict(input_df)
        st.success(f"Estimated price: Lakhs {float(pred[0]):,.2f}")

if __name__ == "__main__":
    main()
