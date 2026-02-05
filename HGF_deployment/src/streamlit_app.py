import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "AbdramaneB/used-cars-price-prediction"
MODEL_FILENAME = "best_price_model_v2.joblib"

SEGMENT_MAP = {
    "luxury segment": 0,
    "non-luxury segment": 1,
}
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
    kilometers_driven = st.number_input("Kilometers_Driven [km] (8350-262000)", min_value=0.0, value=170000.0, step=5000.0)
    mileage = st.number_input("Mileage [km/l] (0-28.4)", min_value=0.0, value=15.0, step=1.0)
    engine = st.number_input("Engine [cc] (624-5461)", min_value=0.0, value=3000.0, step=50.0)
    power = st.number_input("Power [BHP] (35-500)", min_value=35, value=270.0, step=20)
    seats = st.number_input("Seats (2-8)", min_value=1, max_value=10, value=5, step=1)

    input_df = pd.DataFrame([{
        "Segment": 0 if segment == "luxury segment" else 1,
        "Kilometers_Driven": float(kilometers_driven),
        "Mileage": float(mileage),
        "Engine": float(engine),
        "Power": float(power),
        "Seats": int(seats),
    }])

    if st.button("Predict Price"):
        model = load_model()
        # Align columns to the training schema (prevents silent feature-order drift)
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=list(model.feature_names_in_))
        pred = model.predict(input_df)
        st.success(f"Estimated price: Lakhs {float(pred[0]):,.2f}")

if __name__ == "__main__":
    main()
