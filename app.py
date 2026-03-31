from pathlib import Path
import streamlit as st

from src.shared.connection import init_db
from src.models.production_pipeline import load_model


# PATH CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "src" / "models" / "xgb_model.joblib"
DB_PATH = BASE_DIR / "patients.db"

st.session_state.DB_PATH = DB_PATH


# APP CONFIGURATION

def configure_app():
    st.set_page_config(
        page_title="Obesity Risk Prediction System",
        layout="wide"
    )


def load_resources():
    if "model" not in st.session_state:
        st.session_state.model = load_model(MODEL_PATH)

    if "db_initialized" not in st.session_state:
        init_db(DB_PATH)
        st.session_state.db_initialized = True

    if "HOSPITAL_NAME" not in st.session_state:
        st.session_state.HOSPITAL_NAME = "Sistema de Avaliação Preventiva"

    if "LOGO_PATH" not in st.session_state:
        st.session_state.LOGO_PATH = "logo.png"

    if "PRIMARY_COLOR" not in st.session_state:
        st.session_state.PRIMARY_COLOR = (0, 168, 150)

    if "ACCENT_COLOR" not in st.session_state:
        st.session_state.ACCENT_COLOR = (0, 0, 0)


def apply_theme():
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #00A896;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def run_navigation():
    pages = [
        st.Page("src/pages/Home.py", title="Home"),
        st.Page("src/pages/Prever.py", title="Predict"),
        st.Page("src/pages/Historico.py", title="History"),
        st.Page("src/pages/Dashboard.py", title="EDA"),
    ]

    navigation = st.navigation(pages)
    navigation.run()


def main():
    configure_app()
    load_resources()
    apply_theme()
    run_navigation()


if __name__ == "__main__":
    main()