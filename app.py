from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Obesity Risk Prediction System",
    layout="wide"
)

from src.shared.connection import init_db
from src.models.production_pipeline import load_model


# PATH CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "xgb_model.joblib"

DB_PATH = BASE_DIR / "patients.db"

st.session_state.DB_PATH = DB_PATH


# APP CONFIGURATION

def configure_app():
    pass  # já configurado no topo


def load_resources():
    if not MODEL_PATH.exists():
        st.error("Nenhum modelo encontrado.")
        st.stop()

    if "model" not in st.session_state:
        st.session_state.model = load_model(str(MODEL_PATH))

    if "db_initialized" not in st.session_state:
        init_db(DB_PATH)
        st.session_state.db_initialized = True
        
    if "nome_hospital" not in st.session_state:
        st.session_state.nome_hospital = "Sistema de Avaliação Preventiva"

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


def main():
    configure_app()
    load_resources()
    apply_theme()


if __name__ == "__main__":
    main()