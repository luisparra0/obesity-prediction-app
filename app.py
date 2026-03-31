import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Obesity Risk Prediction System",
    layout="wide"
)

from src.shared.connection import init_db
from src.models.production_pipeline import load_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "xgb_model.joblib"
DB_PATH = BASE_DIR / "patients.db"

st.session_state.DB_PATH = DB_PATH


def load_resources():
    if "model" not in st.session_state:
        if MODEL_PATH.exists():
            st.session_state.model = load_model(str(MODEL_PATH))

    if "db_initialized" not in st.session_state:
        init_db(DB_PATH)
        st.session_state.db_initialized = True
        
    if "nome_hospital" not in st.session_state:
        st.session_state.nome_hospital = "Sistema de Avaliação Preventiva"


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
    load_resources()
    apply_theme()
    
    st.title("Sistema de Avaliação")
    st.write("Use o menu lateral para navegar")


if __name__ == "__main__":
    main()