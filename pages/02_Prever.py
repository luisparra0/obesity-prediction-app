"""Page: formulário de triagem e integração com modelo salvo."""

from pathlib import Path
import base64
import re
from typing import Any, Callable, Mapping, Optional

import streamlit as st

from src.shared import utils
from src.shared import plots
import src.shared.connection as connection

from src.models.production_pipeline import predict_from_input, load_model


# =========================
# SESSION DEFAULTS
# =========================

_DEFAULT_SESSION = {
    "DB_PATH": "records.db",
    "nome_hospital": "Sistema de Avaliação Preventiva",
    "LOGO_PATH": "logo.png",
    "PRIMARY_COLOR": (25, 118, 210),
    "ACCENT_COLOR": (255, 152, 0),
}


def _ensure_session_defaults():
    for k, v in _DEFAULT_SESSION.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_session_defaults()


# =========================
# LOAD MODEL (FIXADO)
# =========================

@st.cache_resource
def load_ml_model():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "src" / "models" / "xgb_model.joblib"

    if not MODEL_PATH.exists():
        st.error("❌ Modelo não encontrado no caminho esperado.")
        return None

    try:
        model = load_model(str(MODEL_PATH))
        st.session_state['loaded_model_path'] = MODEL_PATH.name
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None


# =========================
# MAIN PAGE
# =========================

def render_predict():
    st.header('Formulário de triagem')

    model = load_ml_model()

    if model is None:
        st.error("❌ Não é possível fazer predições sem um modelo carregado.")
        return

    with st.form('predict_form'):
        col1, col2 = st.columns(2)

        with col1:
            nome = st.text_input('Nome do paciente')
            family_history = 'yes' if st.radio('Histórico familiar', ['Sim', 'Não']) == 'Sim' else 'no'
            FAVC = 'yes' if st.radio('Alta caloria', ['Sim', 'Não']) == 'Sim' else 'no'
            FCVC = st.number_input('Verduras (1-3)', 1, 3, 3)
            NCP = st.number_input('Refeições (1-4)', 1, 4, 3)
            CAEC = st.selectbox('Lanches', ['no', 'Sometimes', 'Frequently', 'Always'])

        with col2:
            SMOKE = 'yes' if st.radio('Fuma?', ['Sim', 'Não']) == 'Sim' else 'no'
            CH2O = st.number_input('Água (1-3)', 1, 3, 2)
            SCC = 'yes' if st.radio('Controla calorias?', ['Sim', 'Não']) == 'Sim' else 'no'
            FAF = st.number_input('Atividade (0-3)', 0, 3, 1)
            TUE = st.number_input('Tecnologia (0-2)', 0, 2, 1)

        CALC = st.selectbox('Álcool', ['never', 'sometimes', 'frequently', 'always'])
        MTRANS = st.selectbox('Transporte', ['Automobile', 'Motorbike', 'Public_Transportation', 'Bike', 'Walking'])

        gender = st.selectbox('Gênero', ['Male', 'Female'])
        age = st.number_input('Idade', 1, 120, 25)
        height = st.number_input('Altura', 1.0, 2.5, 1.70)
        weight = st.number_input('Peso', 30.0, 200.0, 70.0)

        submitted = st.form_submit_button('🔍 Analisar')

    if submitted:
        inputs = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history': family_history,
            'FAVC': FAVC,
            'FCVC': FCVC,
            'NCP': NCP,
            'CAEC': CAEC,
            'SMOKE': SMOKE,
            'CH2O': CH2O,
            'SCC': SCC,
            'FAF': FAF,
            'TUE': TUE,
            'CALC': CALC,
            'MTRANS': MTRANS
        }

        result = predict_from_input(model, inputs)

        mensagem = result.get("mensagem", "")
        prob = result.get("probabilidade", "0%")

        st.success(mensagem)
        st.metric("Probabilidade", prob)

        # gráfico
        try:
            prob_val = float(prob.replace("%", ""))
            st.pyplot(plots.render_risk_chart(prob_val))
        except:
            pass

        # salvar
        connection.save_record(
            st.session_state.get('user_type', 'anon'),
            st.session_state.get('user_name', 'anon'),
            inputs,
            mensagem,
            prob,
            st.session_state.DB_PATH
        )

        # PDF
        pdf_bytes = utils.generate_pdf(nome or "Paciente", inputs, mensagem, prob)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="relatorio.pdf">⬇️ Baixar PDF</a>'
        st.markdown(href, unsafe_allow_html=True)


# =========================
# RUN
# =========================

render_predict()