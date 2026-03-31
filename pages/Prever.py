"""Page: formulário de triagem e integração com modelo salvo.

Este módulo contém a UI do Streamlit para coletar dados do paciente, chamar a
função de predição em `src.models.production_pipeline` e exibir resultados.
Pequenas melhorias de organização e tipagem foram adicionadas sem alterar
qualquer comportamento funcional do app.
"""

import os
import sys
import glob
import base64
import re
from typing import Any, Callable, Mapping, Optional
from src.shared import utils
from src.shared.paths import DATA_DIR
from src.shared import plots
import src.shared.connection as connection

import streamlit as st

# Defaults kept explicit to ensure the app runs as before
_DEFAULT_SESSION = {
    "DB_PATH": "records.db",
    "HOSPITAL_NAME": "Hospital Padrão",
    "LOGO_PATH": "logo.png",
    "FONT_PATH": "DejaVuSans.ttf",
    "PRIMARY_COLOR": (25, 118, 210),
    "ACCENT_COLOR": (255, 152, 0),
    "FIELD_MAPPING": {},
}

_DEFAULT_CATEGORY_TRANSLATION = {
    'yes': 'Sim',
    'no': 'Não',
    'Sometimes': 'Às vezes',
    'sometimes': 'Às vezes',
    'Frequently': 'Frequentemente',
    'frequently': 'Frequentemente',
    'Always': 'Sempre',
    'always': 'Sempre',
    'never': 'Nunca',
    'Never': 'Nunca',
    'Automobile': 'Automóvel',
    'Motorbike': 'Moto',
    'Public_Transportation': 'Transporte Público',
    'Bike': 'Bicicleta',
    'Walking': 'A pé'
}

_DEFAULT_FIELD_NAME_MAP = {
    'family_history': 'Histórico familiar de obesidade',
    'FAVC': 'Consome alimentos de alta caloria',
    'FCVC': 'Consumo de verduras',
    'NCP': 'Refeições por dia',
    'CAEC': 'Lanches entre refeições',
    'SMOKE': 'Fuma',
    'CH2O': 'Ingestão de água',
    'SCC': 'Controla calorias',
    'FAF': 'Atividade física',
    'TUE': 'Uso de tecnologia',
    'CALC': 'Consumo de álcool',
    'MTRANS': 'Meio de transporte'
}

_DEFAULT_EXPLAIN_NUMERIC = {
    'FCVC': {1: 'Raramente', 2: 'Ocasionalmente', 3: 'Frequentemente'},
    'CH2O': {1: '<1L/dia', 2: '1-2L/dia', 3: '>2L/dia'},
    'NCP': {1: '1 refeição', 2: '2 refeições', 3: '3 refeições', 4: '4 refeições'},
    'FAF': {0: 'Inativo', 1: 'Baixa atividade', 2: 'Atividade moderada', 3: 'Atividade intensa'},
    'TUE': {0: 'Pouco uso', 1: 'Moderado', 2: 'Alto uso'}
}


def _ensure_session_defaults() -> None:
    """Ensure Streamlit session_state contains expected default keys."""
    for k, v in _DEFAULT_SESSION.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "CATEGORY_TRANSLATION" not in st.session_state:
        st.session_state["CATEGORY_TRANSLATION"] = dict(_DEFAULT_CATEGORY_TRANSLATION)

    if "FIELD_NAME_MAP" not in st.session_state:
        st.session_state["FIELD_NAME_MAP"] = dict(_DEFAULT_FIELD_NAME_MAP)

    if "EXPLAIN_NUMERIC" not in st.session_state or not st.session_state.get("EXPLAIN_NUMERIC"):
        st.session_state["EXPLAIN_NUMERIC"] = dict(_DEFAULT_EXPLAIN_NUMERIC)


# Apply defaults at module import time (same behavior as before)
_ensure_session_defaults()

# Tenta importar o pacote `src`; se falhar, adiciona o root do repo ao `sys.path` e tenta novamente.
try:
    from src.models.production_pipeline import predict_from_input, load_model
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.models.production_pipeline import predict_from_input, load_model

# Função para carregar modelo com cache
@st.cache_resource
def load_ml_model() -> Optional[object]:
    """Carrega o modelo XGBoost ou Random Forest disponível em src/models/.

    Retorna o objeto do modelo ou `None` em caso de erro (comportamento inalterado).
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_files = sorted(glob.glob(os.path.join(models_dir, "*.joblib")))

    if not model_files:
        st.error("❌ Nenhum modelo encontrado em `src/models/`. Por favor, treine e salve um modelo.")
        return None

    # Preferir XGBoost se disponível, senão usar Random Forest
    xgb_files = [f for f in model_files if 'xgb' in f.lower()]
    model_path = xgb_files[0] if xgb_files else model_files[0]

    try:
        model = load_model(model_path)
        # Guardar o caminho do modelo carregado no estado da sessão para uso posterior
        try:
            st.session_state['loaded_model_path'] = os.path.basename(model_path)
        except Exception:
            # Não falhar caso session_state não esteja disponível por algum motivo
            pass
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        # Garantir que não fica um caminho antigo em session_state
        try:
            st.session_state['loaded_model_path'] = None
        except Exception:
            pass
        return None


def render_predict(load_model_fn: Optional[Callable[[], object]] = None, predict_fn: Optional[Callable[[Mapping[str, Any]], dict]] = None) -> None:
    """Renderiza o formulário de predição (Streamlit)."""
    col1 = st.columns(1)[0]

    # -----------------------------
    # Mostrar modelo usado e métricas (accuracy, recall, f1)
    # -----------------------------
    # Tentar recuperar o nome do modelo carregado do session_state
    model_file = st.session_state.get('loaded_model_path') if 'loaded_model_path' in st.session_state else None
    if model_file is None:
        # fallback: descobrir arquivo de modelo sem carregá-lo
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_files = sorted(glob.glob(os.path.join(models_dir, "*.joblib")))
        if model_files:
            xgb_files = [f for f in model_files if 'xgb' in f.lower()]
            model_file = os.path.basename(xgb_files[0] if xgb_files else model_files[0])



    st.header('Formulário de triagem')
    # If no predictor function provided, load the model from default location
    if predict_fn is None:
        try:
            model = load_ml_model()
            if model is not None:
                predict_fn = lambda x: predict_from_input(model, x)
        except Exception:
            predict_fn = None

    with st.form('predict_form'):
        
        col1, col2 = st.columns(2)
    
        with col1:
            nome = st.text_input('Nome do paciente', value="")
            # Mostrar opções em português, converter para valores esperados pelo modelo
            family_history_display = st.radio('Histórico familiar de obesidade', ['Sim', 'Não'])
            family_history = 'yes' if family_history_display == 'Sim' else 'no'
            FAVC_display = st.radio('Consome alimentos de alta caloria', ['Sim', 'Não'])
            FAVC = 'yes' if FAVC_display == 'Sim' else 'no'
            FCVC = st.number_input('Frequência consumo verduras (1=raramente,3=sempre)', min_value=1, max_value=3, value=3)
            NCP = st.number_input('Refeições por dia (1-4)', min_value=1, max_value=4, value=3)
            CAEC_display = st.selectbox('Lanche entre refeições', ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'])
            CAEC_MAP = { 'Nunca': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
            CAEC = CAEC_MAP.get(CAEC_display, 'no')

        with col2:
            SMOKE_display = st.radio('Fuma?', ['Sim', 'Não'])
            SMOKE = 'yes' if SMOKE_display == 'Sim' else 'no'
            CH2O = st.number_input('Água por dia (1<1L,2=1-2L,3>2L)', min_value=1, max_value=3, value=2)
            SCC_display = st.radio('Controla calorias?', ['Sim', 'Não'])
            SCC = 'yes' if SCC_display == 'Sim' else 'no'
            FAF = st.number_input('Atividade física (0-3)', min_value=0, max_value=3, value=1)
            TUE = st.number_input('Uso tecnologia (0-2)', min_value=0, max_value=2, value=1)

        col3, col4 = st.columns(2)
        with col3:
            CALC_display = st.selectbox('Consumo de álcool', ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'])
            CALC_MAP = {'Nunca':'never', 'Às vezes':'sometimes', 'Frequentemente':'frequently', 'Sempre':'always', 'no':'never'}
            CALC = CALC_MAP.get(CALC_display, 'never')
        with col4:
            MTRANS_display = st.selectbox('Meio de transporte', ['Automóvel','Moto','Transporte Público','Bicicleta','A pé'])
            MTRANS_MAP = {
                'Automóvel': 'Automobile',
                'Moto': 'Motorbike',
                'Transporte Público': 'Public_Transportation',
                'Bicicleta': 'Bike',
                'A pé': 'Walking'
            }
            MTRANS = MTRANS_MAP.get(MTRANS_display, 'Automobile')

        submitted = st.form_submit_button('🔍 Analisar paciente')

    if submitted:
        
        if predict_fn is None:
            st.error("⚠ Nenhuma função de predição fornecida.")
            return
    
        if not nome.strip():
            st.error("O campo Nome é obrigatório.")
            return

        gender_display = st.selectbox('Gênero', ['Masculino', 'Feminino'])
        gender = 'Male' if gender_display == 'Masculino' else 'Female'

        age = st.number_input('Idade', min_value=1, max_value=120, value=25)

        height = st.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=1.70)

        weight = st.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=70.0)

        inputs = {
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'family_history': family_history,
            'high_calorie_food': FAVC,
            'vegetable_consumption': FCVC,
            'main_meals_per_day': NCP,
            'snacks_between_meals': CAEC,
            'smokes': SMOKE,
            'water_intake': CH2O,
            'calorie_monitoring': SCC,
            'physical_activity': FAF,
            'screen_time': TUE,
            'alcohol_consumption': CALC,
            'transportation': MTRANS
        }
        

        result = predict_fn(inputs)

        mensagem = result.get("mensagem", "Sem mensagem")
        prob_raw = result.get("probabilidade")

        if prob_raw is None:
            st.error("Erro: probabilidade não encontrada na resposta.")
            st.write("RESULT DEBUG:", result)
            return

        # ✅ Extrai apenas o número
        match = re.search(r"([\d.,]+)", prob_raw)

        if not match:
            st.error(f"Erro: probabilidade inválida → {prob_raw}")
            st.write("RESULT DEBUG:", result)
            return

        prob = float(match.group(1).replace(",", "."))

        # ✅ Exibe na interface com rótulos mais claros
        # Interpretação binária (Sim/Não) e classificação de risco por faixas
        previsao_binaria = "Sim" if prob >= 50 else "Não"
        if prob >= 60:
            risco = "Alto"
            st.error(f"🔴 Resultado: {previsao_binaria} — Risco {risco} de obesidade ({prob:.2f}%)")
        elif prob >= 30:
            risco = "Moderado"
            st.warning(f"🟠 Resultado: {previsao_binaria} — Risco {risco} de obesidade ({prob:.2f}%)")
        else:
            risco = "Baixo"
            st.success(f"🟢 Resultado: {previsao_binaria} — Risco {risco} de obesidade ({prob:.2f}%)")

        st.metric("Probabilidade (%)", f"{prob:.2f}%")
        st.pyplot(plots.render_risk_chart(prob))

        st.subheader('Recomendações nutricionais')
        recs = utils.recommend_nutrition_profile(inputs)
        for r in recs:
            st.write('- ' + r)

        # salvar no DB se usuário autenticado (médico) ou se paciente quiser
        user_type = st.session_state.get('user_type', 'anon')
        user_name = st.session_state.get('user_name', 'anon')
        connection.save_record(user_type, user_name, inputs, mensagem, prob, st.session_state.DB_PATH)
        st.info('Registro salvo no histórico.')

        # gerar PDF
        pdf_bytes = utils.generate_pdf(nome or 'Paciente', inputs, mensagem, prob)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="relatorio_{nome or "paciente"}.pdf">⬇️ Baixar relatório em PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

# Carregar modelo e chamar render_predict
if __name__ == '__main__':
    model = load_ml_model()
    if model is not None:
        render_predict(
            load_model_fn=load_ml_model,
            predict_fn=lambda x: predict_from_input(model, x)
        )
    else:
        st.error("❌ Não é possível fazer predições sem um modelo carregado.")