import streamlit as st

st.header('Sobre')

nome = st.session_state.get("nome_hospital", "Sistema")

st.markdown(f"""
O **{nome}** é uma aplicação de Machine Learning desenvolvida para apoiar a triagem clínica e identificar riscos de obesidade com base em dados comportamentais e hábitos de vida.

###  Objetivo
Auxiliar profissionais de saúde na tomada de decisão preventiva, fornecendo:
- Estimativa de risco de obesidade
- Recomendações nutricionais automatizadas
- Geração de relatórios em PDF
- Histórico de avaliações

###  Tecnologias
- Python (Pandas, Scikit-learn, XGBoost)
- Streamlit
- SQLite
""")