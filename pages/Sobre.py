import streamlit as st

st.header('Sobre')

nome = st.session_state.get("nome_hospital", "Sistema")

st.write(f"Este sistema foi desenvolvido para {nome} como protótipo de suporte à triagem clínica.")
st.write('Funcionalidades: gráfico de risco, geração de PDF, histórico em sqlite, autenticação simples e recomendações nutricionais básicas.')