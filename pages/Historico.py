import streamlit as st
import sqlite3
import base64
import pandas as pd


def render_history():
    st.header('Histórico de avaliações')

    conn = sqlite3.connect(st.session_state.DB_PATH)
    c = conn.cursor()

    # ✅ garante que a tabela existe (resolve erro do cloud)
    c.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_type TEXT,
        user_name TEXT,
        mensagem TEXT,
        probabilidade TEXT
    )
    """)

    conn.commit()

    # ✅ tenta buscar dados com segurança
    try:
        c.execute("""
        SELECT id, timestamp, user_type, user_name, mensagem, probabilidade 
        FROM records 
        ORDER BY id DESC 
        LIMIT 200
        """)
        rows = c.fetchall()
    except Exception:
        rows = []

    conn.close()

    # ✅ caso não tenha dados
    if not rows:
        st.info('Nenhum registro encontrado.')
        return

    df = pd.DataFrame(rows, columns=[
        'id', 'timestamp', 'user_type', 'user_name', 'mensagem', 'probabilidade'
    ])

    st.dataframe(df)

    # ✅ exportação CSV
    st.markdown('**Exportar como CSV**')
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()

    href = f'<a href="data:file/csv;base64,{b64}" download="historico.csv">⬇️ Baixar histórico (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)


render_history()