import pandas as pd
import joblib
import pickle
import numpy as np


# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    """Carrega modelo salvo (.joblib ou .pkl)."""
    model_path = str(model_path)

    if model_path.endswith(".joblib"):
        return joblib.load(model_path)
    elif model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Formato inválido. Use .joblib ou .pkl")


# =========================
# PREPROCESS INPUT
# =========================
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    column_name = {
        'Gender': 'sexo',
        'Age': 'idade',
        'Height': 'altura',
        'Weight': 'peso',
        'family_history': 'hist_familiar_obes',
        'FAVC': 'cons_altas_cal_freq',
        'FCVC': 'cons_verduras',
        'NCP': 'refeicoes_principais_dia',
        'CAEC': 'lancha_entre_ref',
        'SMOKE': 'fuma',
        'CH2O': 'agua_dia',
        'SCC': 'controle_calorias',
        'FAF': 'ativ_fisica',
        'TUE': 'uso_tecnologia',
        'CALC': 'cons_alcool',
        'MTRANS': 'transporte',
        'Obesity': 'nivel_obesidade'
    }

    df = pd.DataFrame([input_dict]).rename(columns=column_name)

    # YES/NO
    yes_no_cols = ['hist_familiar_obes', 'cons_altas_cal_freq', 'fuma', 'controle_calorias']
    for c in yes_no_cols:
        if c in df.columns:
            df[c] = df[c].map({'yes': 1, 'no': 0})

    # SEXO
    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].map({'Female': 0, 'Male': 1})

    # ARREDONDAMENTOS
    col_int = ['idade', 'cons_verduras', 'refeicoes_principais_dia', 'ativ_fisica', 'uso_tecnologia']
    for c in col_int:
        if c in df.columns:
            df[c] = df[c].round(0)

    if "agua_dia" in df.columns:
        df["agua_dia"] = df["agua_dia"].round().astype(int)

    for c in ['altura', 'peso']:
        if c in df.columns:
            df[c] = df[c].round(2)

    # MAPEAMENTOS
    freq = {'no': 0, 'Sometimes': 0, 'Frequently': 1, 'Always': 1}
    ativ = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    transp = {'Automobile': 1, 'Motorbike': 1, 'Public_Transportation': 1, 'Bike': 0, 'Walking': 0}

    if "lancha_entre_ref" in df.columns:
        df["lancha_entre_ref_bin"] = df["lancha_entre_ref"].map(freq)

    if "cons_alcool" in df.columns:
        df["cons_alcool_bin"] = df["cons_alcool"].map(freq)

    if "ativ_fisica" in df.columns:
        df["ativ_fisica_bin"] = df["ativ_fisica"].map(ativ)

    if "transporte" in df.columns:
        df["transporte_bin"] = df["transporte"].map(transp)

    # SELEÇÃO FINAL (igual treino)
    cols_model = [
        'hist_familiar_obes', 'cons_altas_cal_freq', 'cons_verduras',
        'refeicoes_principais_dia', 'lancha_entre_ref_bin', 'fuma',
        'agua_dia', 'controle_calorias', 'ativ_fisica_bin',
        'uso_tecnologia', 'cons_alcool_bin', 'transporte_bin'
    ]

    df = df[cols_model]

    return df


# =========================
# PREDICT
# =========================
def predict_from_input(model, input_dict: dict):
    df_processed = preprocess_input(input_dict)

    pred_raw = model.predict(df_processed)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_processed)[0][1]

    if pred_raw == 1:
        msg = "⚠️ Há indícios de que pode ter obesidade."
    else:
        msg = "✅ Baixa probabilidade de obesidade."

    if prob is not None:
        return {
            "mensagem": msg,
            "probabilidade": f"{prob:.2%}"
        }

    return {"mensagem": msg}