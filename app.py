
import numpy as np
import pandas as pd
import requests
import pickle
import os

from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# CONFIGURAÇÕES
# ==========================================

app = Flask(__name__)
CORS(app)

FIREBASE_URL = "https://estufaescritoriojwl-default-rtdb.firebaseio.com/estufa/historico.json"

MODEL_FILE = "model.pkl"
CONFIG_FILE = "config.pkl"

JANELA = 3  # Quantidade de leituras anteriores utilizadas


# ==========================================
# BUSCAR DADOS DO FIREBASE
# ==========================================

def timestamp_do_id(firebase_id):

    alfabeto = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"

    timestamp = 0

    for caractere in firebase_id[:8]:
        timestamp = timestamp * 64 + alfabeto.index(caractere)

    return pd.to_datetime(timestamp, unit="ms", utc=True)

def buscar_dados():

    resposta = requests.get(FIREBASE_URL, timeout=20)
    resposta.raise_for_status()

    dados = resposta.json()

    if not dados:
        raise ValueError("Firebase não possui histórico.")

    registros = []

    for firebase_id, valores in dados.items():

        if not isinstance(valores, dict):
            continue

        registro = valores.copy()

        registro["timestamp"] = timestamp_do_id(firebase_id)

        registros.append(registro)

    return registros
# ==========================================
# TRATAMENTO DOS DADOS
# ==========================================

def preparar_dados(registros):

    df = pd.DataFrame(registros)

    # O timestamp já vem convertido do ID do Firebase
    if "timestamp" not in df.columns:
        raise ValueError("Não foi possível identificar o timestamp.")

    df["_tempo"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # Remover registros com datas inválidas
    df = df.dropna(subset=["_tempo"])

    # Organizar cronologicamente
    df = df.sort_values("_tempo").reset_index(drop=True)

    def converter_data(valor):

        if pd.isna(valor):
            return pd.NaT

        # Timestamp Unix
        if isinstance(valor, (int, float, np.integer, np.floating)):
            unidade = "ms" if abs(valor) > 100000000000 else "s"

            return pd.to_datetime(
                valor,
                unit=unidade,
                utc=True
            )

        return pd.to_datetime(
            valor,
            errors="coerce",
            utc=True
        )

    df["_tempo"] = df[campo_tempo].apply(converter_data)

    # Remover datas inválidas
    df = df.dropna(subset=["_tempo"])

    df = df.sort_values("_tempo").reset_index(drop=True)

    # Identificar automaticamente os sensores numéricos
    colunas = []

    for coluna in df.columns:

        if coluna in [campo_tempo, "_tempo"]:
            continue

        valores = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

        if valores.notna().sum() > 0:
            df[coluna] = valores
            colunas.append(coluna)

    if not colunas:
        raise ValueError("Nenhum sensor numérico encontrado.")

    # Preencher leituras ausentes
    df[colunas] = df[colunas].interpolate(
        method="linear",
        limit_direction="both"
    )

    df = df.dropna(subset=colunas)

    if len(df) < 10:
        raise ValueError("Poucos registros para treinamento.")

    return df, colunas


# ==========================================
# CRIAR DADOS DE TREINAMENTO
# ==========================================

def criar_treinamento(df, sensores, minutos):

    X = []
    y = []

    tempos = df["_tempo"].tolist()

    valores = df[sensores].to_numpy(dtype=float)

    horizonte = pd.Timedelta(minutes=minutos)

    for i in range(JANELA - 1, len(df)):

        tempo_atual = tempos[i]

        tempo_futuro = tempo_atual + horizonte

        # Encontrar a primeira leitura no horizonte desejado
        futuros = [
            j for j in range(i + 1, len(df))
            if tempos[j] >= tempo_futuro
        ]

        if not futuros:
            continue

        j = futuros[0]

        # Leituras anteriores dos sensores
        entrada = valores[i - JANELA + 1:i + 1]

        X.append(entrada.flatten())

        # Valores reais observados no futuro
        y.append(valores[j])

    if len(X) < 5:
        raise ValueError(
            "Dados históricos insuficientes para prever "
            f"{minutos} minutos. Verifique a quantidade "
            "e o intervalo das leituras."
        )

    return np.array(X), np.array(y)


# ==========================================
# ATUALIZAR E TREINAR O MODELO
# ==========================================

@app.route("/atualizar_modelo", methods=["GET"])
def atualizar_modelo():

    try:

        minutos = int(request.args.get("minutos", 50))

        if minutos < 1 or minutos > 1440:
            return jsonify({
                "erro": "O intervalo deve estar entre 1 e 1440 minutos."
            }), 400

        registros = buscar_dados()

        df, sensores = preparar_dados(registros)

        X, y = criar_treinamento(
            df,
            sensores,
            minutos
        )

        # Treinar modelo
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)

        # Salvar arquivos dinamicamente
        with open(MODEL_FILE, "wb") as arquivo:
            pickle.dump(model, arquivo)

        config = {
            "sensores": sensores,
            "janela": JANELA,
            "minutos": minutos,
            "ultima_atualizacao": datetime.now(
                timezone.utc
            ).isoformat(),
            "registros": len(df),
            "amostras_treinamento": len(X)
        }

        with open(CONFIG_FILE, "wb") as arquivo:
            pickle.dump(config, arquivo)

        return jsonify({
            "status": "Modelo treinado com sucesso!",
            "algoritmo": "Random Forest Regressor",
            "previsao_minutos": minutos,
            "sensores": sensores,
            "registros_utilizados": len(df),
            "amostras_treinamento": len(X),
            "arquivos": [
                MODEL_FILE,
                CONFIG_FILE
            ]
        })

    except Exception as e:

        return jsonify({
            "erro": str(e)
        }), 500


# ==========================================
# PREVISÃO DOS SENSORES
# ==========================================

@app.route("/previsao", methods=["GET"])
def previsao():

    try:

        if not os.path.exists(MODEL_FILE):
            return jsonify({
                "erro": "Treine o modelo primeiro."
            }), 400

        with open(MODEL_FILE, "rb") as arquivo:
            model = pickle.load(arquivo)

        with open(CONFIG_FILE, "rb") as arquivo:
            config = pickle.load(arquivo)

        minutos = int(
            request.args.get("minutos", config["minutos"])
        )

        if minutos != config["minutos"]:
            return jsonify({
                "erro": (
                    f"Modelo treinado para {config['minutos']} minutos. "
                    "Atualize o modelo para utilizar outro intervalo."
                )
            }), 400

        # Buscar dados mais recentes
        registros = buscar_dados()

        df, sensores = preparar_dados(registros)

        if sensores != config["sensores"]:
            return jsonify({
                "erro": "Os sensores mudaram. Atualize o modelo."
            }), 400

        valores = df[sensores].to_numpy(dtype=float)

        if len(valores) < JANELA:
            raise ValueError("Leituras insuficientes para previsão.")

        entrada = valores[-JANELA:].flatten()

        array = np.array([entrada])

        resultado = model.predict(array)[0]

        dados_previstos = {}

        for sensor, valor in zip(sensores, resultado):
            dados_previstos[sensor] = round(float(valor), 2)

        ultima_data = df["_tempo"].iloc[-1]

        previsao_data = (
            ultima_data + pd.Timedelta(minutes=minutos)
        )

        dados_atuais = {
            sensor: round(float(df[sensor].iloc[-1]), 2)
            for sensor in sensores
        }

        return jsonify({
            "status": "Previsão realizada!",
            "previsao_para": previsao_data.isoformat(),
            "intervalo_minutos": minutos,
            "ultima_leitura": ultima_data.isoformat(),
            "dados_atuais": dados_atuais,
            "previsao": dados_previstos
        })

    except Exception as e:

        return jsonify({
            "erro": str(e)
        }), 500


# ==========================================
# ROTA PRINCIPAL
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "API Online",
        "projeto": "STUFY - Mini Estufa Inteligente",
        "algoritmo": "Random Forest Regressor",
        "rotas": {
            "atualizar_modelo": "/atualizar_modelo?minutos=50",
            "previsao": "/previsao?minutos=50"
        }
    })


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
