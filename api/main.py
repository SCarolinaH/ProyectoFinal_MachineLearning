import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBRegressor

BASE = Path(__file__).parent

config = json.loads((BASE / "config_modelo.json").read_text(encoding="utf-8"))
VARIABLES = config["variables"]
CATEGORIAS = config["categorias_zona"]
NOMBRES_Q = [f"p{int(round(q * 100))}" for q in config["cuantiles"]]

modelo = XGBRegressor(enable_categorical=True)
modelo.load_model(BASE / "modelo_pronostico.ubj")

demanda = pd.read_parquet(BASE / "demanda.parquet")
zonas = pd.read_csv(BASE / "zonas.csv").set_index("LocationID")
FECHAS = sorted(demanda["hora"].dt.date.unique())

app = FastAPI(title="Taxis NYC · API de pronóstico")


def _numero(x):
    return None if pd.isna(x) else int(x)


@app.get("/")
def raiz():
    return {"estado": "en línea", "mensaje": "Abre /docs para probar los endpoints"}


@app.get("/info")
def info():
    ids = [z for z in zonas.index if z in set(demanda["PULocationID"])]
    lista = [{"id": int(z), "nombre": zonas.loc[z, "Zone"], "distrito": zonas.loc[z, "Borough"]} for z in ids]
    lista.sort(key=lambda z: z["nombre"])
    return {"zonas": lista, "fechas": [f.isoformat() for f in FECHAS], "cuantiles": NOMBRES_Q}


@app.get("/pronostico")
def pronostico(zona_id: int, fecha: date):
    filas = demanda[(demanda["PULocationID"] == zona_id) & (demanda["hora"].dt.date == fecha)]
    if filas.empty:
        raise HTTPException(status_code=404, detail="No hay datos para esa zona o fecha")
    filas = filas.sort_values("hora").copy()
    filas["zona"] = pd.Categorical(filas["PULocationID"], categories=CATEGORIAS)

    pred = modelo.predict(filas[VARIABLES])
    pred = np.clip(np.sort(pred, axis=1), 0, None)

    horas = []
    for i, (_, fila) in enumerate(filas.iterrows()):
        horas.append({
            "hora": int(fila["hora"].hour),
            "real": _numero(fila["viajes"]),
            "hace_una_semana": _numero(fila["lag_168"]),
            "cuantiles": {n: round(float(pred[i, j]), 1) for j, n in enumerate(NOMBRES_Q)},
        })
    return {
        "zona": {"id": zona_id, "nombre": zonas.loc[zona_id, "Zone"], "distrito": zonas.loc[zona_id, "Borough"]},
        "fecha": fecha.isoformat(),
        "horas": horas,
    }
