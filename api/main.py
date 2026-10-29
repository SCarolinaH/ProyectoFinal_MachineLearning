"""API de riesgo de mala reseña (Olist).

GET  /        -> confirma que la API está viva
POST /riesgo  -> recibe las 17 columnas del contrato y devuelve la probabilidad de 1-2 estrellas
"""
import os
import faulthandler
faulthandler.enable()   # si el proceso se cae por un error de sistema, imprime dónde

# Render gratis tiene 512 MB y "ve" muchos CPUs del servidor: XGBoost abre un hilo por CPU
# al predecir y se queda sin memoria (502). Se limita a 1 hilo ANTES de importar las librerías.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Los archivos se buscan junto a este main.py, sin importar desde dónde se lance uvicorn
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# El pipeline guardado usa preparacion.crear_variables: hay que importarlo ANTES de cargarlo
import preparacion  # noqa: F401,E402

app = FastAPI(
    title="API de riesgo de mala reseña · Olist",
    description="Predice la probabilidad de que un pedido entregado reciba 1 o 2 estrellas.",
    version="1.0",
)

print("Cargando modelo...", flush=True)
modelo = joblib.load(BASE / "modelo_riesgo.joblib")
try:
    modelo.named_steps["modelo"].set_params(n_jobs=1)   # XGBoost con un solo hilo
except Exception:
    pass
with open(BASE / "umbral.json") as f:
    UMBRAL = float(json.load(f)["umbral"])
print("Modelo cargado. Umbral:", UMBRAL, flush=True)

FECHAS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]
COLUMNAS_NUMERICAS = ["n_items", "n_vendedores", "precio_total", "flete_total",
                      "peso_total_g", "fotos_promedio", "cuotas_max"]
COLUMNAS_TEXTO = ["estado_cliente", "categoria", "estado_vendedor", "tipo_pago"]


class Pedido(BaseModel):
    """Las 17 columnas del contrato (todas menos mala_resena).
    Obligatorias: las que todo pedido entregado tiene. Opcionales: las que pueden faltar."""

    order_id: str
    order_purchase_timestamp: datetime
    order_delivered_customer_date: datetime
    order_estimated_delivery_date: datetime
    estado_cliente: str
    precio_total: float = Field(..., ge=0)

    order_approved_at: Optional[datetime] = None
    order_delivered_carrier_date: Optional[datetime] = None
    n_items: Optional[int] = Field(None, ge=0)
    n_vendedores: Optional[int] = Field(None, ge=0)
    flete_total: Optional[float] = Field(None, ge=0)
    peso_total_g: Optional[float] = Field(None, ge=0)
    fotos_promedio: Optional[float] = Field(None, ge=0)
    categoria: Optional[str] = None
    estado_vendedor: Optional[str] = None
    tipo_pago: Optional[str] = None
    cuotas_max: Optional[int] = Field(None, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "order_id": "d6cb7abec3164d194483906f3702748a",
                "order_purchase_timestamp": "2018-03-07 22:02:23",
                "order_approved_at": "2018-03-09 02:35:30",
                "order_delivered_carrier_date": "2018-03-09 20:03:25",
                "order_delivered_customer_date": "2018-04-11 20:07:37",
                "order_estimated_delivery_date": "2018-04-04 00:00:00",
                "estado_cliente": "CE",
                "n_items": 1,
                "n_vendedores": 1,
                "precio_total": 420.0,
                "flete_total": 45.3,
                "peso_total_g": 1500.0,
                "fotos_promedio": 2.0,
                "categoria": "moveis_decoracao",
                "estado_vendedor": "SP",
                "tipo_pago": "credit_card",
                "cuotas_max": 5,
            }
        }
    }


def construir_fila(datos: dict) -> pd.DataFrame:
    """Convierte el pedido en una fila con los tipos que espera el pipeline."""
    fila = pd.DataFrame([datos])
    for c in FECHAS:
        s = pd.to_datetime(fila[c])
        if getattr(s.dt, "tz", None) is not None:  # si llega con zona horaria, se quita
            s = s.dt.tz_localize(None)
        fila[c] = s
    for c in COLUMNAS_NUMERICAS:
        fila[c] = pd.to_numeric(fila[c], errors="coerce")
    for c in COLUMNAS_TEXTO:
        fila[c] = fila[c].astype(object).where(fila[c].notna(), np.nan)
    return fila


@app.get("/")
def raiz():
    return {"estado": "vivo", "servicio": "riesgo de mala reseña", "documentacion": "/docs"}


@app.post("/riesgo")
def riesgo(pedido: Pedido):
    print("1. pedido recibido", flush=True)
    fila = construir_fila(pedido.model_dump())
    print("2. fila construida", flush=True)
    probabilidad = float(modelo.predict_proba(fila)[:, 1][0])
    print("3. prediccion lista", probabilidad, flush=True)
    return {
        "order_id": pedido.order_id,
        "probabilidad": round(probabilidad, 4),
        "riesgo_alto": bool(probabilidad >= UMBRAL),
        "umbral": round(UMBRAL, 4),
    }