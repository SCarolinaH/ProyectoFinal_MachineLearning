import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

FECHAS = ["order_purchase_timestamp", "order_approved_at",
          "order_delivered_carrier_date", "order_delivered_customer_date",
          "order_estimated_delivery_date"]

CATEGORICAS = ["categoria", "estado_cliente", "estado_vendedor", "tipo_pago"]

NUMERICAS = ["n_items", "n_vendedores", "precio_total", "flete_total",
             "peso_total_g", "fotos_promedio", "cuotas_max",
             "dias_retraso", "retraso_positivo", "entrega_tardia",
             "dias_entrega", "dias_prometidos", "horas_aprobacion",
             "dias_a_transportista", "dias_ultima_milla",
             "flete_ratio", "mismo_estado", "dia_semana"]


def crear_variables(df):
    d = df.copy()
    for c in FECHAS:
        d[c] = pd.to_datetime(d[c])
    dia = 86400
    d["dias_retraso"] = (d["order_delivered_customer_date"] - d["order_estimated_delivery_date"]).dt.total_seconds() / dia
    d["retraso_positivo"] = d["dias_retraso"].clip(lower=0)
    d["entrega_tardia"] = (d["dias_retraso"] > 0).astype(int)
    d["dias_entrega"] = (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / dia
    d["dias_prometidos"] = (d["order_estimated_delivery_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / dia
    d["horas_aprobacion"] = (d["order_approved_at"] - d["order_purchase_timestamp"]).dt.total_seconds() / 3600
    d["dias_a_transportista"] = (d["order_delivered_carrier_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / dia
    d["dias_ultima_milla"] = (d["order_delivered_customer_date"] - d["order_delivered_carrier_date"]).dt.total_seconds() / dia
    d["flete_ratio"] = d["flete_total"] / (d["precio_total"] + 1)
    d["mismo_estado"] = (d["estado_cliente"] == d["estado_vendedor"]).astype(int)
    d["dia_semana"] = d["order_purchase_timestamp"].dt.dayofweek
    return d


def construir_pipeline(modelo, escalar=False):
    pasos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("esc", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(pasos_num), NUMERICAS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="desconocido")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
    ])
    return Pipeline([
        ("vars", FunctionTransformer(crear_variables)),
        ("prep", prep),
        ("modelo", modelo),
    ])


def precision_top(y_real, score, fraccion=0.10):
    y_real = np.asarray(y_real)
    score = np.asarray(score, dtype=float)
    k = int(np.ceil(len(score) * fraccion))
    idx = np.argsort(-score)[:k]
    return y_real[idx].mean()
