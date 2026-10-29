# App de Streamlit · Olist: del pedido a la reseña
# Páginas: riesgo de un pedido (vía API), comentarios, pronóstico de junio y ficha del modelo.
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Olist · riesgo de mala reseña", page_icon="📦", layout="wide")

CONTRATO_ENTRADA = [
    "order_id", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date", "estado_cliente", "n_items",
    "n_vendedores", "precio_total", "flete_total", "peso_total_g", "fotos_promedio", "categoria",
    "estado_vendedor", "tipo_pago", "cuotas_max",
]
ENTEROS = {"n_items", "n_vendedores", "cuotas_max"}
DECIMALES = {"precio_total", "flete_total", "peso_total_g", "fotos_promedio"}


# ---------- utilidades ----------
def url_api_por_defecto():
    """La URL de la API viene de st.secrets (Streamlit Cloud) o de una variable de entorno."""
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data
def leer_csv(nombre):
    return pd.read_csv(BASE / nombre)


@st.cache_data
def leer_json(nombre):
    return json.loads((BASE / nombre).read_text(encoding="utf-8"))


def a_json(valor, campo):
    """Convierte lo escrito en la tabla al tipo que espera la API (vacío -> null)."""
    texto = "" if valor is None else str(valor).strip()
    if texto == "" or texto.lower() in {"nan", "none", "nat"}:
        return None
    if campo in ENTEROS:
        return int(float(texto))
    if campo in DECIMALES:
        return float(texto)
    return texto


def llamar_api(ruta, metodo="get", **kwargs):
    url = st.session_state.api_url.rstrip("/") + ruta
    return getattr(requests, metodo)(url, timeout=90, **kwargs)


if "api_url" not in st.session_state:
    st.session_state.api_url = url_api_por_defecto()


# ---------- página 1: riesgo de un pedido ----------
def pagina_riesgo():
    st.title("Riesgo de mala reseña")
    st.caption("Elige un pedido de ejemplo, cambia lo que quieras y pregunta a la API.")

    ejemplos = leer_csv("ejemplos_pedidos.csv")
    retraso = (pd.to_datetime(ejemplos.order_delivered_customer_date)
               - pd.to_datetime(ejemplos.order_estimated_delivery_date)).dt.days
    etiquetas = {
        i: f"{ejemplos.order_id[i][:8]}… · " + (f"retraso +{d} días" if d > 0 else f"a tiempo ({d} días)")
        for i, d in retraso.items()
    }
    elegido = st.selectbox("Pedido de ejemplo", ejemplos.index, format_func=lambda i: etiquetas[i])
    fila = ejemplos.loc[elegido]

    tabla = pd.DataFrame({
        "campo": CONTRATO_ENTRADA,
        "valor": ["" if pd.isna(fila[c]) else str(fila[c]) for c in CONTRATO_ENTRADA],
    })
    editada = st.data_editor(tabla, disabled=["campo"], hide_index=True, key=f"editor_{elegido}")

    if st.button("Calcular riesgo", type="primary"):
        try:
            pedido = {r.campo: a_json(r.valor, r.campo) for r in editada.itertuples()}
        except ValueError:
            st.error("Algún campo numérico tiene texto. Revisa n_items, precios, peso, fotos y cuotas.")
            return
        try:
            with st.spinner("Consultando la API… si estaba dormida, la primera respuesta puede tardar hasta un minuto."):
                r = llamar_api("/riesgo", "post", json=pedido)
        except requests.exceptions.RequestException:
            st.error("La API no responde en este momento. Revisa la URL en la barra lateral "
                     "o espera un minuto (Render apaga los servicios gratuitos sin visitas) y vuelve a intentar.")
            return
        if r.status_code == 422:
            st.warning("La API rechazó el pedido: falta un campo obligatorio o hay un valor inválido "
                       "(por ejemplo, un precio negativo o una fecha mal escrita).")
            with st.expander("Detalle de la validación"):
                st.json(r.json())
            return
        if r.status_code != 200:
            st.error(f"La API respondió con un error ({r.status_code}). Intenta de nuevo en unos segundos.")
            return

        res = r.json()
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de 1–2 estrellas", f"{res['probabilidad']:.0%}")
        c2.metric("Umbral de riesgo alto (10% más riesgoso)", f"{res['umbral']:.0%}")
        if res["riesgo_alto"]:
            st.error("Riesgo alto: contactar al cliente")
        else:
            st.success("Riesgo bajo: no es prioridad llamar")
        if "mala_resena" in fila.index and not pd.isna(fila["mala_resena"]):
            real = "1–2 estrellas" if int(fila["mala_resena"]) == 1 else "3–5 estrellas"
            st.caption(f"Reseña real de este pedido (solo como referencia): {real}")
        with st.expander("Respuesta completa de la API"):
            st.json(res)

    with st.expander("¿Qué tan bien funciona el modelo?"):
        try:
            m = leer_json("metricas_modelo.json")
        except FileNotFoundError:
            st.info("No se encontró metricas_modelo.json.")
        else:
            nombres = {"azar": "Azar", "regla_retraso": "Regla: ordenar por retraso",
                       "regresion_logistica": "Regresión logística", "xgboost": "XGBoost (modelo usado)",
                       "red_neuronal": "Red neuronal"}
            comp = pd.DataFrame([{"método": nombres[k], "precisión en el 10% de mayor riesgo": m[k]}
                                 for k in nombres if k in m])
            st.dataframe(comp, hide_index=True)
            st.caption(f"Validación por fecha: entrenado con {m['n_entrenamiento']:,} pedidos y "
                       f"evaluado con los {m['n_validacion']:,} más recientes.")


# ---------- página 2: comentarios ----------
def pagina_comentarios():
    st.title("Qué dicen los comentarios")
    st.caption("Sentimiento de una muestra de comentarios con un modelo preentrenado "
               "(cardiffnlp/twitter-xlm-roberta-base-sentiment), comparado con las estrellas.")

    m = leer_csv("sentimiento_muestra.csv")
    orden = ["negative", "neutral", "positive"]
    coincide = (m.estrellas_grupo == m.sentimiento)

    c1, c2, c3 = st.columns(3)
    c1.metric("Comentarios analizados", f"{len(m):,}")
    c2.metric("Coincidencia con las estrellas", f"{coincide.mean():.1%}")
    c3.metric("Desacuerdos", f"{(~coincide).sum():,}")

    tabla = (pd.crosstab(m.estrellas_grupo, m.sentimiento)
             .reindex(index=orden, columns=orden, fill_value=0))
    tabla.index.name = "estrellas (1–2 / 3 / 4–5)"
    tabla.columns.name = "sentimiento del modelo"
    izq, der = st.columns(2)
    izq.markdown("**Conteos**")
    izq.dataframe(tabla)
    der.markdown("**% por fila**")
    der.dataframe((tabla.div(tabla.sum(axis=1), axis=0) * 100).round(1))

    st.subheader("Explorar comentarios")
    f1, f2, f3 = st.columns(3)
    estrellas = f1.multiselect("Estrellas", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    sentimientos = f2.multiselect("Sentimiento del modelo", orden, default=orden)
    solo_desacuerdos = f3.checkbox("Solo desacuerdos", value=True)
    vista = m[m.review_score.isin(estrellas) & m.sentimiento.isin(sentimientos)]
    if solo_desacuerdos:
        vista = vista[vista.estrellas_grupo != vista.sentimiento]
    st.caption(f"{len(vista):,} comentarios")
    st.dataframe(vista[["review_score", "sentimiento", "confianza", "review_comment_message"]]
                 .rename(columns={"review_score": "estrellas", "review_comment_message": "comentario"}),
                 hide_index=True)

    st.markdown(
        "**Lo que agrega el texto:** las estrellas dicen *cuánto* le gustó la compra; el texto dice *por qué* "
        "(entrega no recibida, producto equivocado, tamaño incorrecto) y deja ver calificaciones que "
        "contradicen lo escrito. La mayor confusión del modelo está en la clase neutral."
    )


# ---------- página 3: pronóstico ----------
def pagina_pronostico():
    st.title("Pronóstico de pedidos diarios · junio 2018")
    pron = leer_csv("pronostico_junio.csv")
    hist = leer_csv("serie_diaria.csv")
    res = leer_json("pronostico_resumen.json")

    c1, c2, c3 = st.columns(3)
    c1.metric("Promedio diario pronosticado", f"{res['promedio_junio']:.0f} pedidos")
    c2.metric("Error medio absoluto (validación)", f"{res['mae_modelo']:.1f}",
              delta=f"{res['mae_modelo'] - res['mae_ingenuo']:.1f} vs. ingenuo semanal", delta_color="inverse")
    c3.metric("Cobertura del intervalo 80% (validación)", f"{res['cobertura']:.0%}")

    historia = (alt.Chart(hist).mark_line(color="#8a8a8a")
                .encode(x=alt.X("fecha:T", title=None), y=alt.Y("pedidos:Q", title="Pedidos por día")))
    banda = (alt.Chart(pron).mark_area(opacity=0.25, color="#2a7ab0")
             .encode(x="fecha:T", y="bajo:Q", y2="alto:Q"))
    linea = alt.Chart(pron).mark_line(color="#2a7ab0").encode(x="fecha:T", y="pronostico:Q")
    st.altair_chart(historia + banda + linea, use_container_width=True)
    st.caption("Gris: pedidos reales desde enero de 2018. Azul: pronóstico de junio con su intervalo de 80%.")

    st.markdown(
        f"""
**Método:** nivel promedio de las últimas 10 semanas normales × un factor por día de la semana.
El intervalo multiplica el pronóstico por los percentiles 10 y 90 del cociente real/pronóstico
en la validación ({res['q10']:.2f} y {res['q90']:.2f}).

**Supuestos:** la caída de pedidos desde el {res['inicio_periodo_excluido']} se trató como un
periodo anómalo y se excluyó; se supone que junio vuelve al nivel de marzo a mayo.
"""
    )
    st.dataframe(pron, hide_index=True)


# ---------- página 4: ficha ----------
def pagina_ficha():
    ruta = BASE / "ficha_modelo.md"
    if ruta.exists():
        st.markdown(ruta.read_text(encoding="utf-8"))
    else:
        st.info("No se encontró ficha_modelo.md en la carpeta de la app.")


# ---------- navegación ----------
with st.sidebar:
    st.text_input("URL de la API", key="api_url")
    if st.button("Probar conexión"):
        try:
            with st.spinner("Despertando la API…"):
                r = llamar_api("/")
            if r.ok:
                st.success("La API responde.")
            else:
                st.error(f"La API respondió con error {r.status_code}.")
        except requests.exceptions.RequestException:
            st.error("La API no responde. Revisa la URL o espera un minuto.")

pagina = st.navigation([
    st.Page(pagina_riesgo, title="Riesgo de un pedido", icon=":material/warning:", default=True),
    st.Page(pagina_comentarios, title="Comentarios", icon=":material/chat:"),
    st.Page(pagina_pronostico, title="Pronóstico", icon=":material/show_chart:"),
    st.Page(pagina_ficha, title="Ficha del modelo", icon=":material/description:"),
])
pagina.run()
