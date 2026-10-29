# Ficha del modelo · Riesgo de mala reseña (Olist)

**Equipo:** Miranda Deleon Diego Antonio, Navas Guzmán Kevin Roberto y Rivera Herrera Silvia Carolina
**Modelo:** XGBoost dentro de un pipeline de scikit-learn (`modelo_riesgo.joblib`)

## 1. Para qué sirve y para qué no

**Sirve para** ordenar los pedidos recién entregados según la probabilidad de que el cliente ponga 1 o 2
estrellas, para que atención al cliente llame primero al 10% más riesgoso (su capacidad de llamadas).

**No debe usarse para:**
- Evaluar, sancionar o premiar vendedores o transportistas.
- Decidir reembolsos, descuentos o negar el servicio a un cliente.
- Predecir antes de la entrega: el modelo necesita la fecha real de entrega.
- Tomar una decisión sobre un pedido sin que una persona revise el caso.

## 2. Datos

- **Origen:** datos públicos de Olist (Brasil), base de datos del curso, esquema `final`.
- **Periodo:** pedidos comprados entre el 2016-09-15 y el 2018-05-31.
- **Cantidad:** 77,304 pedidos, una fila por pedido. El 13.5% tuvo una mala reseña (1–2 estrellas).
- **Se excluyeron:** pedidos no entregados o sin fecha de entrega (no aplica llamar después de la entrega),
  pedidos sin reseña (no se sabe cómo terminaron) y los pedidos desde junio de 2018 (los guarda el docente
  para la evaluación).
- **No se usa** nada de la reseña (texto, fechas de la reseña): esa información aparece después de la predicción.

## 3. Qué tan bien funciona

Se entrenó con los pedidos más antiguos y se probó con los 15,461 más recientes
(del 2018-03-22 al 2018-05-31), como si fueran pedidos nuevos.

| Cómo se elige a quién llamar (10% de los pedidos) | De cada 10 llamadas, cuántos clientes iban a dejar 1–2 estrellas |
|---|---|
| Modelo (XGBoost) | 5.0 |
| Regla simple: llamar a los más retrasados | 3.9 |
| Al azar | 1.2 |

En palabras simples: **de cada 10 clientes que llamamos usando el modelo, unos 5 iban a dejar
una mala reseña**, frente a unos 4 con la regla del retraso y 1 al azar.
Un pedido se marca como "riesgo alto" si su probabilidad es de 26% o más.

## 4. Límites conocidos

- **Depende sobre todo del retraso.** Con retraso, el 48% de los pedidos de validación
  terminó en mala reseña; sin retraso, el 9%. En un periodo con pocos retrasos
  (en entrenamiento fueron el 8.9% de los pedidos) el modelo distingue peor.
- **No ve el producto ni el trato:** un producto dañado o equivocado que llega a tiempo casi no se detecta.
- **Cambios en el negocio:** a finales de mayo de 2018 hubo una caída anómala de pedidos (ver Parte 3);
  periodos así pueden cambiar el comportamiento de los clientes.
- Usa solo la categoría y el vendedor del primer artículo, y 1,263 pedidos no
  tienen categoría ("desconocido").
- El desempeño en junio–agosto de 2018 puede ser distinto al de la validación.

## 5. A quién afecta un error

- **Falso positivo** (llamamos a un cliente satisfecho): cuesta tiempo del equipo y puede molestar un poco
  al cliente. Con el modelo, cerca de 5 de cada 10 llamadas son de este tipo.
- **Falso negativo** (no llamamos a un cliente que se iba a quejar): se pierde la oportunidad de resolver
  el problema y llega la mala reseña, que daña al vendedor y a Olist.
- **Mal uso:** el modelo usa el estado del cliente y del vendedor. Usarlo para castigar vendedores por
  retrasos que causa el transportista, o para dar peor trato a clientes de ciertos estados, sería injusto.

## 6. Recomendación de uso

- Usarlo como **lista de prioridad diaria**: el equipo llama de arriba hacia abajo y una persona decide qué
  ofrecer en cada caso.
- No automatizar sanciones ni beneficios a partir del puntaje.
- **Revisarlo cada mes** con los pedidos que ya tienen reseña (precisión en el 10% de mayor riesgo frente a la
  regla del retraso) y **reentrenarlo cada trimestre**, o antes si la tasa de retrasos cambia mucho o si deja
  de superar a la regla.
- Ajustar el umbral si cambia la capacidad de llamadas del equipo.
