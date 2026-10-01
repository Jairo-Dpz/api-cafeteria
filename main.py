"""
API de segmentación de ventas — Proyecto Final Sistemas Inteligentes
Carga el modelo K-means entrenado en el cuaderno y expone un endpoint
para predecir a qué clúster pertenece una venta nueva.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

# --- Carga de artefactos (una sola vez, al iniciar el servidor) ---
scaler = joblib.load(BASE_DIR / "scaler.pkl")
kmeans = joblib.load(BASE_DIR / "kmeans_model.pkl")

with open(BASE_DIR / "config_modelo.json", "r", encoding="utf-8") as f:
    config = json.load(f)

FEATURES = config["features"]  # ["Monto_Total", "Cantidad_Items", "Hora_Dia", "Es_FinDeSemana"]
NOMBRES_CLUSTER = {int(k): v for k, v in config["nombres_cluster"].items()}

app = FastAPI(
    title="API de Segmentación de Ventas",
    description="Predice el perfil de consumo (clúster K-means) de una venta.",
    version="1.0.0",
)


# --- Esquemas de entrada y salida ---
class VentaEntrada(BaseModel):
    monto_total: float = Field(..., gt=0, description="Monto total de la boleta en soles")
    cantidad_items: int = Field(..., gt=0, description="Cantidad de productos del pedido")
    hora_dia: int = Field(..., ge=0, le=23, description="Hora del pedido, formato 24 h")
    dia_semana: int = Field(..., ge=1, le=7, description="Día de la semana: 1=lunes ... 7=domingo")

    class Config:
        json_schema_extra = {
            "example": {
                "monto_total": 85.0,
                "cantidad_items": 4,
                "hora_dia": 21,
                "dia_semana": 6,
            }
        }


class PrediccionSalida(BaseModel):
    cluster: int
    nombre_cluster: str


# --- Endpoints ---
@app.get("/")
def raiz():
    return {"mensaje": "API activa. Usa POST /predecir para clasificar una venta."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predecir", response_model=PrediccionSalida)
def predecir(venta: VentaEntrada):
    if venta.hora_dia < 7 or venta.hora_dia > 22:
        raise HTTPException(
            status_code=400,
            detail="El local solo atiende entre las 7:00 y las 22:59.",
        )

    es_fin_de_semana = int(venta.dia_semana >= 6)

    fila = pd.DataFrame([{
        "Monto_Total": venta.monto_total,
        "Cantidad_Items": venta.cantidad_items,
        "Hora_Dia": venta.hora_dia,
        "Es_FinDeSemana": es_fin_de_semana,
    }])[FEATURES]  # mismo orden de columnas que en el entrenamiento

    fila_escalada = scaler.transform(fila)
    cluster = int(kmeans.predict(fila_escalada)[0])
    nombre = NOMBRES_CLUSTER.get(cluster, f"Clúster {cluster}")

    return PrediccionSalida(cluster=cluster, nombre_cluster=nombre)
