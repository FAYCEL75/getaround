from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
import traceback
import pandas as pd

# ============================================================
# MODELES Pydantic
# ============================================================

class CarFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: int
    has_gps: int
    has_air_conditioning: int
    automatic_car: int
    has_getaround_connect: int
    has_speed_regulator: int
    winter_tires: int


class PredictRequest(BaseModel):
    input: List[CarFeatures]


class PredictionResponse(BaseModel):
    prediction: List[float]


class ErrorResponse(BaseModel):
    error_type: str
    message: str


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="GetAround Pricing API",
    description="API de prédiction basée sur un pipeline sklearn (OneHotEncoder + Regr.).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CHARGEMENT MODELE
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

model = None
MODEL_ERROR: Optional[str] = None
MODEL_INFO: Optional[dict] = None

try:
    raw_obj = joblib.load(MODEL_PATH)

    # Cas 1 : on a sauvegardé un "bundle" dict (comme dans le notebook)
    if isinstance(raw_obj, dict) and "model" in raw_obj:
        model = raw_obj["model"]
        MODEL_INFO = {
            "wrapped": True,
            "bundle_keys": list(raw_obj.keys()),
            "inner_type": str(type(model)),
        }
    else:
        # Cas 2 : modèle direct (pipeline sklearn)
        model = raw_obj
        MODEL_INFO = {
            "wrapped": False,
            "bundle_keys": None,
            "inner_type": str(type(model)),
        }

    MODEL_ERROR = None

except Exception as e:
    MODEL_ERROR = str(e)
    model = None
    MODEL_INFO = None


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Bienvenue sur la GetAround Pricing API.",
        "usage": "Utilisez POST /predict pour obtenir une prédiction de prix.",
        "model_status": "loaded" if model is not None else "error",
    }


@app.get("/health", tags=["Health"])
def healthcheck():
    return {
        "status": "ok" if model is not None else "error",
        "model_path": MODEL_PATH,
        "model_error": MODEL_ERROR,
        "model_info": MODEL_INFO,
    }


@app.post(
    "/predict",
    tags=["Prediction"],
    response_model=PredictionResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Erreur interne modèle/pipeline",
        }
    },
)
def predict_price(payload: PredictRequest):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Modèle non chargé : {MODEL_ERROR}",
        )

    try:
        # On convertit en DataFrame avec les bons noms de colonnes
        data_dicts = [f.dict() for f in payload.input]
        df = pd.DataFrame(data_dicts)

        preds = model.predict(df)
        preds_list = [float(p) for p in preds]

        return PredictionResponse(prediction=preds_list)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur durant la prédiction : {str(e)}",
        )