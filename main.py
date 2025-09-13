from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Load the model with error handling
model_path = os.path.join("..", "model", "eta_model.pkl")
try:
    model = joblib.load("model/eta_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

# Define request body schema
class ETARequest(BaseModel):
    start_lat: float
    start_lon: float
    dest_lat: float
    dest_lon: float


# Create FastAPI app
app = FastAPI(title="ETA Prediction API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ETA Prediction API. Use /predict_eta with POST to get predictions."}

@app.post("/predict_eta")
async def predict_eta(data: ETARequest) -> dict:
    # Prepare features for prediction
    features = [[data.start_lat, data.start_lon, data.dest_lat, data.dest_lon]]
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"eta_minutes": round(prediction, 2)}
