from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np

# Load the saved model
MODEL_PATH = "best_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define FastAPI app
app = FastAPI(title="Digit Recognition API")

# Define input data schema: exactly 64 float values (8x8 image flattened)
class DigitFeatures(BaseModel):
    features: list[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 64
            }
        }

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_features

    @staticmethod
    def validate_features(values):
        if len(values) != 64:
            raise ValueError("features must contain exactly 64 float values")
        return values

@app.post("/predict")
async def predict_digit(input_data: DigitFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(data)[0]

    return {"prediction": int(prediction)}