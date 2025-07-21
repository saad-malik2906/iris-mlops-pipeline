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
app = FastAPI(title="Iris Species Prediction API")

# Define input data schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
async def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare data for prediction (shape: 1 sample, 4 features)
    data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    
    # Get prediction index
    pred_idx = model.predict(data)[0]

    # Map prediction index to class name
    target_names = ["setosa", "versicolor", "virginica"]
    prediction = target_names[pred_idx]

    return {"prediction": prediction}