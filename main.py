from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
from loguru import logger
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pickle
import os

# Mapping for Breast Cancer classes
CANCER_CLASSES = {
    0: "maligno",
    1: "benigno"
}

# Configure logging
logger.add("api.log", rotation="500 MB", level="INFO")

app = FastAPI(
    title="Classification Model API",
    description="API para realizar predicciones con un modelo de clasificación",
    version="1.0.0"
)

# Load and train a simple model for demonstration
def train_model():
    X, y = load_breast_cancer(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Save model
model = train_model()
if not os.path.exists('models'):
    os.makedirs('models')
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: str
    probability: List[float]

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        logger.info(f"Received prediction request with features: {input_data.features}")
        
        # Convert input features to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        
        # Convert numeric prediction to class name
        prediction_class = CANCER_CLASSES[prediction]
        
        logger.info(f"Prediction successful: {prediction_class}")
        return PredictionOutput(prediction=prediction_class, probability=probabilities)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)