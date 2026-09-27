from tensorflow.keras.models import load_model
from fastapi import HTTPException
import os

model_path = os.path.join(os.path.dirname(__file__), "resnet_model.h5")
model = load_model(model_path) if os.path.exists(model_path) else None


def require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="ResNet model is not installed")
