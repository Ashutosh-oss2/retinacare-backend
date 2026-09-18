import os
from typing import Any
from app.utils.logging import logger

def load_aptos_keras_model(weights_path: str) -> Any:
    """Loads the pre-trained EfficientNetB0 Keras model."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"APTOS model weights not found at: {weights_path}")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(weights_path, compile=False)
    except ImportError:
        import keras
        model = keras.models.load_model(weights_path, compile=False)

    logger.info(f"Loaded APTOS Keras model from: {weights_path}")
    return model
