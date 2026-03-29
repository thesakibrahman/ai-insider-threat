import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim: int) -> models.Model:
    """
    Builds an Autoencoder neural network model for anomaly detection.
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(16, activation='relu')(input_layer)
    encoder = layers.Dense(8, activation='relu')(encoder)
    
    # Bottleneck
    bottleneck = layers.Dense(4, activation='relu')(encoder)
    
    # Decoder
    decoder = layers.Dense(8, activation='relu')(bottleneck)
    decoder = layers.Dense(16, activation='relu')(decoder)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(decoder)
    
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X: pd.DataFrame, existing_model=None, epochs=20, batch_size=32, X_val=None) -> models.Model:
    """
    Trains or updates the autoencoder model incrementally (continual learning).

    Args:
        X:              Training feature matrix (70% split).
        existing_model: If provided, continues training this model (continual learning).
        epochs:         Number of training epochs.
        batch_size:     Mini-batch size.
        X_val:          Validation feature matrix (15% split from paper).
                        When provided, used as explicit Keras validation_data.
                        When None, falls back to internal validation_split=0.1.
    """
    X_min = X.min()
    X_max = X.max()
    
    if existing_model is not None:
        model = existing_model
        # Continual learning: update scaling bounds to cover new data range
        X_min = np.minimum(model.X_min, X_min)
        X_max = np.maximum(model.X_max, X_max)
    else:
        model = build_autoencoder(X.shape[1])
        
    # Normalizing inputs between 0 and 1 before training
    X_scaled = (X - X_min) / (X_max - X_min + 1e-9)

    # Prepare validation data for Keras
    # Paper states: 70% train, 15% validation, 15% test
    if X_val is not None and len(X_val) > 0:
        # Scale val using TRAIN stats (no data leakage)
        X_val_scaled = (X_val - X_min) / (X_max - X_min + 1e-9)
        validation_kwargs = {'validation_data': (X_val_scaled, X_val_scaled)}
    else:
        # Fallback: internal 10% from training (used in demo/simulator mode)
        validation_kwargs = {'validation_split': 0.1}

    # Autoencoder trains to reconstruct its own input
    # Anomaly Score S(x) = ||x - x̂||² (as defined in paper Eq. 1)
    model.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        **validation_kwargs
    )
    
    # Attach scaling metadata to the model object for inference
    model.X_min = X_min
    model.X_max = X_max
    return model

def predict_autoencoder(model: models.Model, X: pd.DataFrame) -> np.ndarray:
    """
    Computes reconstruction error (MSE) as anomaly score.
    Higher error = more anomalous.
    """
    X_scaled = (X - model.X_min) / (model.X_max - model.X_min + 1e-9)
    X_pred = model.predict(X_scaled, verbose=0)
    
    # Calculate Mean Squared Error per sample
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    
    # Normalize between 0 and 1
    min_mse, max_mse = mse.min(), mse.max()
    if max_mse > min_mse:
            normalized_error = (mse - min_mse) / (max_mse - min_mse)
    else:
        normalized_error = mse * 0.0
        
    return normalized_error
