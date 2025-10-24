import numpy as np
import yaml
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_custom_model(params):
    """Create a custom CNN that works with 1-channel images"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(52, 52, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(params['model']['dropout_rate']),
        Dense(128, activation='relu'),
        Dropout(params['model']['dropout_rate']),
        Dense(params['model']['num_classes'], activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=params['training']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    params = load_params()
    
    # Load processed data
    X_train = np.load('data/processed/train_images.npy')
    y_train = np.load('data/processed/train_labels.npy')
    X_val = np.load('data/processed/val_images.npy')
    y_val = np.load('data/processed/val_labels.npy')
    
    print(f"Training on {X_train.shape[0]} samples")
    print(f"Validating on {X_val.shape[0]} samples")
    
    # Use only 1 channel instead of 3
    if X_train.shape[-1] == 3:
        X_train = X_train[:, :, :, 0:1]  # Keep only first channel
        X_val = X_val[:, :, :, 0:1]
        print(f"Converted to 1 channel: {X_train.shape}")
    
    # Update input shape for 1 channel
    params['model']['input_shape'] = [52, 52, 1]
    
    # Create model
    model = create_custom_model(params)
    
    print("Model architecture (custom CNN for 1-channel images):")
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=params['training']['patience'],
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("Starting training with custom 1-channel CNN...")
    history = model.fit(
        X_train, y_train,
        batch_size=params['training']['batch_size'],
        epochs=params['training']['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training metrics
    training_metrics = {
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss']))
    }
    
    with open('metrics/training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print("Training completed!")

if __name__ == "__main__":
    main()