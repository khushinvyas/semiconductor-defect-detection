import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    
    # Load dataset
    data = np.load('data/raw/MixedWM38.npz')
    images = data['arr_0']  # Wafer map images
    labels = data['arr_1']  # Multi-label binary labels
    
    print(f"Dataset loaded:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels are multi-label: {np.any(np.sum(labels, axis=1) > 1)}")
    
    # Reshape images for CNN
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    
    # Convert to 3 channels
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    
    # CORRECT NORMALIZATION - based on actual data range [0,3]
    if params['preprocessing']['normalize']:
        images = images.astype('float32') / 3.0  # Normalize by max value 3
        print("Normalized images to [0, 1] range (divided by 3)")
    
    print(f"Final images shape: {images.shape}")
    print(f"Final labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # For multi-label, we can't use standard stratification
    # Use random split or create a custom stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, 
        test_size=params['data']['test_size'] + params['data']['val_size'],
        random_state=params['data']['random_state']
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=params['data']['test_size'] / (params['data']['test_size'] + params['data']['val_size']),
        random_state=params['data']['random_state']
    )
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Val set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    np.save('data/processed/train_images.npy', X_train)
    np.save('data/processed/train_labels.npy', y_train)
    np.save('data/processed/val_images.npy', X_val)
    np.save('data/processed/val_labels.npy', y_val)
    np.save('data/processed/test_images.npy', X_test)
    np.save('data/processed/test_labels.npy', y_test)
    
    # Save label information
    np.save('data/processed/label_classes.npy', np.arange(labels.shape[1]))
    

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()