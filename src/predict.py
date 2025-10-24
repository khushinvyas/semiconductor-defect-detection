import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import logging
import os
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiconductorDefectPredictor:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the predictor with the trained model
        
        Args:
            model_path (str, optional): Path to the model file. If None, will use default path
        """
        self.model = None
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        if model_path is None:
            model_path = os.path.join(self.project_root, 'models', 'best_model.h5')
            
        self.model_path = model_path
        self.load_model()
        
        # Defect class names based on your 8-class multi-label problem
        self.defect_classes = [
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Random", "Scratch", "Near-full"
        ]
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Load the model with custom_objects if needed
            self.model = load_model(self.model_path)
            
            # Verify model was loaded correctly
            if self.model is None:
                raise ValueError("Model failed to load properly")
                
            # Log model information
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess the wafer map image for prediction
        
        Args:
            image_array: Input image array (any size, will be resized to 52x52)
            
        Returns:
            Preprocessed image ready for model prediction
        """
        try:
            logger.info(f"Input image shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            # Convert to float32 if needed
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32)
            
            # Ensure values are in range 0-3
            if image_array.max() > 3:
                image_array = (image_array / 255.0) * 3
            
            # Ensure the image is the right shape
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
            elif len(image_array.shape) == 3 and image_array.shape[-1] == 3:
                # Convert RGB to grayscale using mean
                image_array = np.mean(image_array, axis=-1, keepdims=True)
            
            # Normalize to 0-1 range
            image_array = image_array / 3.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.info(f"Preprocessed image shape: {image_array.shape}, range: [{image_array.min():.3f}, {image_array.max():.3f}]")
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_array: np.ndarray) -> Dict:
        """
        Predict defects in a wafer map image
        
        Args:
            image_array: Preprocessed wafer map image
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_array)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            binary_predictions = (predictions > 0.5).astype(int)[0]
            
            # Get confident predictions (above threshold)
            confident_defects = []
            confidence_scores = []
            
            for i, (pred, binary) in enumerate(zip(predictions[0], binary_predictions)):
                if binary == 1:
                    confident_defects.append(self.defect_classes[i])
                    confidence_scores.append(float(pred))
            
            # Prepare results
            results = {
                'defects_detected': confident_defects,
                'confidence_scores': confidence_scores,
                'all_predictions': {
                    self.defect_classes[i]: float(pred) 
                    for i, pred in enumerate(predictions[0])
                },
                'binary_predictions': binary_predictions.tolist(),
                'has_defects': len(confident_defects) > 0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'error': str(e)}
    
    def predict_from_file(self, file_path: str) -> Dict:
        """Load image from file and predict"""
        try:
            # For numpy arrays (your wafer maps)
            if file_path.endswith('.npy'):
                image_array = np.load(file_path)
            else:
                # For image files (if you want to support PNG/JPG)
                from PIL import Image
                image = Image.open(file_path).convert('L')  # Convert to grayscale
                image_array = np.array(image)
            
            return self.predict(image_array)
            
        except Exception as e:
            logger.error(f"Error loading image from file: {e}")
            return {'error': str(e)}

# Global predictor instance
_predictor = None

def init_predictor(model_path: Optional[str] = None) -> SemiconductorDefectPredictor:
    """Initialize the predictor (call this at app startup)
    
    Args:
        model_path (str, optional): Path to the model file
        
    Returns:
        SemiconductorDefectPredictor: Initialized predictor instance
    """
    global _predictor
    try:
        logger.info(f"Initializing predictor with model path: {model_path}")
        _predictor = SemiconductorDefectPredictor(model_path)
        logger.info("Predictor initialized successfully")
        return _predictor
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise

def get_predictions(image_data: np.ndarray) -> Dict:
    """Get predictions for image data
    
    Args:
        image_data (np.ndarray): Input image to predict on
        
    Returns:
        Dict: Prediction results
    """
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call init_predictor first.")
    return _predictor.predict(image_data)

# Example usage
if __name__ == "__main__":
    try:
        # Test the predictor
        predictor = init_predictor()
        
        # Create a sample wafer map for testing
        sample_image = np.random.randint(0, 4, size=(52, 52), dtype=np.int32)
        
        results = predictor.predict(sample_image)
        print("Prediction Results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise