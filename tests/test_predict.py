import pytest
import numpy as np
from src.predict import SemiconductorDefectPredictor

def test_preprocess_image():
    """Test image preprocessing functionality"""
    # Create a dummy image
    test_image = np.random.randint(0, 4, size=(52, 52), dtype=np.int32)
    
    # Initialize predictor
    predictor = SemiconductorDefectPredictor()
    
    # Process image
    processed_image = predictor.preprocess_image(test_image)
    
    # Check output shape
    assert processed_image.shape == (1, 52, 52, 1), "Incorrect output shape"
    
    # Check normalization
    assert np.all(processed_image >= 0) and np.all(processed_image <= 1), "Values not normalized to [0,1]"

def test_predict():
    """Test prediction functionality"""
    # Create a dummy image
    test_image = np.random.randint(0, 4, size=(52, 52), dtype=np.int32)
    
    # Initialize predictor
    predictor = SemiconductorDefectPredictor()
    
    # Get predictions
    results = predictor.predict(test_image)
    
    # Check results structure
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'defects_detected' in results, "Missing 'defects_detected' in results"
    assert 'confidence_scores' in results, "Missing 'confidence_scores' in results"
    assert 'all_predictions' in results, "Missing 'all_predictions' in results"
    assert 'binary_predictions' in results, "Missing 'binary_predictions' in results"
    
    # Check predictions format
    assert len(results['binary_predictions']) == 8, "Should have 8 class predictions"
    assert all(isinstance(x, (int, np.integer)) for x in results['binary_predictions']), \
        "Binary predictions should be integers"
    assert all(0 <= x <= 1 for x in results['confidence_scores']), \
        "Confidence scores should be between 0 and 1"