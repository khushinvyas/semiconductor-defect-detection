from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import json
import logging
import os
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import os
from botocore.exceptions import ClientError

def download_model_from_s3():
    """Download model from S3 if not exists locally"""
    if not os.path.exists('models/best_model.h5'):
        os.makedirs('models', exist_ok=True)
        s3 = boto3.client('s3')
        try:
            s3.download_file(
                os.environ.get('S3_BUCKET', 'semiconductor-defect-mlops'),
                'models/best_model.h5',
                'models/best_model.h5'
            )
            logger.info("Model downloaded from S3")
        except ClientError as e:
            logger.error(f"Error downloading model from S3: {e}")
            raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, 'templates')

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Templates directory: {TEMPLATES_DIR}")
logger.info(f"Templates exists: {os.path.exists(TEMPLATES_DIR)}")
logger.info(f"Index.html exists: {os.path.exists(os.path.join(TEMPLATES_DIR, 'index.html'))}")

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import predictor functions
from predict import init_predictor, get_predictions

# Initialize predictor as None
predictor = None

def initialize_app() -> bool:
    """Initialize the model when the app starts
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global predictor
    try:
        # Get absolute path to model
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.h5')
        logger.info(f"Looking for model at: {model_path}")
        
        # Check if model exists, download if needed
        if not os.path.exists(model_path):
            logger.info("Model not found, attempting to download from S3...")
            try:
                download_model_from_s3()
            except Exception as e:
                logger.error(f"Failed to download model from S3: {e}")
                return False
        
        # Verify model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Initialize the predictor
        predictor = init_predictor(model_path)
        logger.info("Model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

# Initialize the app
initialize_app()

@app.route('/')
def home():
    """Render the home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template rendering error: {e}")
        return f"""
        <html>
            <body>
                <h1>Template Error</h1>
                <p>Error: {e}</p>
                <p>Project root: {PROJECT_ROOT}</p>
                <p>Templates dir: {TEMPLATES_DIR}</p>
                <p>Templates exists: {os.path.exists(TEMPLATES_DIR)}</p>
            </body>
        </html>
        """

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'templates_loaded': os.path.exists(TEMPLATES_DIR),
        'message': 'Semiconductor Defect Detection API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for defect prediction"""
    try:
        if predictor is None:
            if not initialize_app():
                return jsonify({'error': 'Model not loaded', 'details': 'Model initialization failed'}), 500
        
        # Check if image file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'details': 'Please select a file to upload'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'details': 'The selected file has no name'}), 400
        
        # Check if JSON data is provided (for direct array input)
        if request.content_type == 'application/json':
            data = request.get_json()
            if 'image_array' in data:
                try:
                    image_array = np.array(data['image_array'])
                    logger.info(f"Received image array with shape: {image_array.shape}")
                    results = get_predictions(image_array)
                    return jsonify(results)
                except Exception as e:
                    logger.error(f"Error processing JSON input: {e}")
                    return jsonify({'error': 'Invalid image data', 'details': str(e)}), 400
        
        # Handle file upload
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                logger.info(f"Processing file: {filename}")
                
                try:
                    # Process based on file type
                    if filename.endswith('.npy'):
                        image_array = np.load(filepath)
                    else:
                        # Convert image to numpy array
                        from PIL import Image
                        img = Image.open(filepath).convert('L')  # Convert to grayscale
                        image_array = np.array(img, dtype=np.float32)
                        
                        # Normalize to 0-3 range (assuming this is what the model expects)
                        if image_array.max() > 3:
                            image_array = (image_array / 255.0) * 3
                    
                    # Validate image dimensions
                    logger.info(f"Image shape: {image_array.shape}")
                    if len(image_array.shape) not in [2, 3]:
                        raise ValueError(f"Invalid image dimensions: {image_array.shape}")
                    
                    # Resize if needed (assuming model expects 52x52)
                    if image_array.shape[:2] != (52, 52):
                        from PIL import Image
                        img = Image.fromarray(image_array.astype(np.uint8))
                        img = img.resize((52, 52), Image.Resampling.LANCZOS)
                        image_array = np.array(img, dtype=np.float32)
                        if image_array.max() > 3:
                            image_array = (image_array / 255.0) * 3
                        logger.info(f"Resized image to: {image_array.shape}")
                    
                    # Get predictions
                    results = get_predictions(image_array)
                    if results is None:
                        raise ValueError("Model returned no predictions")
                    
                    return jsonify(results)
                    
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    return jsonify({
                        'error': 'Image processing failed',
                        'details': str(e)
                    }), 400
                    
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            return jsonify({
                'error': 'Invalid file type',
                'details': f'Supported formats are: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample', methods=['GET'])
def get_sample_prediction():
    """Get prediction using a random sample from test dataset"""
    try:
        # Check if predictor is initialized
        if predictor is None:
            if not initialize_app():
                return jsonify({
                    'error': 'Model not initialized',
                    'details': 'Failed to initialize the model. Please check the logs.'
                }), 500
        
        # Load test data
        test_images_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_images.npy')
        test_labels_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_labels.npy')
        
        if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
            logger.error("Test data files not found")
            return jsonify({
                'error': 'Test data not found',
                'details': 'Test dataset files are missing'
            }), 404
            
        # Load test images and labels
        logger.info("Loading test dataset...")
        test_images = np.load(test_images_path)
        test_labels = np.load(test_labels_path)
        
        # Randomly select one test image
        idx = np.random.randint(0, len(test_images))
        test_image = test_images[idx]
        true_labels = test_labels[idx]
        
        logger.info(f"Selected test image index: {idx}")
        
        # Get predictions
        results = get_predictions(test_image)
        if not results:
            return jsonify({
                'error': 'Prediction failed',
                'details': 'Model returned no results'
            }), 500
        
        # Add ground truth to results
        results['true_labels'] = true_labels.tolist()
        results['sample_index'] = int(idx)
        
        # Add comparison between predicted and true labels
        matching_predictions = 0
        for pred, true in zip(results['binary_predictions'], true_labels):
            if pred == true:
                matching_predictions += 1
        
        results['accuracy'] = matching_predictions / len(true_labels)
        results['is_test_sample'] = True
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Sample prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_prediction():
    """Create visualization of prediction results"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.npy'):
            image_array = np.load(filepath)
        else:
            from PIL import Image
            img = Image.open(filepath).convert('L')
            image_array = np.array(img)
        
        # Get predictions
        results = get_predictions(image_array)
        
        # Create visualization
        is_test_sample = 'true_labels' in results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot wafer map
        im = ax1.imshow(image_array, cmap='viridis')
        ax1.set_title('Wafer Map')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, label='Defect Level')
        
        # Plot defect probabilities
        defects = list(results['all_predictions'].keys())
        probabilities = list(results['all_predictions'].values())
        
        # If this is a test sample, show both predicted and true values
        if is_test_sample:
            true_labels = results['true_labels']
            x = np.arange(len(defects))
            width = 0.35
            
            # Plot predicted probabilities
            rects1 = ax2.barh([i + width/2 for i in x], probabilities, width, 
                            label='Predicted', color=['red' if p > 0.5 else 'blue' for p in probabilities])
            
            # Plot true labels
            rects2 = ax2.barh([i - width/2 for i in x], true_labels, width,
                            label='True Labels', color=['green' if t == 1 else 'gray' for t in true_labels], alpha=0.7)
            
            ax2.set_xlabel('Probability / True Label')
            ax2.set_title(f'Predictions vs Ground Truth (Accuracy: {results["accuracy"]:.2%})')
            ax2.legend()
            
        else:
            # Original single bar plot for non-test samples
            colors = ['red' if prob > 0.5 else 'blue' for prob in probabilities]
            bars = ax2.barh(defects, probabilities, color=colors)
            ax2.set_xlabel('Probability')
            ax2.set_title('Defect Probabilities')
            
        ax2.set_xlim(0, 1)
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        
        # Add probability values on bars
        if is_test_sample:
            for i, (pred, true) in enumerate(zip(probabilities, true_labels)):
                ax2.text(pred + 0.01, i + width/2, f'{pred:.3f}', va='center')
                ax2.text(true + 0.01, i - width/2, f'{true:.0f}', va='center')
        
        plt.tight_layout()
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Return base64 encoded image
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        return jsonify({
            'visualization': f'data:image/png;base64,{img_base64}',
            'predictions': results
        })
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': str(e)}), 500

# Allowed file extensions
ALLOWED_EXTENSIONS = {'npy', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Run the app
    logger.info("Starting Flask application...")
    # In production, debug should be False
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)