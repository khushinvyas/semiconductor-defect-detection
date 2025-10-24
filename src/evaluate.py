import numpy as np
import yaml
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def main():
    params = load_params()
    
    # Load test data
    X_test = np.load('data/processed/test_images.npy')
    y_test = np.load('data/processed/test_labels.npy')
    label_classes = np.load('data/processed/label_classes.npy')
    
    print(f"Test data loaded: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")
    
    # Handle 1-channel input (convert from 3 channels if needed)
    if X_test.shape[-1] == 3:
        # Use only the first channel (convert from 3 channels to 1 channel)
        X_test = X_test[:, :, :, 0:1]
        print(f"Converted to 1 channel: {X_test.shape}")
    
    # Load model
    try:
        model = load_model('models/best_model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"Predictions shape: {y_pred.shape}")
    
    # Calculate metrics for multi-label classification
    print("Evaluating model...")
    try:
        # For models without precision/recall in metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_precision = 0.0
        test_recall = 0.0
    except:
        # If the model was compiled with additional metrics
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Multi-label specific metrics
    hamming = hamming_loss(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
    
    # Classification report for each label
    clf_report = classification_report(
        y_test, 
        y_pred,
        target_names=[f'Class_{i}' for i in range(len(label_classes))],
        output_dict=True,
        zero_division=0
    )
    
    # Calculate additional per-class metrics
    per_class_metrics = {}
    for i in range(len(label_classes)):
        class_name = f'Class_{i}'
        if class_name in clf_report:
            per_class_metrics[class_name] = {
                'precision': clf_report[class_name]['precision'],
                'recall': clf_report[class_name]['recall'],
                'f1_score': clf_report[class_name]['f1-score'],
                'support': clf_report[class_name]['support']
            }
    
    # Save metrics
    test_metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'hamming_loss': float(hamming),
        'jaccard_score': float(jaccard),
        'macro_avg_f1': float(clf_report['macro avg']['f1-score']),
        'weighted_avg_f1': float(clf_report['weighted avg']['f1-score'])
    }
    
    # Create directories
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Convert numpy types before saving
    clf_report = convert_numpy_types(clf_report)
    test_metrics = convert_numpy_types(test_metrics)
    per_class_metrics = convert_numpy_types(per_class_metrics)
    
    # Save all metrics
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    with open('plots/classification_report.json', 'w') as f:
        json.dump(clf_report, f, indent=2)
    
    with open('plots/per_class_metrics.json', 'w') as f:
        json.dump(per_class_metrics, f, indent=2)
    
    # Create multi-label metrics visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Precision and Recall per Class
    plt.subplot(2, 2, 1)
    precision_per_class = [per_class_metrics[f'Class_{i}']['precision'] for i in range(len(label_classes))]
    recall_per_class = [per_class_metrics[f'Class_{i}']['recall'] for i in range(len(label_classes))]
    
    x = np.arange(len(label_classes))
    width = 0.35
    
    plt.bar(x - width/2, precision_per_class, width, label='Precision', alpha=0.7)
    plt.bar(x + width/2, recall_per_class, width, label='Recall', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision and Recall per Class')
    plt.xticks(x, [f'{i}' for i in range(len(label_classes))])
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: F1-Score per Class
    plt.subplot(2, 2, 2)
    f1_per_class = [per_class_metrics[f'Class_{i}']['f1_score'] for i in range(len(label_classes))]
    
    plt.bar(x, f1_per_class, alpha=0.7, color='green')
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.title('F1-Score per Class')
    plt.xticks(x, [f'{i}' for i in range(len(label_classes))])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Support (Number of samples) per Class
    plt.subplot(2, 2, 3)
    support_per_class = [per_class_metrics[f'Class_{i}']['support'] for i in range(len(label_classes))]
    
    plt.bar(x, support_per_class, alpha=0.7, color='orange')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Test Set')
    plt.xticks(x, [f'{i}' for i in range(len(label_classes))])
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Overall metrics
    plt.subplot(2, 2, 4)
    overall_metrics = ['Accuracy', 'Jaccard', 'Hamming Loss']
    overall_values = [test_accuracy, jaccard, hamming]
    colors = ['blue', 'green', 'red']
    
    bars = plt.bar(overall_metrics, overall_values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Overall Model Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/multi_label_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix heatmap for multi-label (showing co-occurrence)
    plt.figure(figsize=(12, 10))
    
    # Calculate label co-occurrence matrix
    cooccurrence = np.zeros((len(label_classes), len(label_classes)))
    for i in range(len(label_classes)):
        for j in range(len(label_classes)):
            cooccurrence[i, j] = np.sum((y_test[:, i] == 1) & (y_test[:, j] == 1))
    
    # Normalize by row for better visualization
    cooccurrence_norm = cooccurrence / np.sum(cooccurrence, axis=1, keepdims=True)
    cooccurrence_norm = np.nan_to_num(cooccurrence_norm)
    
    sns.heatmap(cooccurrence_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'Class {i}' for i in range(len(label_classes))],
                yticklabels=[f'Class {i}' for i in range(len(label_classes))])
    plt.title('Label Co-occurrence Matrix (Normalized)')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig('plots/cooccurrence_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("MULTI-LABEL EVALUATION COMPLETED!")
    print("="*50)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Macro Avg F1: {clf_report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {clf_report['weighted avg']['f1-score']:.4f}")
    print("\nPer-class metrics saved to: plots/per_class_metrics.json")
    print("Visualizations saved to: plots/")
    print("="*50)

if __name__ == "__main__":
    main()