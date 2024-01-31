from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from progress_monitor import log_info

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print various performance metrics.
    :param model: Trained TensorFlow model.
    :param X_test: Test features.
    :param y_test: True labels for test data.
    """
    # Model Prediction
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Calculate Metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    # Log the Metrics
    log_info(f'Accuracy: {accuracy}')
    log_info(f'Precision: {precision}')
    log_info(f'Recall: {recall}')
    log_info(f'F1 Score: {f1}')

    return accuracy, precision, recall, f1

def load_and_evaluate(model_path, X_test, y_test):
    """
    Load a saved model and evaluate it on the test data.
    :param model_path: Path to the saved model.
    :param X_test: Test features.
    :param y_test: True labels for test data.
    """
    log_info(f'Loading model from {model_path}')
    model = load_model(model_path)
    log_info('Model loaded successfully.')

    return evaluate_model(model, X_test, y_test)

