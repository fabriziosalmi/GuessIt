import tensorflow as tf
from progress_monitor import setup_progress_monitor, update_progress, close_progress, log_info
from data_preprocessing import load_data, preprocess_data, split_data
import model_architecture  # Assuming this module contains the model architecture

def train_model(train_data, val_data):
    """
    Train the model with the given data.
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    """
    log_info("Starting model training.")
    
    # Initialize model
    model = model_architecture.create_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Progress bar for training
    progress_bar = setup_progress_monitor(len(train_data))

    # Training the model
    history = model.fit(train_data, validation_data=val_data, epochs=10,
                        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_progress(progress_bar, 1))])

    close_progress(progress_bar)
    log_info("Model training completed.")
    
    return model, history

def evaluate_model(model, test_data):
    """
    Evaluate the trained model on the test dataset.
    :param model: Trained model.
    :param test_data: Test dataset.
    """
    log_info("Starting model evaluation.")
    evaluation_metrics = model.evaluate(test_data)
    log_info(f"Evaluation Results - Loss: {evaluation_metrics[0]}, Accuracy: {evaluation_metrics[1]}")
    
    return evaluation_metrics

def main():
    file_path = 'dataset.csv'
    data = load_data(file_path)
    preprocessed_data = preprocess_data(data)
    train_data, test_data = split_data(preprocessed_data)

    model, history = train_model(train_data, test_data)
    evaluation_metrics = evaluate_model(model, test_data)

    # Optionally, save the trained model
    model.save('path/to/save/model')

if __name__ == "__main__":
    main()
