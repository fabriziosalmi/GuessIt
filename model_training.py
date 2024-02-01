import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_string_features(column, column_name):
    """
    Extract features from a string column such as length,
    number of unique characters, and presence of special characters.
    """
    return pd.DataFrame({
        f'{column_name}_length': column.apply(len),
        f'{column_name}_unique_chars': column.apply(lambda x: len(set(x))),
        f'{column_name}_has_special': column.str.contains(r'[!@#$%^&*(),.?":{}|<>]', regex=True).astype(int)
    })

def load_and_prepare_data(file_path, nrows=None, skiprows=None):
    logging.info("Loading and preparing data...")
    # Only read specific rows if nrows and skiprows are provided for sampling
    data = pd.read_csv(file_path, nrows=nrows, skiprows=skiprows)

    # Feature engineering for Email and Password
    email_features = extract_string_features(data['Email'], 'email')
    password_features = extract_string_features(data['Password'], 'password')

    # Combine all features
    features = pd.concat([email_features, password_features], axis=1)

    # Replace 'target' with your actual target column
    target = data['YourActualTargetColumn']  # Corrected target column

    return features, target

def train_random_forest_model(features, target):
    logging.info("Training RandomForest model...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest Classifier with adjusted parameters
    model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)  # Adjusted for memory efficiency
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"RandomForest Model Accuracy: {accuracy:.2%}")

    return model

def main():
    file_path = 'preprocessed_data.csv'
    nrows = 10000  # Number of rows to load for a large dataset, adjust as needed
    skiprows = lambda x: x > 0 and np.random.rand() > 0.1  # Skip 90% of the data randomly

    try:
        # Load and prepare data
        features, target = load_and_prepare_data(file_path, nrows=nrows, skiprows=skiprows)

        # Train the model
        model = train_random_forest_model(features, target)

        # Save the model
        joblib.dump(model, 'random_forest_model.pkl')
        logging.info("Model saved as random_forest_model.pkl")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
