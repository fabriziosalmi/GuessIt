import pandas as pd
from sklearn.model_selection import train_test_split
from progress_monitor import setup_progress_monitor, log_info, update_progress
import subword_tokenization

def load_data(file_path):
    """
    Load dataset from the given file path.
    :param file_path: Path to the dataset file.
    :return: Loaded dataset.
    """
    log_info("Loading data from: " + file_path)
    data = pd.read_csv(file_path)
    log_info("Data loaded successfully.")
    return data

def preprocess_data(data):
    """
    Preprocess the data.
    :param data: Raw dataset.
    :return: Preprocessed dataset.
    """
    log_info("Starting data preprocessing.")
    # Tokenization and other preprocessing steps here
    # Example: data['processed'] = data['text'].apply(lambda x: tokenize(x))
    log_info("Data preprocessing completed.")
    return data

def split_data(data):
    """
    Split the dataset into training and testing sets.
    :param data: Preprocessed dataset.
    :return: Training and testing sets.
    """
    log_info("Splitting data into training and testing sets.")
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    log_info("Data split completed.")
    return train, test

def main():
    file_path = 'dataset.csv'
    data = load_data(file_path)

    progress_bar = setup_progress_monitor(len(data))
    
    preprocessed_data = preprocess_data(data)
    train_data, test_data = split_data(preprocessed_data)

    update_progress(progress_bar, len(data))
    close_progress(progress_bar)

if __name__ == "__main__":
    main()
