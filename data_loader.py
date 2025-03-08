import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config_loader import DATA_DIR

# Define the path to the dataset
DATASET_PATH = os.path.join(DATA_DIR, 'sensor_data.csv')

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Loads the dataset, preprocesses it, and splits it into training and testing sets.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training features, training labels, testing features, and testing labels.
    """
    try:
        # Load the dataset
        data = pd.read_csv(DATASET_PATH)

        # Data preprocessing steps
        data.dropna(inplace=True)  # Drop rows with missing values
        data = pd.get_dummies(data, columns=['categorical_feature'])  # Convert categorical features to numerical

        # Separate features and labels
        X = data.drop('target_variable', axis=1)
        y = data['target_variable']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print(f"Error: The dataset file was not found at {DATASET_PATH}.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
