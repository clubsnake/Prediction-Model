import os

import joblib


class ModelTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def save_model(self, model, filename):
        file_path = os.path.join(self.data_dir, filename)
        try:
            joblib.dump(model, file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        try:
            return joblib.load(file_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
