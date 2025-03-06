import pandas as pd


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """Loads data from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return None

    def clean_data(self):
        """Cleans the loaded data (e.g., handling missing values)."""
        if self.data is not None:
            self.data = self.data.dropna()  # Example: Remove rows with missing values
            # Add more cleaning steps as needed
            return self.data
        else:
            print("Error: No data loaded. Call load_data() first.")
            return None

    def preprocess_data(self):
        """Preprocesses the data for model training."""
        # Example: Feature scaling or encoding
        # Add preprocessing steps as needed
        return self.data
