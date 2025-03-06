import os

import pandas as pd


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def save_data(self, df, filename):
        file_path = os.path.join(self.data_dir, filename)
        try:
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
