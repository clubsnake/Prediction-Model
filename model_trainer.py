import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config_loader import MODELS_DIR

def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.

    Args:
        model (RandomForestClassifier): Trained RandomForestClassifier model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.

    Returns:
        float: Accuracy of the model on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, filename='model.joblib'):
    """
    Saves the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained RandomForestClassifier model.
        filename (str): Name of the file to save the model to.
    """
    model_path = os.path.join(MODELS_DIR, filename) # Use MODELS_DIR from __init__.py
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
