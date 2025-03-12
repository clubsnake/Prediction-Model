"""
Test script for TabNet integration with the ensemble prediction pipeline.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory
from src.training.walk_forward import (
    analyze_model_performance,
    generate_predictions,
    train_model_for_fold,
)


def test_tabnet_model_creation():
    """Test creating a TabNet model through the model factory"""
    print("Testing TabNet model creation...")

    # Basic TabNet parameters
    tabnet_params = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 3,
        "gamma": 1.5,
        "lambda_sparse": 0.001,
        "batch_size": 1024,
        "virtual_batch_size": 128,
    }

    try:
        # Create model using factory
        model = ModelFactory.create_model("tabnet", tabnet_params)
        print("✓ Successfully created TabNet model")

        # Check model type
        assert model.model_type == "tabnet", "Model type should be tabnet"
        print("✓ Model type verified")

        # Check parameters
        for key, value in tabnet_params.items():
            assert model.params[key] == value, f"Parameter {key} mismatch"
        print("✓ Parameters verified")

        return True
    except Exception as e:
        print(f"✗ TabNet model creation failed: {e}")
        return False


def test_tabnet_training():
    """Test training a TabNet model with sample data"""
    print("\nTesting TabNet model training...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_samples),
            "feature2": np.random.normal(0, 1, n_samples),
            "feature3": np.random.normal(0, 1, n_samples),
            "feature4": np.random.normal(0, 1, n_samples),
        }
    )
    y = (
        2 * X["feature1"]
        + 0.5 * X["feature2"] ** 2
        + np.random.normal(0, 0.5, n_samples)
    )

    # Split data
    train_idx = int(0.8 * n_samples)
    X_train, X_test = X[:train_idx], X[train_idx:]
    y_train, y_test = y[:train_idx], y[train_idx:]

    try:
        # Create TabNet model
        model = ModelFactory.create_model(
            "tabnet",
            {
                "n_d": 16,
                "n_a": 16,
                "n_steps": 3,
                "max_epochs": 10,  # Use a small value for testing
                "patience": 3,
            },
        )

        # Train model
        model.fit(X_train, y_train)
        print("✓ Successfully trained TabNet model")

        # Make predictions
        preds = model.predict(X_test)
        print(f"✓ Generated predictions, shape: {preds.shape}")

        # Check feature importance
        importance = model.get_feature_importance()
        print(f"✓ Feature importance: {importance}")

        return True
    except Exception as e:
        print(f"✗ TabNet training failed: {e}")
        return False


def test_tabnet_in_ensemble():
    """Test TabNet in an ensemble configuration with walk-forward"""
    print("\nTesting TabNet in ensemble...")

    # Create sample time series data
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "date": dates,
            "feature1": np.random.normal(0, 1, 500),
            "feature2": np.random.normal(0, 1, 500),
            "feature3": np.random.normal(0, 1, 500),
            "Close": np.random.normal(100, 10, 500),  # Target column
        }
    )

    # Create a trend in the target
    df["Close"] = df["Close"].cumsum() + 1000

    # Configure ensemble
    ensemble_config = {
        "models": [
            {
                "name": "tabnet_model",
                "type": "tabnet",
                "weight": 1.0,
                "params": {
                    "n_d": 16,
                    "n_a": 16,
                    "n_steps": 3,
                    "max_epochs": 5,  # Small for testing
                    "patience": 2,
                },
            },
            # Add a second model type for ensemble testing
            {
                "name": "xgboost_model",
                "type": "xgboost",
                "weight": 1.0,
                "params": {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
            },
        ]
    }

    try:
        # Create training and test sets
        train_df = df.iloc[:-50]  # Use last 50 rows for testing
        test_df = df.iloc[-50:]

        # Define features and target
        feature_cols = ["feature1", "feature2", "feature3"]
        target_col = "Close"

        # Train models for one fold
        fold_models = []

        for model_config in ensemble_config["models"]:
            # Prepare data
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = test_df[feature_cols].iloc[:25]  # First half of test for validation
            y_val = test_df[target_col].iloc[:25]

            # Train model
            model, updated_config = train_model_for_fold(
                model_config, X_train, y_train, X_val, y_val
            )

            if model is not None:
                fold_models.append((model, updated_config))

        print(f"✓ Trained {len(fold_models)} models for ensemble")

        # Generate predictions
        X_test = test_df[feature_cols].iloc[25:]  # Second half of test for prediction
        ensemble_preds, all_preds = generate_predictions(
            fold_models, X_test, ensemble_config
        )

        print(f"✓ Generated ensemble predictions, shape: {ensemble_preds.shape}")

        # Analyze performance
        y_test = test_df[target_col].iloc[25:].values
        results = {"predictions": ensemble_preds, "actuals": y_test}

        performance = analyze_model_performance(
            results, fold_models, X_test, feature_cols
        )
        print("✓ Successfully analyzed ensemble performance with TabNet")

        return True
    except Exception as e:
        print(f"✗ TabNet ensemble testing failed: {e}")
        return False


if __name__ == "__main__":
    print("=== TESTING TABNET INTEGRATION ===\n")

    # Run tests
    model_creation_ok = test_tabnet_model_creation()
    training_ok = test_tabnet_training()
    ensemble_ok = test_tabnet_in_ensemble()

    # Summary
    print("\n=== TEST RESULTS ===")
    print(f"Model creation: {'PASS' if model_creation_ok else 'FAIL'}")
    print(f"Model training: {'PASS' if training_ok else 'FAIL'}")
    print(f"Ensemble integration: {'PASS' if ensemble_ok else 'FAIL'}")

    if model_creation_ok and training_ok and ensemble_ok:
        print("\n✅ All tests passed! TabNet successfully integrated.")
    else:
        print("\n❌ Some tests failed. Please check the error messages.")
