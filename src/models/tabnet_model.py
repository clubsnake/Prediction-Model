"""
TabNet Model for Price Prediction

This module implements a TabNet-based model for price prediction that seamlessly
integrates with existing ensemble pipelines. TabNet provides explainable deep learning
for tabular data with feature selection capabilities.
"""

import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TabNet_PricePrediction")

# Add project root to Python path
current_file = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import TabNet
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor

    TABNET_AVAILABLE = True
    logger.info("TabNet successfully imported")
except ImportError:
    logger.warning("TabNet module not available. TabNet models will not be supported.")
    TABNET_AVAILABLE = False

from src.utils.gpu_memory_manager import GPUMemoryManager
from src.models.model import place_on_device

class TabNetPricePredictor(BaseEstimator, RegressorMixin):
    """
    TabNet model for price prediction that integrates with ensemble pipelines.

    This class implements a TabNet-based regression model using PyTorch-TabNet,
    with support for hyperparameter optimization, feature importance analysis,
    and ensemble integration.

    Attributes:
        model (TabNetRegressor): The underlying TabNet model
        params (dict): Model hyperparameters
        scaler (StandardScaler): Feature scaler for preprocessing
        feature_names (list): Names of features used in the model
        fitted (bool): Whether the model has been fitted
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.5,
        lambda_sparse: float = 0.01,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        mask_type: str = "entmax",
        n_shared: Optional[int] = None,
        momentum: float = 0.02,
        clip_value: Optional[float] = None,
        verbose: int = 1,
        device_name: str = "auto",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
    ):
        """
        Initialize the TabNet price predictor.

        Args:
            n_d: Width of the decision prediction layer
            n_a: Width of the attention embedding for each mask step
            n_steps: Number of steps in the architecture
            gamma: Scaling factor for attention updates
            lambda_sparse: Sparsity regularization parameter
            optimizer_params: Parameters for the optimizer
            scheduler_params: Parameters for the learning rate scheduler
            mask_type: Type of mask to use (default: "entmax")
            n_shared: Number of shared blocks before split
            momentum: Momentum for batch normalization
            clip_value: Gradient clipping value
            verbose: Verbosity level (0: silent, 1: progress bar)
            device_name: Device to use ('auto', 'cpu', 'cuda')
            feature_names: Names of features
            random_state: Random seed for reproducibility
            max_epochs: Maximum number of epochs for training
            patience: Patience for early stopping
            batch_size: Batch size for training
            virtual_batch_size: Virtual batch size for Ghost Batch Normalization
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_params = optimizer_params if optimizer_params else {"lr": 2e-2}
        self.scheduler_params = (
            scheduler_params if scheduler_params else {"step_size": 50, "gamma": 0.9}
        )
        self.mask_type = mask_type
        self.n_shared = n_shared
        self.momentum = momentum
        self.clip_value = clip_value
        self.verbose = verbose
        self.device_name = device_name
        self.feature_names = feature_names
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size

        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

        # Initialize GPU manager for proper device placement
        self.gpu_manager = GPUMemoryManager(allow_growth=True)
        self.gpu_manager.initialize()
        
        # Override device_name if set to "auto"
        if device_name == "auto":
            # Let the GPU manager determine the appropriate device
            self.device = self.gpu_manager.get_torch_device()
            self.device_name = "cuda" if str(self.device) == "cuda" else "cpu"
        else:
            self.device_name = device_name
            self.device = device_name
        
        # Log the selected device
        logger.info(f"TabNet using device: {self.device}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        eval_metric: Optional[List[str]] = None,
        eval_name: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        batch_size: Optional[int] = None,
        virtual_batch_size: Optional[int] = None,
        from_unsupervised: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        pretraining_ratio: Optional[float] = None,
        warm_start: bool = False,
        augmentations: Optional[Any] = None,
    ) -> "TabNetPricePredictor":
        """
        Fit the TabNet model to the training data.

        Args:
            X: Training data features
            y: Training data targets
            eval_set: Validation set (X_val, y_val)
            eval_metric: List of evaluation metrics
            eval_name: Names for the evaluation metrics
            weights: Sample weights
            max_epochs: Maximum number of epochs
            patience: Patience for early stopping
            batch_size: Batch size for training
            virtual_batch_size: Virtual batch size
            from_unsupervised: Pretrained unsupervised model
            loss_fn: Custom loss function
            pretraining_ratio: Ratio for pretraining
            warm_start: Whether to continue training from current state
            augmentations: Data augmentation functions

        Returns:
            self: The fitted model
        """
        try:
            # Determine feature names if not provided
            if self.feature_names is None and hasattr(X, "columns"):
                self.feature_names = X.columns.tolist()

            # Scale features
            if isinstance(X, pd.DataFrame):
                X_np = self.scaler.fit_transform(X)
            else:
                X_np = self.scaler.fit_transform(X)

            # Convert target to numpy if needed
            if isinstance(y, pd.Series):
                y_np = y.values
            else:
                y_np = y

            # Initialize TabNet model if not already created or warm start is False
            if self.model is None or not warm_start:
                self.model = TabNetRegressor(
                    n_d=self.n_d,
                    n_a=self.n_a,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                    lambda_sparse=self.lambda_sparse,
                    optimizer_params=self.optimizer_params,
                    scheduler_params=self.scheduler_params,
                    mask_type=self.mask_type,
                    n_shared=self.n_shared,
                    momentum=self.momentum,
                    clip_value=self.clip_value,
                    verbose=self.verbose,
                    device_name=self.device_name,
                    seed=self.random_state,
                )

            # Set up validation data if not provided
            if eval_set is None and len(X_np) > 100:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_np, y_np, test_size=0.2, random_state=self.random_state
                )
                eval_set = [(X_val, y_val)]
            else:
                X_train, y_train = X_np, y_np

            # Override instance parameters if provided
            max_epochs = max_epochs or self.max_epochs
            patience = patience or self.patience
            batch_size = batch_size or self.batch_size
            virtual_batch_size = virtual_batch_size or self.virtual_batch_size

            # Fit the model
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=eval_set,
                eval_metric=eval_metric or ["rmse"],
                eval_name=eval_name,
                weights=weights,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                from_unsupervised=from_unsupervised,
                loss_fn=loss_fn,
                pretraining_ratio=pretraining_ratio,
                warm_start=warm_start,
                augmentations=augmentations,
            )

            self.fitted = True
            return self

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise RuntimeError(f"Failed to fit TabNet model: {str(e)}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained TabNet model.

        Args:
            X: Features to make predictions on

        Returns:
            Predictions array

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            # Scale features
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            # Generate predictions
            return self.model.predict(X_scaled)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Failed to generate predictions: {str(e)}") from e

    def feature_importances(self, kind: str = "gain") -> np.ndarray:
        """
        Get feature importances from the trained model.

        Args:
            kind: Type of feature importance ('gain' or 'split')

        Returns:
            Array of feature importance scores

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            if kind == "gain":
                return self.model.feature_importances_
            elif kind == "mask":
                return self.model.feature_importances_
            else:
                raise ValueError(
                    f"Unknown importance type: {kind}. Use 'gain' or 'mask'."
                )

        except Exception as e:
            logger.error(f"Error retrieving feature importances: {str(e)}")
            raise RuntimeError(f"Failed to get feature importances: {str(e)}") from e

    def explain(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate model explanations using TabNet's built-in explainability.

        Args:
            X: Features to explain

        Returns:
            Dictionary with masks and aggregated feature importance

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            # Scale features
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            # Get masks from the model
            masks = self.model.explain(X_scaled)

            # Aggregate masks across decision steps
            agg_masks = masks.sum(dim=0).cpu().numpy()

            # Normalize to get feature importance per instance
            instance_importances = agg_masks / agg_masks.sum(axis=1, keepdims=True)

            # Overall feature importance
            overall_importance = instance_importances.mean(axis=0)

            result = {
                "instance_importances": instance_importances,
                "overall_importance": overall_importance,
                "masks": masks.cpu().numpy(),
            }

            return result

        except Exception as e:
            logger.error(f"Error during explanation: {str(e)}")
            raise RuntimeError(f"Failed to generate explanations: {str(e)}") from e

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            RuntimeError: If saving fails
        """
        if not self.fitted or self.model is None:
            logger.warning("Saving an unfitted model.")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            # Save TabNet model separately
            model_path = f"{path}_tabnet_model"
            if self.model is not None:
                self.model.save_model(model_path)

            # Save the wrapper without the model
            temp_model = self.model
            self.model = None

            with open(path, "wb") as f:
                pickle.dump(self, f)

            # Restore the model
            self.model = temp_model

            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}") from e

    @classmethod
    def load(cls, path: str) -> "TabNetPricePredictor":
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded TabNetPricePredictor instance

        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Load the wrapper
            with open(path, "rb") as f:
                model = pickle.load(f)

            # Load the TabNet model
            model_path = f"{path}_tabnet_model"
            if os.path.exists(f"{model_path}.zip"):
                model.model = TabNetRegressor()
                model.model.load_model(model_path)
                model.fitted = True

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (scikit-learn compatibility)."""
        return {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "lambda_sparse": self.lambda_sparse,
            "optimizer_params": self.optimizer_params,
            "scheduler_params": self.scheduler_params,
            "mask_type": self.mask_type,
            "n_shared": self.n_shared,
            "momentum": self.momentum,
            "clip_value": self.clip_value,
            "verbose": self.verbose,
            "device_name": self.device_name,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
        }

    def set_params(self, **params) -> "TabNetPricePredictor":
        """Set parameters for this estimator (scikit-learn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            # Make predictions
            y_pred = self.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Calculate MAPE
            epsilon = 1e-10  # To avoid division by zero
            mape = np.mean(np.abs((y - y_pred) / (np.abs(y) + epsilon))) * 100

            return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise RuntimeError(f"Failed to evaluate model: {str(e)}") from e


class TabNetOptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for TabNet models.

    This class handles hyperparameter optimization for TabNet models
    using Optuna, with support for parallel execution and pruning.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        study_name: str = "tabnet_optimization",
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the TabNet Optuna optimizer.

        Args:
            X: Training features
            y: Training targets
            feature_names: Names of features
            eval_set: Validation set (X_val, y_val)
            study_name: Name of the Optuna study
            storage: Optuna storage URL (default: None, in-memory)
            direction: Optimization direction ('minimize' or 'maximize')
            n_trials: Number of optimization trials to run
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            random_state: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.eval_set = eval_set
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.random_state = random_state

        # Create train/val split if eval_set not provided
        if self.eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state
            )
            self.X_train = X_train
            self.y_train = y_train
            self.eval_set = [(X_val, y_val)]
        else:
            self.X_train = self.X
            self.y_train = self.y

        self.study = None
        self.best_params = None
        self.best_value = None
        self.best_model = None

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation metric (to be minimized or maximized)
        """
        # Sample hyperparameters
        n_d = trial.suggest_int("n_d", 8, 128)
        n_a = trial.suggest_int("n_a", 8, 128)
        n_steps = trial.suggest_int("n_steps", 3, 10)
        gamma = trial.suggest_float("gamma", 1.0, 2.0)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True)

        # Learning rate
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        optimizer_params = {"lr": lr}

        # Scheduler parameters
        scheduler_step = trial.suggest_int("scheduler_step", 10, 100)
        scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.5, 0.95)
        scheduler_params = {"step_size": scheduler_step, "gamma": scheduler_gamma}

        # Training parameters
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
        virtual_batch_size = trial.suggest_int("virtual_batch_size", 64, 512)

        # Momentum for batch normalization
        momentum = trial.suggest_float("momentum", 0.01, 0.4)

        # Create and train the model
        model = TabNetPricePredictor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            momentum=momentum,
            verbose=0,
            feature_names=self.feature_names,
            random_state=self.random_state,
            max_epochs=200,  # Use early stopping
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
        )

        try:
            # Fit the model
            model.fit(
                X=self.X_train,
                y=self.y_train,
                eval_set=self.eval_set,
                eval_metric=["rmse"],
                patience=20,
            )

            # Get validation score
            if hasattr(model.model, "best_cost"):
                validation_score = model.model.best_cost
            else:
                # Fallback to calculating validation metrics manually
                X_val, y_val = self.eval_set[0]
                y_pred = model.predict(X_val)
                validation_score = np.sqrt(mean_squared_error(y_val, y_pred))

            return validation_score

        except Exception as e:
            logger.error(f"Error in trial: {str(e)}")
            raise optuna.exceptions.TrialPruned() from e

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            # Create Optuna study
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                load_if_exists=True,
            )

            # Set random seed
            np.random.seed(self.random_state)

            # Run optimization
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                timeout=self.timeout,
                catch=(Exception,),
            )

            # Get best results
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value

            # Train best model
            self.best_model = TabNetPricePredictor(
                n_d=self.best_params["n_d"],
                n_a=self.best_params["n_a"],
                n_steps=self.best_params["n_steps"],
                gamma=self.best_params["gamma"],
                lambda_sparse=self.best_params["lambda_sparse"],
                optimizer_params={"lr": self.best_params["lr"]},
                scheduler_params={
                    "step_size": self.best_params["scheduler_step"],
                    "gamma": self.best_params["scheduler_gamma"],
                },
                momentum=self.best_params["momentum"],
                feature_names=self.feature_names,
                random_state=self.random_state,
                batch_size=self.best_params["batch_size"],
                virtual_batch_size=self.best_params["virtual_batch_size"],
            )

            self.best_model.fit(
                X=self.X, y=self.y, eval_set=self.eval_set, eval_metric=["rmse"]
            )

            logger.info(f"Optimization completed. Best value: {self.best_value:.6f}")
            logger.info(f"Best parameters: {self.best_params}")

            return {
                "best_params": self.best_params,
                "best_value": self.best_value,
                "best_model": self.best_model,
                "study": self.study,
            }

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise RuntimeError(f"Failed to optimize hyperparameters: {str(e)}") from e

    def plot_study(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate and save optimization visualizations.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot type to file path

        Raises:
            RuntimeError: If plotting fails
        """
        if self.study is None:
            raise RuntimeError("No study available. Run optimize() first.")

        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt

            # Create output directory
            if output_dir is None:
                output_dir = "optuna_plots"
            os.makedirs(output_dir, exist_ok=True)

            # Generate plots
            plot_files = {}

            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            history_path = os.path.join(output_dir, f"{self.study_name}_history.png")
            fig.savefig(history_path)
            plt.close(fig)
            plot_files["history"] = history_path

            # Parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            importances_path = os.path.join(
                output_dir, f"{self.study_name}_importances.png"
            )
            fig.savefig(importances_path)
            plt.close(fig)
            plot_files["importances"] = importances_path

            # Slice plot
            fig = optuna.visualization.matplotlib.plot_slice(self.study)
            slice_path = os.path.join(output_dir, f"{self.study_name}_slice.png")
            fig.savefig(slice_path)
            plt.close(fig)
            plot_files["slice"] = slice_path

            # Contour plot for most important parameters
            try:
                param_importances = optuna.importance.get_param_importances(self.study)
                top_params = list(param_importances.keys())[:2]  # Get top 2 parameters

                if len(top_params) >= 2:
                    fig = optuna.visualization.matplotlib.plot_contour(
                        self.study, top_params
                    )
                    contour_path = os.path.join(
                        output_dir, f"{self.study_name}_contour.png"
                    )
                    fig.savefig(contour_path)
                    plt.close(fig)
                    plot_files["contour"] = contour_path
            except Exception as e:
                logger.warning(f"Could not generate contour plot: {str(e)}")

            return plot_files

        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise RuntimeError(f"Failed to generate plots: {str(e)}") from e


class EnsembleTabNetIntegration:
    """
    Utilities for integrating TabNet models with ensemble prediction pipelines.

    This class provides helper functions for ensemble integration, feature importance
    aggregation, and ensemble weight optimization.
    """

    @staticmethod
    def create_ensemble_ready_model(
        tabnet_model: TabNetPricePredictor,
        ensemble_weight: float = 1.0,
        model_name: str = "tabnet",
    ) -> Dict[str, Any]:
        """
        Create an ensemble-ready model dictionary.

        Args:
            tabnet_model: Trained TabNet model
            ensemble_weight: Weight in the ensemble
            model_name: Name of the model in the ensemble

        Returns:
            Dictionary with model and metadata
        """
        return {
            "model": tabnet_model,
            "weight": ensemble_weight,
            "name": model_name,
            "type": "tabnet",
        }

    @staticmethod
    def combine_feature_importances(
        models: List[Dict[str, Any]], feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Combine feature importances from multiple models.

        Args:
            models: List of model dictionaries (must include 'model' and 'weight')
            feature_names: Names of features

        Returns:
            DataFrame with combined feature importances
        """
        combined_importances = pd.DataFrame(index=feature_names)

        for model_dict in models:
            model = model_dict["model"]
            weight = model_dict["weight"]
            name = model_dict["name"]

            # Skip models without feature_importances method
            if not hasattr(model, "feature_importances"):
                continue

            try:
                # Get feature importances
                importances = model.feature_importances()

                # Ensure correct length
                if len(importances) == len(feature_names):
                    # Apply model weight
                    weighted_importances = importances * weight
                    combined_importances[name] = weighted_importances
            except Exception as e:
                logger.warning(
                    f"Error getting feature importances for {name}: {str(e)}"
                )

        # Calculate weighted average
        if combined_importances.shape[1] > 0:
            total_weight = sum(
                model_dict["weight"]
                for model_dict in models
                if hasattr(model_dict["model"], "feature_importances")
            )

            if total_weight > 0:
                combined_importances["ensemble_average"] = (
                    combined_importances.sum(axis=1) / total_weight
                )

        return combined_importances

    @staticmethod
    def optimize_ensemble_weights(
        models: List[Dict[str, Any]],
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using Optuna.

        Args:
            models: List of model dictionaries
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            random_state: Random seed

        Returns:
            Dictionary mapping model names to optimized weights
        """
        # Create a study
        study = optuna.create_study(direction="minimize")

        # Define objective function
        def objective(trial):
            # Sample weights for each model
            weights = {}
            for model_dict in models:
                name = model_dict["name"]
                weights[name] = trial.suggest_float(f"weight_{name}", 0.0, 1.0)

            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                for name in weights:
                    weights[name] /= total
            else:
                # If all weights are 0, set equal weights
                for name in weights:
                    weights[name] = 1.0 / len(models)

            # Make predictions with weighted ensemble
            ensemble_pred = np.zeros_like(y_val)

            for model_dict in models:
                name = model_dict["name"]
                weight = weights[name]
                model = model_dict["model"]

                if weight > 0:
                    # Get predictions
                    try:
                        pred = model.predict(X_val)
                        ensemble_pred += weight * pred
                    except Exception as e:
                        logger.warning(f"Error predicting with {name}: {str(e)}")

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            return rmse

        # Run optimization
        np.random.seed(random_state)
        study.optimize(objective, n_trials=n_trials)

        # Get best weights
        best_weights = {}
        for model_dict in models:
            name = model_dict["name"]
            best_weights[name] = study.best_params[f"weight_{name}"]

        # Normalize weights
        total = sum(best_weights.values())
        if total > 0:
            for name in best_weights:
                best_weights[name] /= total

        return best_weights


def test_tabnet_module():
    """
    Test function to verify the TabNet module functionality.

    This function creates sample data, trains a TabNet model,
    optimizes hyperparameters, and checks integration capabilities.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + X[:, 1] ** 2 + np.random.randn(n_samples) * 0.1

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train a model
        model = TabNetPricePredictor(
            n_d=16,
            n_a=16,
            n_steps=3,
            feature_names=feature_names,
            max_epochs=10,  # Small number for testing
            patience=5,
        )

        logger.info("Training TabNet model...")
        model.fit(X_train, y_train)

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)

        # Evaluate
        logger.info("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Metrics: {metrics}")

        # Check feature importances
        logger.info("Getting feature importances...")
        importances = model.feature_importances()
        logger.info(f"Top 3 features: {np.argsort(importances)[-3:]}")

        # Test model saving and loading
        logger.info("Testing save/load functionality...")
        model.save("temp_tabnet_model.pkl")
        loaded_model = TabNetPricePredictor.load("temp_tabnet_model.pkl")

        # Verify loaded model works
        loaded_pred = loaded_model.predict(X_test)
        loaded_metrics = loaded_model.evaluate(X_test, y_test)
        logger.info(f"Loaded model metrics: {loaded_metrics}")

        # Clean up
        if os.path.exists("temp_tabnet_model.pkl"):
            os.remove("temp_tabnet_model.pkl")
        if os.path.exists("temp_tabnet_model_tabnet_model.zip"):
            os.remove("temp_tabnet_model_tabnet_model.zip")

        # Test Optuna optimization with fewer trials for testing
        logger.info("Testing Optuna optimization...")
        optimizer = TabNetOptunaOptimizer(
            X=X_train,
            y=y_train,
            feature_names=feature_names,
            n_trials=2,  # Small number for testing
            timeout=300,  # 5 minutes maximum
            random_state=42,
        )

        optimization_result = optimizer.optimize()
        logger.info(f"Best parameters: {optimization_result['best_params']}")

        # Test ensemble integration
        logger.info("Testing ensemble integration...")
        ensemble_model = EnsembleTabNetIntegration.create_ensemble_ready_model(
            tabnet_model=model, ensemble_weight=1.0, model_name="tabnet_model"
        )

        logger.info("All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False


def example_usage():
    """
    Example usage of the TabNet module.

    This function demonstrates how to use the TabNet module
    for price prediction in a real-world scenario.
    """
    # Load your dataset
    # df = pd.read_csv('your_price_data.csv')

    # For demonstration, we'll create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 15

    # Create synthetic price data
    X = np.random.randn(n_samples, n_features)
    y = (
        100
        + 5 * X[:, 0]
        + 2 * X[:, 1] ** 2
        - 3 * X[:, 2]
        + np.random.randn(n_samples) * 10
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]

    print("Synthetic data created.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("\nData split into train/val/test sets.")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Optimize hyperparameters
    print("\nOptimizing hyperparameters (this may take a while)...")
    optimizer = TabNetOptunaOptimizer(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        eval_set=[(X_val, y_val)],
        n_trials=20,  # Increase for better results
        timeout=600,  # 10 minutes maximum
        random_state=42,
    )

    optimization_result = optimizer.optimize()
    best_params = optimization_result["best_params"]
    best_model = optimization_result["best_model"]

    print("\nOptimization complete!")
    print(f"Best parameters: {best_params}")

    # Evaluate on test set
    test_metrics = best_model.evaluate(X_test, y_test)
    print("\nTest set evaluation:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Feature importance analysis
    importances = best_model.feature_importances()
    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    print("\nTop 5 features:")
    print(feature_imp_df.head(5))

    # Save the model
    model_path = "tabnet_price_model.pkl"
    best_model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Example of integration with existing pipeline
    print("\nIntegrating with ensemble pipeline...")

    # Create a dummy second model (in practice, this would be your existing models)
    from sklearn.ensemble import RandomForestRegressor

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Create ensemble models
    ensemble_models = [
        EnsembleTabNetIntegration.create_ensemble_ready_model(
            tabnet_model=best_model, ensemble_weight=0.5, model_name="tabnet"
        ),
        {"model": rf_model, "weight": 0.5, "name": "random_forest", "type": "sklearn"},
    ]

    # Optimize ensemble weights
    print("\nOptimizing ensemble weights...")
    optimized_weights = EnsembleTabNetIntegration.optimize_ensemble_weights(
        models=ensemble_models, X_val=X_val, y_val=y_val, n_trials=10
    )

    print(f"Optimized weights: {optimized_weights}")

    # Make ensemble predictions
    ensemble_pred = np.zeros_like(y_test)
    for model_dict in ensemble_models:
        name = model_dict["name"]
        model = model_dict["model"]
        weight = optimized_weights[name]

        if name == "tabnet":
            pred = model.predict(X_test)
        else:
            pred = model.predict(X_test)

        ensemble_pred += weight * pred

    # Calculate ensemble metrics
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mape = (
        np.mean(np.abs((y_test - ensemble_pred) / (np.abs(y_test) + 1e-10))) * 100
    )

    print("\nEnsemble performance:")
    print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
    print(f"Ensemble MAPE: {ensemble_mape:.4f}")
    print(f"TabNet-only RMSE: {test_metrics['rmse']:.4f}")

    print("\nExample complete!")


if __name__ == "__main__":
    # Run tests
    print("Running module tests...")
    test_result = test_tabnet_module()

    if test_result:
        print("\n=== TabNet Module Tests Passed ===\n")
        print("Running usage example...")
        example_usage()
    else:
        print("\n=== TabNet Module Tests Failed ===\n")
        print("Please check the logs for details")
