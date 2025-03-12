"""
CNN Price Prediction Model

A PyTorch-based Convolutional Neural Network (CNN) for financial price prediction
that seamlessly integrates with scikit-learn pipelines and ensemble models.
"""

import logging
import os
import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class CNNDataset(Dataset):
    """
    Dataset class for CNN price prediction model.
    Handles conversion between numpy arrays and PyTorch tensors.
    """

    def __init__(self, X, y, lookback=20, horizon=1, transform=None):
        """
        Initialize the dataset.

        Args:
            X (np.ndarray): Features array
            y (np.ndarray): Target array
            lookback (int): Number of past time steps to use as input
            horizon (int): Number of future time steps to predict
            transform (callable, optional): Optional transform to be applied to the data
        """
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
        self.lookback = lookback
        self.horizon = horizon
        self.transform = transform

        # Reshape X to (batch_size, channels, sequence_length, features)
        if len(self.X.shape) == 2:  # (batch_size, features)
            self.X = self.X.unsqueeze(1)  # Add channel dimension

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class CNNBlock(nn.Module):
    """
    Basic building block for CNN architecture.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_rate=0.1,
        use_batchnorm=True,
        activation="relu",
    ):
        """
        Initialize a CNN block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride of the convolution
            padding (int): Padding size
            dropout_rate (float): Dropout rate
            use_batchnorm (bool): Whether to use batch normalization
            activation (str): Activation function to use ('relu', 'leaky_relu', 'elu')
        """
        super(CNNBlock, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        # Batch normalization
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_channels)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass through the CNN block."""
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNNPriceModel(nn.Module):
    """
    CNN model for financial price prediction.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,
        num_conv_layers=5,
        num_filters=128,
        kernel_size=3,
        stride=1,
        dropout_rate=0.3,
        activation="relu",
        use_adaptive_pooling=True,
        fc_layers=[512, 256, 128, 64],
        lookback=20,
    ):
        """
        Initialize CNN Price Prediction model.

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features (prediction horizon)
            num_conv_layers (int): Number of convolutional layers
            num_filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride of the convolution
            dropout_rate (float): Dropout rate
            activation (str): Activation function to use
            use_adaptive_pooling (bool): Whether to use adaptive pooling
            fc_layers (List[int]): List of fully connected layer sizes
            lookback (int): Number of timesteps to look back
        """
        super(CNNPriceModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_adaptive_pooling = use_adaptive_pooling
        self.fc_layers = fc_layers
        self.lookback = lookback

        # Convolutional layers
        self.conv_layers = nn.ModuleList()

        # First convolutional layer
        self.conv_layers.append(
            CNNBlock(
                1,
                num_filters,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                dropout_rate=dropout_rate,
                activation=activation,
            )
        )

        # Additional convolutional layers
        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                CNNBlock(
                    num_filters,
                    num_filters,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
            )

        # Adaptive pooling
        if use_adaptive_pooling:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(16)  # Fixed output size
            cnn_output_size = 16 * num_filters
        else:
            # Calculate output size after convolutions
            cnn_output_size = lookback * num_filters

        # Fully connected layers
        self.fc_blocks = nn.ModuleList()
        prev_size = cnn_output_size

        for fc_size in fc_layers:
            self.fc_blocks.append(nn.Linear(prev_size, fc_size))
            self.fc_blocks.append(nn.ReLU())
            self.fc_blocks.append(nn.Dropout(dropout_rate))
            prev_size = fc_size

        # Output layer
        self.output = nn.Linear(prev_size, output_dim)

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length, features)
                            or (batch_size, channels, sequence_length)

        Returns:
            torch.Tensor: Predictions
        """
        batch_size = x.size(0)

        # Reshape input if needed
        if len(x.shape) == 4:  # (batch_size, channels, sequence_length, features)
            # Reshape to (batch_size, channels, sequence_length * features)
            x = x.view(batch_size, x.size(1), -1)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Apply adaptive pooling if specified
        if self.use_adaptive_pooling:
            x = self.adaptive_pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(batch_size, -1)

        # Apply fully connected layers
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # Output layer
        x = self.output(x)

        return x

    def predict(self, X):
        """
        Make predictions using the model.

        Args:
            X (np.ndarray): Input features

        Returns:
            np.ndarray: Predictions
        """
        self.eval()  # Set model to evaluation mode

        # Convert input to PyTorch tensor
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X).to(device)

        # Reshape if needed
        if len(X.shape) == 2:  # (batch_size, features)
            X = X.unsqueeze(1)  # Add channel dimension

        # Make predictions
        with torch.no_grad():
            predictions = self.forward(X)

        # Convert to numpy array
        predictions = predictions.cpu().numpy()

        return predictions


class CNNPricePredictor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for CNN price prediction model.
    Allows the CNN model to be used in scikit-learn pipelines and ensembles.
    """

    def __init__(
        self,
        input_dim=None,
        output_dim=1,
        num_conv_layers=3,
        num_filters=64,
        kernel_size=3,
        stride=1,
        dropout_rate=0.2,
        activation="relu",
        use_adaptive_pooling=True,
        fc_layers=[128, 64],
        lookback=20,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        optimizer="adam",
        early_stopping_patience=10,
        validation_split=0.1,
        random_state=None,
        verbose=0,
    ):
        """
        Initialize CNN Price Predictor.

        Args:
            input_dim (int, optional): Number of input features. If None, inferred from data.
            output_dim (int): Number of output features (prediction horizon)
            num_conv_layers (int): Number of convolutional layers
            num_filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride of the convolution
            dropout_rate (float): Dropout rate
            activation (str): Activation function to use
            use_adaptive_pooling (bool): Whether to use adaptive pooling
            fc_layers (List[int]): List of fully connected layer sizes
            lookback (int): Number of timesteps to look back
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            optimizer (str): Optimizer to use ('adam', 'sgd', 'rmsprop')
            early_stopping_patience (int): Patience for early stopping
            validation_split (float): Fraction of training data to use for validation
            random_state (int, optional): Random state for reproducibility
            verbose (int): Verbosity level (0, 1, or 2)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = self.stride
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_adaptive_pooling = use_adaptive_pooling
        self.fc_layers = fc_layers
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

        # Initialize model to None - will be created during fit
        self.model_ = None
        self.history_ = None
        self.best_loss_ = float("inf")
        self.scaler_ = StandardScaler()

    def _create_model(self, input_dim):
        """
        Create the CNN model.

        Args:
            input_dim (int): Number of input features

        Returns:
            CNNPriceModel: Created model
        """
        model = CNNPriceModel(
            input_dim=input_dim,
            output_dim=self.output_dim,
            num_conv_layers=self.num_conv_layers,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            use_adaptive_pooling=self.use_adaptive_pooling,
            fc_layers=self.fc_layers,
            lookback=self.lookback,
        ).to(device)

        return model

    def _get_optimizer(self, model):
        """
        Get the optimizer based on the specified optimizer type.

        Args:
            model: PyTorch model

        Returns:
            torch.optim.Optimizer: Optimizer
        """
        if self.optimizer.lower() == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer.lower() == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            return optim.Adam(model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        """
        Fit the CNN model to the data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples, output_dim)

        Returns:
            self: The trained model
        """
        # Set random seed if specified
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Determine input_dim if not specified
        if self.input_dim is None:
            self.input_dim = X.shape[1] if len(X.shape) > 1 else 1

        # Scale the features
        X_scaled = self.scaler_.fit_transform(X)

        # Reshape y to ensure it has the right dimensions
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.validation_split, random_state=self.random_state
        )

        # Create datasets and dataloaders
        train_dataset = CNNDataset(X_train, y_train, lookback=self.lookback)
        val_dataset = CNNDataset(X_val, y_val, lookback=self.lookback)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Create model
        self.model_ = self._create_model(self.input_dim)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = self._get_optimizer(self.model_)

        # For early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model_(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation phase
            self.model_.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model_(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Print progress
            if self.verbose > 0 and (epoch % 10 == 0 or epoch == self.epochs - 1):
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state_ = {
                    key: value.cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose > 0:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model state
        if hasattr(self, "best_model_state_"):
            self.model_.load_state_dict(self.best_model_state_)

        self.history_ = history
        self.best_loss_ = best_val_loss

        return self

    def predict(self, X):
        """
        Make predictions using the trained CNN model.

        Args:
            X (np.ndarray): Features to predict on

        Returns:
            np.ndarray: Predictions
        """
        if self.model_ is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Scale the features
        X_scaled = self.scaler_.transform(X)

        # Create dataset and dataloader
        dataset = CNNDataset(
            X_scaled, np.zeros((len(X_scaled), self.output_dim)), lookback=self.lookback
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in dataloader:
                output = self.model_(data)
                predictions.append(output.cpu().numpy())

        # Concatenate predictions
        predictions = np.vstack(predictions)

        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the model on test data.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True values

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Reshape y to ensure it has the right dimensions
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Calculate MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            mape = np.nan_to_num(mape)  # Replace NaN with 0

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    def save_model(self, path):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model
        """
        if self.model_ is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model parameters and configuration
        state = {
            "model_state_dict": self.model_.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_conv_layers": self.num_conv_layers,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_adaptive_pooling": self.use_adaptive_pooling,
            "fc_layers": self.fc_layers,
            "lookback": self.lookback,
            "scaler": self.scaler_,
        }

        torch.save(state, path)

        if self.verbose > 0:
            logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        """
        Load a trained model from a file.

        Args:
            path (str): Path to the saved model

        Returns:
            CNNPricePredictor: Loaded model
        """
        # Load the state dictionary
        state = torch.load(path, map_location=device)

        # Create a new instance of the model
        model = cls(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            num_conv_layers=state["num_conv_layers"],
            num_filters=state["num_filters"],
            kernel_size=state["kernel_size"],
            stride=state["stride"],
            dropout_rate=state["dropout_rate"],
            activation=state["activation"],
            use_adaptive_pooling=state["use_adaptive_pooling"],
            fc_layers=state["fc_layers"],
            lookback=state["lookback"],
        )

        # Create the PyTorch model
        model.model_ = CNNPriceModel(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            num_conv_layers=state["num_conv_layers"],
            num_filters=state["num_filters"],
            kernel_size=state["kernel_size"],
            stride=state["stride"],
            dropout_rate=state["dropout_rate"],
            activation=state["activation"],
            use_adaptive_pooling=state["use_adaptive_pooling"],
            fc_layers=state["fc_layers"],
            lookback=state["lookback"],
        ).to(device)

        # Load the model weights
        model.model_.load_state_dict(state["model_state_dict"])

        # Load the scaler
        model.scaler_ = state["scaler"]

        return model

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, return the parameters of all sub-objects

        Returns:
            dict: Parameter names mapped to their values
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_conv_layers": self.num_conv_layers,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_adaptive_pooling": self.use_adaptive_pooling,
            "fc_layers": self.fc_layers,
            "lookback": self.lookback,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "early_stopping_patience": self.early_stopping_patience,
            "validation_split": self.validation_split,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Args:
            **parameters: Parameter names and their values

        Returns:
            self: The estimator
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


from src.tuning.study_manager import study_manager


def optimize_cnn_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_trials=50,
    timeout=3600,
    study_name="cnn_optimization",
    storage=None,
    load_if_exists=True,
    verbose=True,
):
    """
    Optimize CNN model hyperparameters using Optuna.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray, optional): Validation features. If None, a portion of training data is used.
        y_val (np.ndarray, optional): Validation targets. If None, a portion of training data is used.
        n_trials (int): Number of optimization trials
        timeout (int): Timeout in seconds
        study_name (str): Name of the Optuna study
        storage (str, optional): Storage URL for the Optuna study
        load_if_exists (bool): Whether to load a previous study if it exists
        verbose (bool): Whether to print progress information

    Returns:
        tuple: (best_params, study)
            - best_params (dict): Best hyperparameters
            - study (optuna.Study): Completed Optuna study
    """
    # Reshape y_train if needed
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    # Create validation set if not provided
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    # Reshape y_val if needed
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)

    def objective(trial):
        """Optuna objective function."""
        # Sample hyperparameters
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
        num_filters = trial.suggest_int("num_filters", 16, 256, log=True)
        kernel_size = trial.suggest_int("kernel_size", 2, 7)
        stride = trial.suggest_int("stride", 1, 2)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "elu"]
        )
        optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        use_adaptive_pooling = trial.suggest_categorical(
            "use_adaptive_pooling", [True, False]
        )
        lookback = trial.suggest_int("lookback", 10, 50)

        # Sample fully connected layer architecture
        n_fc_layers = trial.suggest_int("n_fc_layers", 1, 3)
        fc_layers = []
        for i in range(n_fc_layers):
            fc_layers.append(trial.suggest_int(f"fc_layer_{i}", 16, 256, log=True))

        # Create and train model
        model = CNNPricePredictor(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            dropout_rate=dropout_rate,
            activation=activation,
            use_adaptive_pooling=use_adaptive_pooling,
            fc_layers=fc_layers,
            lookback=lookback,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=100,  # Use fixed epochs for optimization
            optimizer=optimizer,
            early_stopping_patience=10,
            validation_split=0.1,
            random_state=42,
            verbose=0,
        )

        try:
            # Fit model
            model.fit(X_train, y_train)

            # Evaluate on validation data
            metrics = model.evaluate(X_val, y_val)

            # Use RMSE as the objective to minimize
            rmse = metrics["rmse"]

            # Report intermediate results
            trial.report(rmse, step=0)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return rmse

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    # Create or load study
    study = study_manager.create_study(
        study_name=study_name,
        storage_path=storage,  # or some resolved path
        direction="minimize",
        load_if_exists=load_if_exists,
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=verbose,
    )

    # Get best parameters
    best_params = study.best_params

    # Add parameters that weren't optimized
    best_fc_layers = []
    for i in range(best_params["n_fc_layers"]):
        best_fc_layers.append(best_params[f"fc_layer_{i}"])

    # Clean up best_params by removing individual fc_layer parameters
    for i in range(best_params["n_fc_layers"]):
        best_params.pop(f"fc_layer_{i}")

    best_params.pop("n_fc_layers")
    best_params["fc_layers"] = best_fc_layers

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best RMSE: {study.best_value:.6f}")

    return best_params, study


def create_optimized_model(X_train, y_train, best_params):
    """
    Create a CNN model with optimized hyperparameters.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        best_params (dict): Best hyperparameters from Optuna optimization

    Returns:
        CNNPricePredictor: Model with optimized hyperparameters
    """
    # Reshape y_train if needed
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    # Create model with best parameters
    model = CNNPricePredictor(
        input_dim=X_train.shape[1], output_dim=y_train.shape[1], **best_params
    )

    return model


class EnsembleModel:
    """
    Ensemble model that combines CNN model with other models.
    """

    def __init__(self, models, weights=None):
        """
        Initialize ensemble model.

        Args:
            models (list): List of models to ensemble
            weights (list, optional): List of weights for each model. If None, use equal weights.
        """
        self.models = models

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(models) for _ in range(len(models))]
        else:
            if len(weights) != len(models):
                raise ValueError("Length of weights must match length of models")

            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    def predict(self, X):
        """
        Make predictions using the ensemble.

        Args:
            X (np.ndarray): Features to predict on

        Returns:
            np.ndarray: Ensemble predictions
        """
        predictions = []

        # Get predictions from each model
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Ensure all predictions have the same shape
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            logger.warning(f"Models have different output shapes: {shapes}")

            # Try to reshape all predictions to match the first model's output shape
            output_shape = predictions[0].shape
            for i in range(1, len(predictions)):
                if predictions[i].shape != output_shape:
                    try:
                        predictions[i] = predictions[i].reshape(output_shape)
                    except:
                        logger.error(
                            f"Could not reshape prediction {i} to {output_shape}"
                        )
                        raise ValueError(
                            f"Models have incompatible output shapes: {shapes}"
                        )

        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred

        return ensemble_pred

    def evaluate(self, X, y):
        """
        Evaluate the ensemble model on test data.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True values

        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Reshape y to ensure it has the right dimensions
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Calculate MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            mape = np.nan_to_num(mape)  # Replace NaN with 0

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# Unit tests
def test_cnn_model():
    """Test CNN model functionality."""
    # Create random data
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)

    # Test CNNPriceModel
    model = CNNPriceModel(input_dim=10, output_dim=1)
    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
    output = model(X_tensor)
    assert output.shape == (
        100,
        1,
    ), f"Expected output shape (100, 1), got {output.shape}"

    # Test CNNPricePredictor
    predictor = CNNPricePredictor(
        input_dim=10, output_dim=1, epochs=2, batch_size=32, verbose=0
    )
    predictor.fit(X, y)
    predictions = predictor.predict(X)
    assert predictions.shape == (
        100,
        1,
    ), f"Expected predictions shape (100, 1), got {predictions.shape}"

    # Test model evaluation
    metrics = predictor.evaluate(X, y)
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "rmse" in metrics, "Metrics should include RMSE"

    print("All tests passed!")
    return True


if __name__ == "__main__":
    # Run tests
    test_result = test_cnn_model()

    if test_result:
        print("CNN model is working correctly!")
