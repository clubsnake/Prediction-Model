# Add this to a new file Scripts/explainable_ai.py

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import lime
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import tensorflow as tf
from alibi.explainers import AnchorTabular, IntegratedGradients
from eli5.sklearn import PermutationImportance

logger = logging.getLogger("XAI")


class XAIExplainer:
    """
    Unified explainable AI toolkit for financial time series models.
    Supports multiple explanation techniques:

    1. SHAP (SHapley Additive exPlanations)
    2. LIME (Local Interpretable Model-agnostic Explanations)
    3. Integrated Gradients
    4. Permutation Importance
    5. Anchor Explanations
    6. PDP (Partial Dependence Plots)
    7. Feature interactions

    Can explain predictions for TensorFlow, scikit-learn, and XGBoost models.
    """

    def __init__(
        self, model, model_type: str = "tensorflow", feature_names: List[str] = None
    ):
        """
        Initialize the explainer with a model.

        Args:
            model: The model to explain (TensorFlow, scikit-learn, or XGBoost)
            model_type: Type of model ('tensorflow', 'sklearn', 'xgboost')
            feature_names: Names of the features
        """
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names
        self.explainers = {}
        self.supports_batch = True

        # Validate model type
        if self.model_type not in ["tensorflow", "sklearn", "xgboost"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Check if we're using a sequential model
        self.is_sequential = False
        if self.model_type == "tensorflow":
            # Check the input shape
            if hasattr(model, "input_shape") and len(model.input_shape) > 2:
                self.is_sequential = True
                # Sequential models (like LSTM/RNN) require special handling for explainers
                self.supports_batch = False

    def predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper around model.predict() to handle different model types and input shapes.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        if self.model_type == "tensorflow":
            # Handle sequential models by checking input shape
            if self.is_sequential:
                # Check if input has time dimension
                if len(X.shape) == 2:
                    # Add time dimension if missing
                    lookback = self.model.input_shape[1]
                    n_features = X.shape[1]
                    X_reshaped = np.zeros((1, lookback, n_features))

                    # Use the input as a single sequence
                    for i in range(min(lookback, X.shape[0])):
                        X_reshaped[0, lookback - i - 1] = X[i]

                    return self.model.predict(X_reshaped)
                elif len(X.shape) == 3:
                    # Already has time dimension
                    return self.model.predict(X)
            else:
                # Regular neural network
                return self.model.predict(X)
        elif self.model_type == "sklearn":
            return self.model.predict(X)
        elif self.model_type == "xgboost":
            return self.model.predict(X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def flatten_sequential_input(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten sequential input for explainers that don't support sequential data.

        Args:
            X: Input data with shape (samples, timesteps, features)

        Returns:
            Flattened data with shape (samples, timesteps*features)
        """
        if len(X.shape) == 3:
            # Flatten time and feature dimensions
            return X.reshape(X.shape[0], -1)
        return X

    def explain_with_shap(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        plot: bool = True,
        n_samples: int = 100,
    ) -> Any:
        """
        Explain predictions using SHAP values.

        Args:
            X: Input data to explain
            background_data: Background data for Integrated Gradients (None for KernelExplainer)
            plot: Whether to create and return a plot
            n_samples: Number of samples for KernelExplainer

        Returns:
            shap_values: SHAP values and/or plot
        """
        try:
            # Initialize the appropriate explainer if not already done
            if "shap" not in self.explainers:
                if self.model_type == "tensorflow":
                    if background_data is not None:
                        # Use DeepExplainer for neural networks with background data
                        self.explainers["shap"] = shap.DeepExplainer(
                            self.model, background_data
                        )
                    else:
                        # Use KernelExplainer as a fallback
                        predict_fn = lambda x: self.predict_wrapper(x)
                        self.explainers["shap"] = shap.KernelExplainer(
                            predict_fn, shap.sample(X, n_samples)
                        )
                elif self.model_type in ["sklearn", "xgboost"]:
                    # Use TreeExplainer for tree-based models
                    self.explainers["shap"] = shap.TreeExplainer(self.model)

            # Calculate SHAP values
            shap_values = self.explainers["shap"].shap_values(X)

            # Create plot if requested
            if plot:
                plt.figure(figsize=(12, 8))

                if self.is_sequential and not self.supports_batch:
                    # For sequential models, we'll show feature importance for one instance
                    X_flat = self.flatten_sequential_input(X)

                    if self.feature_names:
                        # Create feature names for each timestep
                        timesteps = X.shape[1] if len(X.shape) > 2 else 1
                        n_features = X.shape[2] if len(X.shape) > 2 else X.shape[1]
                        temporal_feature_names = []

                        for t in range(timesteps):
                            for f in range(min(n_features, len(self.feature_names))):
                                temporal_feature_names.append(
                                    f"{self.feature_names[f]} (t-{timesteps-t})"
                                )

                        # Plot for the first instance
                        shap.summary_plot(
                            shap_values[0:1],
                            X_flat[0:1],
                            feature_names=temporal_feature_names,
                            show=False,
                        )
                    else:
                        shap.summary_plot(shap_values[0:1], X_flat[0:1], show=False)
                else:
                    # For regular models, show summary plot
                    shap.summary_plot(
                        shap_values, X, feature_names=self.feature_names, show=False
                    )

                # Get the figure
                fig = plt.gcf()

                # Return both SHAP values and the figure
                return {"shap_values": shap_values, "figure": fig}
            else:
                return {"shap_values": shap_values}
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            raise

    def explain_with_lime(
        self,
        X: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000,
        plot: bool = True,
    ) -> Any:
        """
        Explain predictions using LIME.

        Args:
            X: Input data to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            plot: Whether to create and return a plot

        Returns:
            lime_explanation: LIME explanation and/or plot
        """
        try:
            # Initialize the LIME explainer if not already done
            if "lime" not in self.explainers:
                # For sequential models, we need to flatten the input
                if self.is_sequential:
                    # Create new feature names if we have timesteps
                    if self.feature_names:
                        timesteps = X.shape[1] if len(X.shape) > 2 else 1
                        n_features = X.shape[2] if len(X.shape) > 2 else X.shape[1]
                        temporal_feature_names = []

                        for t in range(timesteps):
                            for f in range(min(n_features, len(self.feature_names))):
                                temporal_feature_names.append(
                                    f"{self.feature_names[f]} (t-{timesteps-t})"
                                )

                        feature_names = temporal_feature_names
                    else:
                        feature_names = None

                    # Flatten input
                    X_flat = self.flatten_sequential_input(X)

                    # Create a predict function that handles flattened input
                    predict_fn = lambda x: self.predict_wrapper(x.reshape(X.shape))

                    self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(
                        X_flat,
                        feature_names=feature_names,
                        class_names=["Price"],
                        mode="regression",
                    )
                else:
                    # Regular models
                    self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(
                        X,
                        feature_names=self.feature_names,
                        class_names=["Price"],
                        mode="regression",
                    )

            # We'll explain the first instance
            instance_to_explain = X[0]

            # Flatten if needed
            if self.is_sequential:
                instance_to_explain = self.flatten_sequential_input(X[0:1])[0]
                predict_fn = lambda x: self.predict_wrapper(
                    x.reshape((1,) + X.shape[1:])
                )[0]
            else:
                predict_fn = lambda x: self.predict_wrapper(x.reshape(1, -1))[0]

            # Generate explanation
            explanation = self.explainers["lime"].explain_instance(
                instance_to_explain,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            # Create plot if requested
            if plot:
                # Get the explanation as a pyplot figure
                fig = plt.figure(figsize=(10, 6))
                explanation.as_pyplot_figure()

                # Return both explanation and figure
                return {"lime_explanation": explanation, "figure": fig}
            else:
                return {"lime_explanation": explanation}
        except Exception as e:
            logger.error(f"Error in LIME explanation: {e}")
            raise

    def explain_with_integrated_gradients(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
        plot: bool = True,
    ) -> Any:
        """
        Explain predictions using Integrated Gradients.

        Args:
            X: Input data to explain
            baseline: Baseline data (None for zero baseline)
            n_steps: Number of steps for integration
            plot: Whether to create and return a plot

        Returns:
            ig_explanation: Integrated Gradients explanation and/or plot
        """
        try:
            # Only applicable for TensorFlow models
            if self.model_type != "tensorflow":
                raise ValueError(
                    "Integrated Gradients is only supported for TensorFlow models"
                )

            # Initialize the IG explainer if not already done
            if "integrated_gradients" not in self.explainers:
                # For sequential models, we need special handling
                if self.is_sequential:
                    # Create an intermediate model that returns both predictions and gradients
                    self.explainers["integrated_gradients"] = IntegratedGradients(
                        model=self.model, n_steps=n_steps, method="gausslegendre"
                    )
                else:
                    # Regular neural network
                    self.explainers["integrated_gradients"] = IntegratedGradients(
                        model=self.model, n_steps=n_steps, method="gausslegendre"
                    )

            # Create baseline if not provided
            if baseline is None:
                baseline = np.zeros_like(X)

            # Generate explanation
            explanation = self.explainers["integrated_gradients"].explain(
                X, baselines=baseline
            )

            # Get the attributions
            attributions = explanation.attributions[0]

            # Create plot if requested
            if plot:
                plt.figure(figsize=(12, 8))

                # For sequential data, average across timesteps
                if self.is_sequential and len(attributions.shape) > 2:
                    # Average attributions across timesteps
                    avg_attributions = np.mean(attributions, axis=1)

                    # Create bar plot of feature attributions
                    plt.bar(
                        range(avg_attributions.shape[1]),
                        avg_attributions[0],
                        tick_label=self.feature_names if self.feature_names else None,
                    )
                    plt.title("Average Feature Attributions Across Time")
                    plt.xticks(rotation=90)
                else:
                    # For regular models, show attributions directly
                    plt.bar(
                        range(attributions.shape[1]),
                        attributions[0],
                        tick_label=self.feature_names if self.feature_names else None,
                    )
                    plt.title("Feature Attributions")
                    plt.xticks(rotation=90)

                # Get the figure
                fig = plt.gcf()

                # Return both explanation and figure
                return {"integrated_gradients": attributions, "figure": fig}
            else:
                return {"integrated_gradients": attributions}
        except Exception as e:
            logger.error(f"Error in Integrated Gradients explanation: {e}")
            raise

    def explain_with_permutation_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, plot: bool = True
    ) -> Any:
        """
        Explain predictions using Permutation Importance.

        Args:
            X: Input data
            y: Target values
            n_repeats: Number of times to permute each feature
            plot: Whether to create and return a plot

        Returns:
            perm_importance: Permutation Importance and/or plot
        """
        try:
            # Initialize the permutation importance explainer
            # This is calculated on the fly, so no need to store in self.explainers

            # For sequential models, we need to flatten the input
            if self.is_sequential:
                # Flatten input
                X_flat = self.flatten_sequential_input(X)

                # Create a scoring function that handles flattened input
                def score_func(X_subset):
                    X_subset_reshaped = X_subset.reshape(X.shape)
                    preds = self.predict_wrapper(X_subset_reshaped)
                    return -np.mean((y - preds.flatten()) ** 2)  # Negative MSE

                # Calculate permutation importance
                perm = PermutationImportance(
                    score_func, random_state=42, n_iter=n_repeats
                ).fit(X_flat, y)

                # Create feature names for each timestep if available
                if self.feature_names:
                    timesteps = X.shape[1] if len(X.shape) > 2 else 1
                    n_features = X.shape[2] if len(X.shape) > 2 else X.shape[1]
                    temporal_feature_names = []

                    for t in range(timesteps):
                        for f in range(min(n_features, len(self.feature_names))):
                            temporal_feature_names.append(
                                f"{self.feature_names[f]} (t-{timesteps-t})"
                            )

                    feature_names = temporal_feature_names
                else:
                    feature_names = None
            else:
                # For sklearn and xgboost models
                if self.model_type in ["sklearn", "xgboost"]:
                    # Use sklearn's PermutationImportance
                    perm = PermutationImportance(
                        self.model, random_state=42, n_repeats=n_repeats
                    ).fit(X, y)
                    feature_names = self.feature_names
                else:
                    # For TensorFlow models
                    def score_func(X_subset):
                        preds = self.predict_wrapper(X_subset)
                        return -np.mean((y - preds.flatten()) ** 2)  # Negative MSE

                    perm = PermutationImportance(
                        score_func, random_state=42, n_iter=n_repeats
                    ).fit(X, y)
                    feature_names = self.feature_names

            # Get the permutation importance scores
            importances = perm.feature_importances_
            std = perm.feature_importances_std_

            # Create plot if requested
            if plot:
                plt.figure(figsize=(12, 8))

                # Sort by importance
                indices = np.argsort(importances)[::-1]
                sorted_importances = importances[indices]
                sorted_std = std[indices]

                if feature_names:
                    sorted_names = [feature_names[i] for i in indices]
                else:
                    sorted_names = [f"Feature {i}" for i in indices]

                # Plot bar chart with error bars
                plt.bar(
                    range(len(sorted_importances)),
                    sorted_importances,
                    yerr=sorted_std,
                    tick_label=sorted_names,
                )
                plt.title("Permutation Feature Importance")
                plt.xticks(rotation=90)
                plt.tight_layout()

                # Get the figure
                fig = plt.gcf()

                # Return both importance scores and figure
                return {
                    "permutation_importance": {
                        "importances": importances,
                        "std": std,
                        "indices": indices,
                    },
                    "figure": fig,
                }
            else:
                return {
                    "permutation_importance": {"importances": importances, "std": std}
                }
        except Exception as e:
            logger.error(f"Error in Permutation Importance explanation: {e}")
            raise

    def explain_with_anchor(
        self,
        X: np.ndarray,
        threshold: float = 0.95,
        coverage_samples: int = 1000,
        plot: bool = True,
    ) -> Any:
        """
        Explain predictions using Anchor Explanations.

        Args:
            X: Input data to explain
            threshold: Minimum precision threshold for anchors
            coverage_samples: Number of samples for coverage estimation
            plot: Whether to create and return a plot

        Returns:
            anchor_explanation: Anchor explanation and/or plot
        """
        try:
            # Anchor Explanations don't work well with sequential data
            if self.is_sequential:
                raise ValueError(
                    "Anchor Explanations are not supported for sequential data"
                )

            # Initialize the anchor explainer if not already done
            if "anchor" not in self.explainers:
                # Create a predict function
                predict_fn = lambda x: self.predict_wrapper(x)

                # Create categorical features mask (all False for numeric data)
                categorical_features = []

                # Initialize Anchor explainer
                self.explainers["anchor"] = AnchorTabular(
                    predict_fn, feature_names=self.feature_names
                )

                # Fit the explainer on the data
                self.explainers["anchor"].fit(
                    X, categorical_features=categorical_features
                )

            # We'll explain the first instance
            instance_to_explain = X[0:1]

            # Generate explanation
            explanation = self.explainers["anchor"].explain(
                instance_to_explain, threshold=threshold
            )

            # Create plot if requested
            if plot:
                plt.figure(figsize=(10, 6))

                # Extract the anchor rules
                anchor_rules = explanation.anchor

                # Plot the anchor rules
                plt.barh(
                    range(len(anchor_rules)),
                    [1] * len(anchor_rules),
                    tick_label=anchor_rules,
                )
                plt.title(
                    f"Anchor Explanation (Precision: {explanation.precision:.2f})"
                )
                plt.tight_layout()

                # Get the figure
                fig = plt.gcf()

                # Return both explanation and figure
                return {"anchor_explanation": explanation, "figure": fig}
            else:
                return {"anchor_explanation": explanation}
        except Exception as e:
            logger.error(f"Error in Anchor explanation: {e}")
            raise

    def create_partial_dependence_plot(
        self, X: np.ndarray, feature_idx: int, num_points: int = 50, ice: bool = False
    ) -> Any:
        """
        Create Partial Dependence Plot (PDP) or Individual Conditional Expectation (ICE).

        Args:
            X: Input data
            feature_idx: Index of the feature to analyze
            num_points: Number of points to sample
            ice: Whether to create ICE (Individual Conditional Expectation) plots

        Returns:
            pdp_data: PDP/ICE data and plot
        """
        try:
            # Not applicable for sequential models
            if self.is_sequential:
                raise ValueError(
                    "Partial Dependence Plots are not supported for sequential data"
                )

            # Get feature values range
            feature_min = X[:, feature_idx].min()
            feature_max = X[:, feature_idx].max()

            # Create grid of values
            grid = np.linspace(feature_min, feature_max, num_points)

            # Initialize result arrays
            pdp_values = np.zeros(num_points)
            ice_values = np.zeros((len(X), num_points)) if ice else None

            # Calculate PDP/ICE
            for i, val in enumerate(grid):
                # Create a copy of X with the feature value modified
                X_mod = X.copy()
                X_mod[:, feature_idx] = val

                # Get predictions
                predictions = self.predict_wrapper(X_mod).flatten()

                # Store values
                pdp_values[i] = predictions.mean()

                if ice:
                    ice_values[:, i] = predictions

            # Create plot
            plt.figure(figsize=(10, 6))

            if ice:
                # Plot ICE curves
                for i in range(min(len(X), 30)):  # Limit to 30 curves to avoid clutter
                    plt.plot(grid, ice_values[i], color="blue", alpha=0.1)

            # Plot PDP curve
            plt.plot(grid, pdp_values, color="red", linewidth=2)

            # Set plot labels
            feature_name = (
                self.feature_names[feature_idx]
                if self.feature_names
                else f"Feature {feature_idx}"
            )
            plt.xlabel(feature_name)
            plt.ylabel("Predicted Price")
            plt.title(f"{'ICE and ' if ice else ''}PDP for {feature_name}")

            # Get the figure
            fig = plt.gcf()

            # Return the data and figure
            return {
                "pdp_data": {
                    "grid": grid,
                    "pdp_values": pdp_values,
                    "ice_values": ice_values,
                },
                "figure": fig,
            }
        except Exception as e:
            logger.error(f"Error creating PDP: {e}")
            raise

    def analyze_feature_interactions(
        self, X: np.ndarray, feature1_idx: int, feature2_idx: int, num_points: int = 20
    ) -> Any:
        """
        Analyze interactions between two features.

        Args:
            X: Input data
            feature1_idx: Index of the first feature
            feature2_idx: Index of the second feature
            num_points: Number of points to sample for each feature

        Returns:
            interaction_data: Interaction data and plot
        """
        try:
            # Not applicable for sequential models
            if self.is_sequential:
                raise ValueError(
                    "Feature interaction analysis is not supported for sequential data"
                )

            # Get feature values range
            feature1_min = X[:, feature1_idx].min()
            feature1_max = X[:, feature1_idx].max()
            feature2_min = X[:, feature2_idx].min()
            feature2_max = X[:, feature2_idx].max()

            # Create grids
            grid1 = np.linspace(feature1_min, feature1_max, num_points)
            grid2 = np.linspace(feature2_min, feature2_max, num_points)

            # Create meshgrid for heatmap
            xx, yy = np.meshgrid(grid1, grid2)
            zz = np.zeros((num_points, num_points))

            # Calculate interaction values
            for i, val1 in enumerate(grid1):
                for j, val2 in enumerate(grid2):
                    # Create a copy of X with both feature values modified
                    X_mod = X.copy()
                    X_mod[:, feature1_idx] = val1
                    X_mod[:, feature2_idx] = val2

                    # Get predictions
                    predictions = self.predict_wrapper(X_mod).flatten()

                    # Store mean prediction
                    zz[j, i] = predictions.mean()

            # Create plot
            plt.figure(figsize=(10, 8))

            # Create heatmap
            plt.pcolormesh(xx, yy, zz, cmap="viridis")
            plt.colorbar(label="Predicted Price")

            # Set plot labels
            feature1_name = (
                self.feature_names[feature1_idx]
                if self.feature_names
                else f"Feature {feature1_idx}"
            )
            feature2_name = (
                self.feature_names[feature2_idx]
                if self.feature_names
                else f"Feature {feature2_idx}"
            )
            plt.xlabel(feature1_name)
            plt.ylabel(feature2_name)
            plt.title(f"Feature Interaction: {feature1_name} vs. {feature2_name}")

            # Get the figure
            fig = plt.gcf()

            # Return the data and figure
            return {
                "interaction_data": {"grid1": grid1, "grid2": grid2, "values": zz},
                "figure": fig,
            }
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {e}")
            raise

    def get_saliency_map(self, X: np.ndarray) -> Any:
        """
        Generate saliency map for the input.

        Args:
            X: Input data

        Returns:
            saliency_data: Saliency map data and plot
        """
        try:
            # Only applicable for TensorFlow models
            if self.model_type != "tensorflow":
                raise ValueError(
                    "Saliency maps are only supported for TensorFlow models"
                )

            # Create TensorFlow Gradient Tape to calculate gradients
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)

            # Calculate gradients
            gradients = tape.gradient(predictions, X_tensor)

            # Take absolute value for saliency
            saliency = tf.abs(gradients).numpy()

            # Check if we have sequential data
            if self.is_sequential:
                # Average across samples (batch) if needed
                if saliency.shape[0] > 1:
                    saliency = np.mean(saliency, axis=0, keepdims=True)

                # Create heatmap
                plt.figure(figsize=(12, 8))
                plt.imshow(saliency[0], aspect="auto", cmap="viridis")

                # Set labels
                plt.xlabel("Features")
                plt.ylabel("Timesteps")

                # Set feature names if available
                if self.feature_names:
                    plt.xticks(
                        range(len(self.feature_names)), self.feature_names, rotation=90
                    )

                plt.title("Saliency Map (Absolute Gradients)")
                plt.colorbar(label="Saliency")
            else:
                # For regular models, show bar chart
                plt.figure(figsize=(12, 8))

                # Average across samples
                avg_saliency = np.mean(saliency, axis=0)

                # Create bar chart
                plt.bar(
                    range(len(avg_saliency)),
                    avg_saliency,
                    tick_label=self.feature_names if self.feature_names else None,
                )
                plt.title("Feature Saliency")
                plt.xlabel("Features")
                plt.ylabel("Saliency (Absolute Gradient)")
                plt.xticks(rotation=90)

            # Get the figure
            fig = plt.gcf()

            # Return the data and figure
            return {"saliency_data": saliency, "figure": fig}
        except Exception as e:
            logger.error(f"Error creating saliency map: {e}")
            raise

    def create_counterfactual_example(
        self,
        X: np.ndarray,
        target_prediction: float,
        feature_constraints: Dict[int, Tuple[float, float]] = None,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
    ) -> Any:
        """
        Generate a counterfactual example with gradient descent.

        Args:
            X: Input data (single instance)
            target_prediction: Target prediction value
            feature_constraints: Dictionary mapping feature indices to (min, max) constraints
            max_iterations: Maximum number of iterations for optimization
            learning_rate: Learning rate for gradient descent

        Returns:
            counterfactual_data: Counterfactual example data and plot
        """
        try:
            # Not applicable for sequential models
            if self.is_sequential:
                raise ValueError(
                    "Counterfactual examples are not supported for sequential data"
                )

            # Make sure we're working with a single instance
            if len(X.shape) > 2 or (len(X.shape) == 2 and X.shape[0] > 1):
                X = X[0:1]

            # Only applicable for TensorFlow models
            if self.model_type != "tensorflow":
                # Could implement for other models using a custom optimizer
                raise ValueError(
                    "Counterfactual examples are only supported for TensorFlow models"
                )

            # Create TensorFlow Variable for gradient-based optimization
            X_cf = tf.Variable(X, dtype=tf.float32)

            # Define optimizer
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            # Optimization loop
            for i in range(max_iterations):
                with tf.GradientTape() as tape:
                    # Get current prediction
                    current_pred = self.model(X_cf)[0][0]

                    # Define loss as MSE to target prediction
                    loss = tf.square(current_pred - target_prediction)

                # Calculate gradients
                gradients = tape.gradient(loss, X_cf)

                # Apply gradients
                optimizer.apply_gradients([(gradients, X_cf)])

                # Apply constraints
                if feature_constraints:
                    X_cf_np = X_cf.numpy()

                    for feat_idx, (min_val, max_val) in feature_constraints.items():
                        X_cf_np[0, feat_idx] = max(
                            min_val, min(max_val, X_cf_np[0, feat_idx])
                        )

                    X_cf.assign(X_cf_np)

                # Check convergence
                if i % 100 == 0:
                    current_pred_np = current_pred.numpy()
                    if abs(current_pred_np - target_prediction) < 0.01:
                        break

            # Get the final counterfactual
            counterfactual = X_cf.numpy()

            # Get predictions for original and counterfactual
            original_pred = self.model(X).numpy()[0][0]
            counterfactual_pred = self.model(counterfactual).numpy()[0][0]

            # Create plot
            plt.figure(figsize=(14, 8))

            # Calculate feature changes
            changes = counterfactual[0] - X[0]

            # Sort by magnitude of change
            indices = np.argsort(np.abs(changes))[::-1]

            # Create bar chart with original and counterfactual values
            feature_names = (
                self.feature_names
                if self.feature_names
                else [f"Feature {i}" for i in range(len(X[0]))]
            )
            sorted_names = [feature_names[i] for i in indices]

            # Limit to top changes for clarity
            top_n = min(10, len(indices))
            indices = indices[:top_n]
            sorted_names = sorted_names[:top_n]

            # Create bar chart
            plt.figure(figsize=(12, 6))
            x = np.arange(len(indices))
            width = 0.35

            plt.bar(x - width / 2, X[0][indices], width, label="Original")
            plt.bar(
                x + width / 2, counterfactual[0][indices], width, label="Counterfactual"
            )

            plt.xlabel("Features")
            plt.ylabel("Value")
            plt.title(
                f"Counterfactual Example: {original_pred:.2f} → {counterfactual_pred:.2f}"
            )
            plt.xticks(x, sorted_names, rotation=90)
            plt.legend()
            plt.tight_layout()

            # Get the figure
            fig = plt.gcf()

            # Return the data and figure
            return {
                "counterfactual_data": {
                    "original": X[0],
                    "counterfactual": counterfactual[0],
                    "original_prediction": float(original_pred),
                    "counterfactual_prediction": float(counterfactual_pred),
                    "changes": changes,
                },
                "figure": fig,
            }
        except Exception as e:
            logger.error(f"Error creating counterfactual example: {e}")
            raise

    def figure_to_html(self, fig: plt.Figure) -> str:
        """
        Convert Matplotlib figure to HTML string.

        Args:
            fig: Matplotlib figure

        Returns:
            html: HTML string with the figure
        """
        # Convert matplotlib figure to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Create HTML
        html = f'<img src="data:image/png;base64,{img_data}" alt="Figure" style="max-width:100%;">'
        return html

    def create_feature_importance_plot(
        self, feature_importances: Dict[str, float]
    ) -> go.Figure:
        """
        Create a Plotly bar chart for feature importances.

        Args:
            feature_importances: Dictionary mapping feature names to importance values

        Returns:
            fig: Plotly figure
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]

        # Create bar chart
        fig = px.bar(
            x=feature_names,
            y=importance_values,
            labels={"x": "Features", "y": "Importance"},
            title="Feature Importance",
        )

        # Customize layout
        fig.update_layout(
            xaxis_tickangle=-45, xaxis={"categoryorder": "total descending"}, height=500
        )

        return fig

    def create_summary_dashboard(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary dashboard with multiple explanations.

        Args:
            X: Input data
            y: Optional target values for certain explanations

        Returns:
            dashboard: Dictionary with multiple explanation results
        """
        dashboard = {}

        # Add SHAP explanation
        try:
            dashboard["shap"] = self.explain_with_shap(X, plot=True)
        except Exception as e:
            logger.warning(f"Could not create SHAP explanation: {e}")
            dashboard["shap"] = {"error": str(e)}

        # Add LIME explanation
        try:
            dashboard["lime"] = self.explain_with_lime(X, plot=True)
        except Exception as e:
            logger.warning(f"Could not create LIME explanation: {e}")
            dashboard["lime"] = {"error": str(e)}

        # Add Integrated Gradients explanation (only for TensorFlow)
        if self.model_type == "tensorflow":
            try:
                dashboard["integrated_gradients"] = (
                    self.explain_with_integrated_gradients(X, plot=True)
                )
            except Exception as e:
                logger.warning(
                    f"Could not create Integrated Gradients explanation: {e}"
                )
                dashboard["integrated_gradients"] = {"error": str(e)}

        # Add Permutation Importance (if y is provided)
        if y is not None:
            try:
                dashboard["permutation_importance"] = (
                    self.explain_with_permutation_importance(X, y, plot=True)
                )
            except Exception as e:
                logger.warning(
                    f"Could not create Permutation Importance explanation: {e}"
                )
                dashboard["permutation_importance"] = {"error": str(e)}

        # Add Saliency Map (only for TensorFlow)
        if self.model_type == "tensorflow":
            try:
                dashboard["saliency"] = self.get_saliency_map(X)
            except Exception as e:
                logger.warning(f"Could not create Saliency Map: {e}")
                dashboard["saliency"] = {"error": str(e)}

        # Add PDP for the most important feature (if we have feature names)
        if self.feature_names and not self.is_sequential:
            try:
                # Use SHAP values if available, otherwise first feature
                if "shap" in dashboard and "shap_values" in dashboard["shap"]:
                    shap_values = dashboard["shap"]["shap_values"]
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    most_important_feature = np.argmax(feature_importance)
                else:
                    most_important_feature = 0

                dashboard["pdp"] = self.create_partial_dependence_plot(
                    X, most_important_feature, ice=True
                )
            except Exception as e:
                logger.warning(f"Could not create PDP: {e}")
                dashboard["pdp"] = {"error": str(e)}

        return dashboard


def explain_model_prediction(
    model, features, feature_names=None, model_type="tensorflow"
):
    """
    Utility function to explain a model's prediction for a single instance.

    Args:
        model: The trained model
        features: Feature values for the instance to explain
        feature_names: Names of the features
        model_type: Type of model ('tensorflow', 'sklearn', 'xgboost')

    Returns:
        explanations: Dictionary with multiple explanation methods
    """
    # Ensure features is a 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    # Create explainer
    explainer = XAIExplainer(model, model_type, feature_names)

    # Get predictions
    predictions = explainer.predict_wrapper(features)

    # Generate explanations
    explanations = {
        "prediction": (
            float(predictions[0][0])
            if len(predictions.shape) > 1
            else float(predictions[0])
        ),
        "features": (
            {name: float(value) for name, value in zip(feature_names, features[0])}
            if feature_names
            else features[0].tolist()
        ),
    }

    # Add SHAP explanation
    try:
        shap_result = explainer.explain_with_shap(features, plot=True)
        explanations["shap"] = shap_result
    except Exception as e:
        explanations["shap"] = {"error": str(e)}

    # Add LIME explanation
    try:
        lime_result = explainer.explain_with_lime(features, plot=True)
        explanations["lime"] = lime_result
    except Exception as e:
        explanations["lime"] = {"error": str(e)}

    # Add Integrated Gradients for TensorFlow models
    if model_type == "tensorflow":
        try:
            ig_result = explainer.explain_with_integrated_gradients(features, plot=True)
            explanations["integrated_gradients"] = ig_result
        except Exception as e:
            explanations["integrated_gradients"] = {"error": str(e)}

    return explanations


def get_feature_importance(
    model, X, feature_names=None, model_type="tensorflow", method="shap"
):
    """
    Get feature importance for a model.

    Args:
        model: The trained model
        X: Input data
        feature_names: Names of the features
        model_type: Type of model ('tensorflow', 'sklearn', 'xgboost')
        method: Method to use for feature importance ('shap', 'permutation', 'integrated_gradients')

    Returns:
        importance: Dictionary mapping feature names to importance values
    """
    # Create explainer
    explainer = XAIExplainer(model, model_type, feature_names)

    if method == "shap":
        # Use SHAP for feature importance
        result = explainer.explain_with_shap(X, plot=False)
        shap_values = result["shap_values"]

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # For multi-output models, use the first output
            feature_importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            feature_importance = np.abs(shap_values).mean(axis=0)
    elif method == "permutation":
        # Use Permutation Importance
        y_pred = explainer.predict_wrapper(X).flatten()
        result = explainer.explain_with_permutation_importance(X, y_pred, plot=False)
        feature_importance = result["permutation_importance"]["importances"]
    elif method == "integrated_gradients":
        # Use Integrated Gradients (only for TensorFlow)
        if model_type != "tensorflow":
            raise ValueError(
                "Integrated Gradients is only supported for TensorFlow models"
            )

        result = explainer.explain_with_integrated_gradients(X, plot=False)
        attributions = result["integrated_gradients"]

        # Take mean absolute attributions
        if len(attributions.shape) > 2:
            # For sequential data, average across timesteps
            feature_importance = np.abs(attributions).mean(axis=(0, 1))
        else:
            feature_importance = np.abs(attributions).mean(axis=0)
    else:
        raise ValueError(f"Unknown feature importance method: {method}")

    # Create dictionary mapping feature names to importance values
    if feature_names:
        importance = {
            name: float(imp) for name, imp in zip(feature_names, feature_importance)
        }
    else:
        importance = {
            f"Feature_{i}": float(imp) for i, imp in enumerate(feature_importance)
        }

    return importance


def create_xai_component_for_dashboard(
    model, X_sample, feature_names, predictions=None, actuals=None
):
    """
    Create XAI components for the Streamlit dashboard.

    Args:
        model: The trained model
        X_sample: Sample input data to explain
        feature_names: Names of the features
        predictions: Model predictions
        actuals: Actual values

    Returns:
        explanation_data: Dictionary with explanation data for the dashboard
    """
    # Detect model type
    if isinstance(model, tf.keras.Model):
        model_type = "tensorflow"
    elif hasattr(model, "feature_importances_"):
        model_type = "xgboost" if "xgboost" in str(type(model)).lower() else "sklearn"
    else:
        model_type = "sklearn"

    # Create explainer
    explainer = XAIExplainer(model, model_type, feature_names)

    # Prepare output dictionary
    explanation_data = {}

    # Add SHAP explanation
    try:
        shap_result = explainer.explain_with_shap(X_sample, plot=True)
        explanation_data["shap"] = {
            "figure": explainer.figure_to_html(shap_result["figure"]),
            "values": (
                shap_result["shap_values"] if "shap_values" in shap_result else None
            ),
        }
    except Exception as e:
        explanation_data["shap"] = {"error": str(e)}

    # Add feature importance
    try:
        importance = get_feature_importance(model, X_sample, feature_names, model_type)
        explanation_data["feature_importance"] = importance
    except Exception as e:
        explanation_data["feature_importance"] = {"error": str(e)}

    # Add PDP for top feature
    if feature_names:
        try:
            # Find top feature
            top_feature = max(
                explanation_data["feature_importance"].items(), key=lambda x: x[1]
            )[0]
            top_idx = feature_names.index(top_feature)

            pdp_result = explainer.create_partial_dependence_plot(
                X_sample, top_idx, ice=True
            )
            explanation_data["pdp"] = {
                "feature": top_feature,
                "figure": explainer.figure_to_html(pdp_result["figure"]),
                "data": pdp_result["pdp_data"],
            }
        except Exception as e:
            explanation_data["pdp"] = {"error": str(e)}

    # Add feature interactions
    if feature_names and len(feature_names) > 1:
        try:
            # Get top 2 features
            sorted_features = sorted(
                explanation_data["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_features = [f[0] for f in sorted_features[:2]]
            feature_idx1 = feature_names.index(top_features[0])
            feature_idx2 = feature_names.index(top_features[1])

            interaction_result = explainer.analyze_feature_interactions(
                X_sample, feature_idx1, feature_idx2
            )
            explanation_data["interactions"] = {
                "features": (top_features[0], top_features[1]),
                "figure": explainer.figure_to_html(interaction_result["figure"]),
                "data": interaction_result["interaction_data"],
            }
        except Exception as e:
            explanation_data["interactions"] = {"error": str(e)}

    # Add counterfactual example (only for simple models)
    if model_type == "tensorflow" and not explainer.is_sequential:
        try:
            # Create a target prediction (increase by 10%)
            current_pred = explainer.predict_wrapper(X_sample[:1])[0][0]
            target_pred = current_pred * 1.10  # 10% increase

            cf_result = explainer.create_counterfactual_example(
                X_sample[:1], target_pred
            )
            explanation_data["counterfactual"] = {
                "figure": explainer.figure_to_html(cf_result["figure"]),
                "data": cf_result["counterfactual_data"],
            }
        except Exception as e:
            explanation_data["counterfactual"] = {"error": str(e)}

    return explanation_data


if __name__ == "__main__":
    # Simple example of using the XAI explainer

    # Create a simple model
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=5, random_state=42
    )

    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create explainer
    explainer = XAIExplainer(model, "sklearn", feature_names)

    # Generate explanations
    result = explainer.explain_with_shap(X[:10], plot=True)

    # Show the plot
    plt.show()

    # Get feature importance
    importance = get_feature_importance(model, X[:10], feature_names, "sklearn")
    print("Feature Importance:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.4f}")


def render_explainable_ai_tab():
    """
    Render the Explainable AI tab in the dashboard.
    This function provides an interactive interface for exploring model explanations.
    """
    import streamlit as st

    st.header("Explainable AI Analysis")

    # Get the current model from session state
    model = st.session_state.get("model")

    if model is None:
        st.warning("Please load or train a model to use Explainable AI features.")
        return

    # Get sample data for explanation
    df = st.session_state.get("df_raw")
    if df is None or df.empty:
        st.warning("No data available for analysis.")
        return

    # Determine feature columns (exclude date and target)
    feature_cols = [col for col in df.columns if col not in ["date", "Date", "Close"]]

    # Use XAIWrapper from xai_integration to create interface
    try:
        from src.dashboard.xai_integration import create_xai_explorer

        # Create sample data for explanation (last 30 days)
        X_sample = df.iloc[-30:][feature_cols].values

        # Get feature names
        feature_names = feature_cols

        # Create the XAI explorer
        xai_wrapper = create_xai_explorer(st, model, X_sample, feature_names)

        # Add additional XAI information
        st.subheader("Understanding Model Decisions")

        st.markdown(
            """
        Explainable AI helps us understand how the model makes predictions. Use the tools above to:
        
        1. **SHAP Analysis** - See how each feature contributes to predictions
        2. **Feature Importance** - Identify which features are most influential
        3. **Partial Dependence Plots** - Understand how changes in a feature affect predictions
        4. **What-If Analysis** - Explore how changing inputs affects the output
        
        These techniques help build trust in the model by making its decision-making process transparent.
        """
        )

    except Exception as e:
        st.error(f"Error initializing Explainable AI components: {str(e)}")
        st.info("The XAI module may not be properly installed or configured.")
