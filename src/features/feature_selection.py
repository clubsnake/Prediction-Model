import os
import sys
import time

# Add project root to sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor


class FeatureSelector:
    """
    Comprehensive feature selection toolkit with multiple methods:
    - Statistical filtering (variance, correlation)
    - Mutual information
    - Model-based selection (Random Forest, Gradient Boosting, Lasso)
    - Recursive feature elimination (RFE)
    - Optimal feature subset search
    """

    def __init__(
        self,
        df=None,
        target_col="Close",
        feature_cols=None,
        min_features=5,
        max_features=None,
        verbose=True,
    ):
        """
        Initialize the feature selector.

        Args:
            df: DataFrame containing features and target
            target_col: Name of the target column
            feature_cols: List of feature column names (if None, all columns except target are used)
            min_features: Minimum number of features to select
            max_features: Maximum number of features to select
            verbose: Whether to print progress and results
        """
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.min_features = min_features
        self.max_features = max_features
        self.verbose = verbose
        self.results = {}
        self.selected_features = {}

        if df is not None:
            self._prepare_data()

    def _prepare_data(self):
        """Prepare the data for feature selection."""
        if self.feature_cols is None:
            self.feature_cols = [
                col
                for col in self.df.columns
                if col != self.target_col and col != "date"
            ]

        if self.max_features is None:
            self.max_features = len(self.feature_cols)

        # Ensure all columns exist
        missing_cols = [
            col
            for col in self.feature_cols + [self.target_col]
            if col not in self.df.columns
        ]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Extract X and y
        self.X = self.df[self.feature_cols].values
        self.y = self.df[self.target_col].values

        if self.verbose:
            print(
                f"Data prepared with {len(self.feature_cols)} features and {len(self.y)} samples"
            )

    def set_data(self, df, target_col="Close", feature_cols=None):
        """Set or update the data for feature selection."""
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self._prepare_data()

    def filter_by_variance(self, threshold=0.01):
        """
        Remove features with low variance.

        Args:
            threshold: Variance threshold for feature selection

        Returns:
            List of selected feature names
        """
        if self.verbose:
            print(f"Filtering features by variance (threshold={threshold})...")

        start_time = time.time()
        selector = VarianceThreshold(threshold=threshold)
        selector.fit_transform(self.X)

        # Get the selected feature indices and names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_cols[i] for i in selected_indices]

        self.results["variance_threshold"] = {
            "method": "Variance Threshold",
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "threshold": threshold,
            "time_taken": time.time() - start_time,
        }

        self.selected_features["variance_threshold"] = selected_features

        if self.verbose:
            print(
                f"Selected {len(selected_features)} features using variance threshold"
            )
            print(
                f"Time taken: {self.results['variance_threshold']['time_taken']:.2f} seconds"
            )

        return selected_features

    def filter_by_correlation(self, threshold=0.95, method="pearson"):
        """
        Remove highly correlated features.

        Args:
            threshold: Correlation threshold (features with correlation above this are considered redundant)
            method: Correlation method ('pearson', 'kendall', 'spearman')

        Returns:
            List of selected feature names
        """
        if self.verbose:
            print(
                f"Filtering highly correlated features (threshold={threshold}, method={method})..."
            )

        start_time = time.time()

        # Calculate correlation matrix
        corr_matrix = (
            pd.DataFrame(self.X, columns=self.feature_cols).corr(method=method).abs()
        )

        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Create the list of features to keep
        selected_features = [f for f in self.feature_cols if f not in to_drop]

        self.results["correlation_filter"] = {
            "method": f"Correlation Filter ({method})",
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "threshold": threshold,
            "dropped_features": to_drop,
            "time_taken": time.time() - start_time,
        }

        self.selected_features["correlation_filter"] = selected_features

        if self.verbose:
            print(
                f"Selected {len(selected_features)} features after dropping {len(to_drop)} highly correlated features"
            )
            print(
                f"Time taken: {self.results['correlation_filter']['time_taken']:.2f} seconds"
            )

        return selected_features

    def select_by_mutual_info(self, k="auto", strategy="percentile", percentile=80):
        """
        Select features based on mutual information.

        Args:
            k: Number of features to select or 'auto' for automatic selection
            strategy: Selection strategy ('fixed' for top k, 'percentile' for percentile)
            percentile: Percentile for percentile strategy

        Returns:
            List of selected feature names
        """
        if self.verbose:
            print("Selecting features using mutual information...")

        start_time = time.time()

        # Determine k if auto
        if k == "auto":
            if strategy == "percentile":
                # Automatically set k based on percentile
                k = max(
                    self.min_features, int(len(self.feature_cols) * percentile / 100)
                )
            else:
                # Default to half of features
                k = max(self.min_features, len(self.feature_cols) // 2)

        # Make sure k is within bounds
        k = min(max(k, self.min_features), len(self.feature_cols))

        # Apply mutual information selection
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit_transform(self.X, self.y)

        # Get selected feature indices and names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_cols[i] for i in selected_indices]

        # Get scores for reporting
        scores = selector.scores_
        feature_scores = [(f, s) for f, s in zip(self.feature_cols, scores)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        self.results["mutual_info"] = {
            "method": "Mutual Information",
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "k": k,
            "feature_scores": feature_scores,
            "time_taken": time.time() - start_time,
        }

        self.selected_features["mutual_info"] = selected_features

        if self.verbose:
            print(
                f"Selected {len(selected_features)} features using mutual information"
            )
            print(
                f"Time taken: {self.results['mutual_info']['time_taken']:.2f} seconds"
            )
            print("Top 10 features by mutual information:")
            for f, s in feature_scores[:10]:
                print(f"  {f}: {s:.4f}")

        return selected_features

    def select_by_model(self, model_type="rf", threshold="mean", max_features=None):
        """
        Select features using a model-based approach.

        Args:
            model_type: Type of model to use ('rf' for Random Forest,
                       'gb' for Gradient Boosting, 'lasso' for Lasso)
            threshold: Threshold for feature importance ('mean', 'median', or a float)
            max_features: Maximum number of features to select (overrides threshold if provided)

        Returns:
            List of selected feature names
        """
        if self.verbose:
            print(f"Selecting features using {model_type.upper()} model...")

        start_time = time.time()

        # Create the model
        if model_type.lower() == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            method_name = "Random Forest"
        elif model_type.lower() == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            method_name = "Gradient Boosting"
        elif model_type.lower() == "lasso":
            model = Lasso(alpha=0.01, random_state=42)
            method_name = "Lasso"
        elif model_type.lower() == "xgb":
            model = XGBRegressor(n_estimators=100, random_state=42)
            method_name = "XGBoost"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Apply the selection
        if max_features is not None:
            selector = SelectFromModel(
                model, max_features=max_features, threshold=-np.inf
            )
        else:
            selector = SelectFromModel(model, threshold=threshold)

        selector.fit_transform(self.X, self.y)

        # Get selected feature indices and names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_cols[i] for i in selected_indices]

        # Get importance scores
        selector.estimator_.fit(self.X, self.y)
        importances = (
            selector.estimator_.feature_importances_
            if hasattr(selector.estimator_, "feature_importances_")
            else None
        )

        if importances is not None:
            feature_importances = [
                (f, i) for f, i in zip(self.feature_cols, importances)
            ]
            feature_importances.sort(key=lambda x: x[1], reverse=True)
        else:
            feature_importances = None

        self.results[f"model_{model_type}"] = {
            "method": f"{method_name} Model",
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "threshold": threshold,
            "feature_importances": feature_importances,
            "time_taken": time.time() - start_time,
        }

        self.selected_features[f"model_{model_type}"] = selected_features

        if self.verbose:
            print(f"Selected {len(selected_features)} features using {method_name}")
            print(
                f"Time taken: {self.results[f'model_{model_type}']['time_taken']:.2f} seconds"
            )
            if feature_importances:
                print("Top 10 features by importance:")
                for f, i in feature_importances[:10]:
                    print(f"  {f}: {i:.4f}")

        return selected_features

    def select_by_rfe(self, model_type="rf", n_features=None, step=1, cv=None):
        """
        Select features using Recursive Feature Elimination.

        Args:
            model_type: Type of model to use
            n_features: Number of features to select (if None, use min_features)
            step: Number of features to remove at each iteration
            cv: Number of cross-validation folds (if provided, use RFECV)

        Returns:
            List of selected feature names
        """
        if self.verbose:
            print(
                f"Selecting features using {'CV-' if cv else ''}RFE with {model_type.upper()}..."
            )

        start_time = time.time()

        # Set the target number of features
        if n_features is None:
            n_features = self.min_features

        # Create the model
        if model_type.lower() == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            method_name = "Random Forest"
        elif model_type.lower() == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            method_name = "Gradient Boosting"
        elif model_type.lower() == "lasso":
            model = Lasso(alpha=0.01, random_state=42)
            method_name = "Lasso"
        elif model_type.lower() == "xgb":
            model = XGBRegressor(n_estimators=100, random_state=42)
            method_name = "XGBoost"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Apply RFE or RFECV
        if cv:
            selector = RFECV(
                estimator=model,
                step=step,
                cv=cv,
                scoring="neg_mean_squared_error",
                min_features_to_select=n_features,
            )
            method_key = f"rfecv_{model_type}"
            method_display = f"RFECV with {method_name}"
        else:
            selector = RFE(estimator=model, n_features_to_select=n_features, step=step)
            method_key = f"rfe_{model_type}"
            method_display = f"RFE with {method_name}"

        selector.fit_transform(self.X, self.y)

        # Get selected feature indices and names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_cols[i] for i in selected_indices]

        # Get feature rankings if available
        rankings = selector.ranking_ if hasattr(selector, "ranking_") else None

        if rankings is not None:
            feature_rankings = [(f, r) for f, r in zip(self.feature_cols, rankings)]
            feature_rankings.sort(key=lambda x: x[1])
        else:
            feature_rankings = None

        self.results[method_key] = {
            "method": method_display,
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "n_features": n_features,
            "feature_rankings": feature_rankings,
            "time_taken": time.time() - start_time,
        }

        if cv and hasattr(selector, "grid_scores_"):
            self.results[method_key]["cv_scores"] = selector.grid_scores_

        self.selected_features[method_key] = selected_features

        if self.verbose:
            print(f"Selected {len(selected_features)} features using {method_display}")
            print(f"Time taken: {self.results[method_key]['time_taken']:.2f} seconds")
            if feature_rankings:
                print("Top 10 features by ranking:")
                for f, r in feature_rankings[:10]:
                    print(f"  {f}: {r}")

        return selected_features

    def find_optimal_feature_set(
        self, model_type="rf", max_subset_size=20, method="add"
    ):
        """
        Find the optimal feature subset by sequentially adding or removing features.

        Args:
            model_type: Type of model to use for evaluation
            max_subset_size: Maximum number of features to consider
            method: 'add' for forward selection, 'remove' for backward elimination

        Returns:
            List of selected features in order of importance
        """
        if self.verbose:
            print(f"Finding optimal feature set using {method} method...")

        start_time = time.time()

        # Create the model
        if model_type.lower() == "rf":
            model_class = RandomForestRegressor
            model_params = {"n_estimators": 100, "random_state": 42}
        elif model_type.lower() == "gb":
            model_class = GradientBoostingRegressor
            model_params = {"n_estimators": 100, "random_state": 42}
        elif model_type.lower() == "xgb":
            model_class = XGBRegressor
            model_params = {"n_estimators": 100, "random_state": 42}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Ensure reasonable subset size
        max_subset_size = min(max_subset_size, len(self.feature_cols))

        # Initialize variables
        features_df = pd.DataFrame(self.X, columns=self.feature_cols)
        best_score = float("inf")
        best_features = []
        performance_history = []

        # Forward selection
        if method == "add":
            remaining_features = self.feature_cols.copy()
            selected_features = []

            for _ in range(max_subset_size):
                best_new_score = float("inf")
                best_feature = None

                # Test each remaining feature
                for feature in remaining_features:
                    candidate_features = selected_features + [feature]
                    X_subset = features_df[candidate_features].values

                    # Create and train model
                    model = model_class(**model_params)
                    model.fit(X_subset, self.y)

                    # Predict and calculate MSE
                    preds = model.predict(X_subset)
                    mse = np.mean((self.y - preds) ** 2)

                    # Update best feature if better
                    if mse < best_new_score:
                        best_new_score = mse
                        best_feature = feature

                if best_feature is None:
                    break

                # Add the best feature
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)

                # Update best overall if better
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_features = selected_features.copy()

                performance_history.append(
                    {
                        "n_features": len(selected_features),
                        "mse": best_new_score,
                        "features": selected_features.copy(),
                    }
                )

                if self.verbose:
                    print(f"  Added feature: {best_feature}, MSE: {best_new_score:.4f}")

        # Backward elimination
        elif method == "remove":
            selected_features = self.feature_cols.copy()

            while len(selected_features) > self.min_features:
                best_new_score = float("inf")
                worst_feature = None

                # Test removing each feature
                for feature in selected_features:
                    candidate_features = [f for f in selected_features if f != feature]
                    X_subset = features_df[candidate_features].values

                    # Create and train model
                    model = model_class(**model_params)
                    model.fit(X_subset, self.y)

                    # Predict and calculate MSE
                    preds = model.predict(X_subset)
                    mse = np.mean((self.y - preds) ** 2)

                    # Update worst feature if better
                    if mse < best_new_score:
                        best_new_score = mse
                        worst_feature = feature

                if worst_feature is None:
                    break

                # Remove the worst feature
                selected_features.remove(worst_feature)

                # Update best overall if better
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_features = selected_features.copy()

                performance_history.append(
                    {
                        "n_features": len(selected_features),
                        "mse": best_new_score,
                        "features": selected_features.copy(),
                    }
                )

                if self.verbose:
                    print(
                        f"  Removed feature: {worst_feature}, MSE: {best_new_score:.4f}"
                    )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Store results
        method_key = f"optimal_{method}_{model_type}"

        self.results[method_key] = {
            "method": f"Optimal Feature Set ({method})",
            "original_features": len(self.feature_cols),
            "selected_features": len(best_features),
            "best_score": best_score,
            "performance_history": performance_history,
            "time_taken": time.time() - start_time,
        }

        self.selected_features[method_key] = best_features

        if self.verbose:
            print(
                f"Selected {len(best_features)} features with best MSE: {best_score:.4f}"
            )
            print(f"Time taken: {self.results[method_key]['time_taken']:.2f} seconds")

        return best_features

    def run_all_methods(self, quick=False):
        """
        Run all feature selection methods and return a combined result.

        Args:
            quick: If True, only run the faster methods

        Returns:
            Dictionary of results for each method
        """
        # Apply variance threshold
        self.filter_by_variance(threshold=0.01)

        # Apply correlation filter
        self.filter_by_correlation(threshold=0.95)

        # Apply mutual information
        self.select_by_mutual_info(k="auto")

        # Apply model-based selection
        self.select_by_model(model_type="rf")
        self.select_by_model(model_type="xgb")

        if not quick:
            # Apply RFE
            self.select_by_rfe(model_type="rf")

            # Apply RFECV (slower)
            self.select_by_rfe(model_type="rf", cv=3)

            # Find optimal feature set (slowest)
            self.find_optimal_feature_set(model_type="rf", method="add")

        # Calculate feature voting
        self.combine_by_voting()

        return self.results

    def combine_by_voting(self, methods=None, min_votes=2):
        """
        Combine multiple feature selection methods by voting.

        Args:
            methods: List of method keys to use (if None, use all available)
            min_votes: Minimum number of votes required for a feature

        Returns:
            List of selected features
        """
        if self.verbose:
            print("Combining feature selection methods by voting...")

        start_time = time.time()

        # Use all available methods if not specified
        if methods is None:
            methods = list(self.selected_features.keys())

        # Count votes for each feature
        votes = {}
        for feature in self.feature_cols:
            votes[feature] = 0
            for method in methods:
                if (
                    method in self.selected_features
                    and feature in self.selected_features[method]
                ):
                    votes[feature] += 1

        # Select features with enough votes
        selected_features = [f for f, v in votes.items() if v >= min_votes]

        # Sort by number of votes
        feature_votes = [(f, v) for f, v in votes.items()]
        feature_votes.sort(key=lambda x: x[1], reverse=True)

        self.results["voting"] = {
            "method": "Voting Combination",
            "original_features": len(self.feature_cols),
            "selected_features": len(selected_features),
            "methods_used": methods,
            "min_votes": min_votes,
            "feature_votes": feature_votes,
            "time_taken": time.time() - start_time,
        }

        self.selected_features["voting"] = selected_features

        if self.verbose:
            print(
                f"Selected {len(selected_features)} features by voting (min votes: {min_votes})"
            )
            print(f"Time taken: {self.results['voting']['time_taken']:.2f} seconds")
            print("Top 10 features by votes:")
            for f, v in feature_votes[:10]:
                print(f"  {f}: {v} votes")

        return selected_features

    def visualize_results(self, save_dir=None):
        """
        Visualize the results of feature selection methods.

        Args:
            save_dir: Directory to save the visualizations (if None, show only)

        Returns:
            Dictionary of figure objects
        """
        figures = {}

        # 1. Bar chart of selected features by method
        methods = []
        selected_counts = []

        for method, result in self.results.items():
            methods.append(result.get("method", method))
            selected_counts.append(result.get("selected_features", 0))

        if methods:
            feature_counts_fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(methods, selected_counts)
            ax.set_ylabel("Number of Selected Features")
            ax.set_title("Number of Features Selected by Each Method")
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            figures["feature_counts"] = feature_counts_fig

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                feature_counts_fig.savefig(os.path.join(save_dir, "feature_counts.png"))

        # 2. Feature importance plot for applicable methods
        for method, result in self.results.items():
            if "feature_importances" in result and result["feature_importances"]:
                # Extract feature names and importances
                features = [f for f, _ in result["feature_importances"][:15]]
                importances = [i for _, i in result["feature_importances"][:15]]

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(features[::-1], importances[::-1])
                ax.set_xlabel("Importance")
                ax.set_title(f'Top 15 Features by {result.get("method", method)}')
                ax.grid(axis="x", linestyle="--", alpha=0.7)
                plt.tight_layout()

                figures[f"{method}_importance"] = fig

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fig.savefig(os.path.join(save_dir, f"{method}_importance.png"))

        # 3. Voting visualization if available
        if "voting" in self.results and "feature_votes" in self.results["voting"]:
            # Extract feature names and votes
            feature_votes = self.results["voting"]["feature_votes"]
            features = [f for f, _ in feature_votes[:20]]
            votes = [v for _, v in feature_votes[:20]]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(features[::-1], votes[::-1])
            ax.set_xlabel("Votes")
            ax.set_title("Top 20 Features by Voting")
            ax.axvline(
                x=self.results["voting"]["min_votes"],
                color="r",
                linestyle="--",
                label=f'Min Votes ({self.results["voting"]["min_votes"]})',
            )
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.legend()
            plt.tight_layout()

            figures["voting"] = fig

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, "voting.png"))

        # 4. Performance history plot for optimal feature set if available
        for method, result in self.results.items():
            if "performance_history" in result and result["performance_history"]:
                # Extract performance data
                history = result["performance_history"]
                n_features = [h["n_features"] for h in history]
                mse = [h["mse"] for h in history]

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(n_features, mse, "o-")
                ax.set_xlabel("Number of Features")
                ax.set_ylabel("MSE")
                ax.set_title(
                    f'Performance vs Number of Features ({result.get("method", method)})'
                )
                ax.grid(linestyle="--", alpha=0.7)

                # Mark the best point
                best_idx = mse.index(min(mse))
                ax.plot(
                    n_features[best_idx],
                    mse[best_idx],
                    "ro",
                    markersize=8,
                    label=f"Best ({n_features[best_idx]} features, MSE={mse[best_idx]:.4f})",
                )
                ax.legend()

                plt.tight_layout()

                figures[f"{method}_performance"] = fig

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fig.savefig(os.path.join(save_dir, f"{method}_performance.png"))

        # Show all figures if not saving
        if not save_dir:
            plt.show()

        return figures

    def get_recommendation(self, method="voting"):
        """
        Get the recommended features based on a specific method.

        Args:
            method: Method to use for recommendation

        Returns:
            List of recommended features
        """
        if method in self.selected_features:
            return self.selected_features[method]
        else:
            # Default to voting if available
            if "voting" in self.selected_features:
                return self.selected_features["voting"]
            # Otherwise use any available method
            elif self.selected_features:
                return list(self.selected_features.values())[0]
            else:
                return self.feature_cols

    def generate_report(self, save_path=None):
        """
        Generate a comprehensive feature selection report.

        Args:
            save_path: Path to save the report (if None, print only)

        Returns:
            Report as a string
        """
        report = []
        report.append("=" * 80)
        report.append("FEATURE SELECTION REPORT")
        report.append("=" * 80)
        report.append("")

        # 1. Summary
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total features: {len(self.feature_cols)}")
        report.append(f"Methods applied: {len(self.results)}")
        if "voting" in self.selected_features:
            report.append(
                f"Recommended features (voting): {len(self.selected_features['voting'])}"
            )
        report.append("")

        # 2. Methods summary
        report.append("METHODS SUMMARY")
        report.append("-" * 80)
        for method, result in self.results.items():
            report.append(
                f"{result.get('method', method)}: "
                f"{result.get('selected_features', 0)} features, "
                f"{result.get('time_taken', 0):.2f} seconds"
            )
        report.append("")

        # 3. Voting results if available
        if "voting" in self.results and "feature_votes" in self.results["voting"]:
            report.append("TOP FEATURES BY VOTING")
            report.append("-" * 80)
            for i, (feature, votes) in enumerate(
                self.results["voting"]["feature_votes"][:20]
            ):
                report.append(f"{i+1:2d}. {feature:30s}: {votes} votes")
            report.append("")

        # 4. Method-specific details
        report.append("METHOD DETAILS")
        report.append("-" * 80)
        for method, result in self.results.items():
            report.append(f"Method: {result.get('method', method)}")
            report.append(f"  Selected features: {result.get('selected_features', 0)}")
            report.append(f"  Time taken: {result.get('time_taken', 0):.2f} seconds")

            # Add method-specific details
            if "feature_importances" in result and result["feature_importances"]:
                report.append("  Top 10 features by importance:")
                for i, (feature, importance) in enumerate(
                    result["feature_importances"][:10]
                ):
                    report.append(f"    {i+1:2d}. {feature:30s}: {importance:.4f}")

            if "feature_rankings" in result and result["feature_rankings"]:
                report.append("  Top 10 features by ranking:")
                for i, (feature, ranking) in enumerate(result["feature_rankings"][:10]):
                    report.append(f"    {i+1:2d}. {feature:30s}: {ranking}")

            if "performance_history" in result and result["performance_history"]:
                report.append("  Performance history:")
                best_mse = float("inf")
                best_n = 0
                for entry in result["performance_history"]:
                    if entry["mse"] < best_mse:
                        best_mse = entry["mse"]
                        best_n = entry["n_features"]
                report.append(
                    f"    Best performance: {best_n} features, MSE={best_mse:.4f}"
                )

            report.append("")

        # 5. Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        if "voting" in self.selected_features:
            report.append(
                f"Recommended features (voting, {len(self.selected_features['voting'])} features):"
            )
            for feature in self.selected_features["voting"]:
                report.append(f"  - {feature}")
        else:
            # Use the first available method
            for method, features in self.selected_features.items():
                report.append(
                    f"Recommended features ({method}, {len(features)} features):"
                )
                for feature in features:
                    report.append(f"  - {feature}")
                break

        report_text = "\n".join(report)

        # Save or print
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)

        if self.verbose:
            print(report_text)

        return report_text


def select_optimal_features(
    df,
    target_col="Close",
    feature_cols=None,
    min_features=10,
    quick=True,
    save_dir=None,
):
    """
    Convenience function to run feature selection and return recommended features.

    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_cols: List of feature columns
        min_features: Minimum number of features to select
        quick: If True, run only fast methods
        save_dir: Directory to save visualizations and report

    Returns:
        List of recommended features
    """
    # Create selector
    selector = FeatureSelector(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_features=min_features,
    )

    # Run methods
    selector.run_all_methods(quick=quick)

    # Visualize
    if save_dir:
        selector.visualize_results(save_dir=save_dir)
        selector.generate_report(
            save_path=os.path.join(save_dir, "feature_selection_report.txt")
        )

    # Return recommended features
    return selector.get_recommendation(method="voting")
