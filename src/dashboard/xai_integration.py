# xai_integration.py
"""
Integration module for Explainable AI (XAI) functionality.
Provides streamlined access to the XAIExplainer from explainable_ai.py.
"""
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Try importing the XAIExplainer
try:
    from src.dashboard.explainable_ai import (
        XAIExplainer,
        explain_model_prediction,
        get_feature_importance,
    )

    HAS_XAI = True
except ImportError:
    logger.warning(
        "Could not import XAIExplainer from Scripts.explainable_ai. XAI functionality will be limited."
    )
    HAS_XAI = False


class XAIWrapper:
    """
    Wrapper class for the XAIExplainer to provide consistent API
    and fallbacks when the original explainer is not available.
    """

    def __init__(
        self,
        model=None,
        model_type: str = "tensorflow",
        feature_names: List[str] = None,
    ):
        """
        Initialize the XAI wrapper with a model.

        Args:
            model: The model to explain
            model_type: Model type ('tensorflow', 'sklearn', or 'xgboost')
            feature_names: Names of features used by the model
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None

        # Try to initialize the explainer if we have a model and XAI is available
        if HAS_XAI and model is not None:
            try:
                self.explainer = XAIExplainer(model, model_type, feature_names)
                logger.info(
                    f"Successfully initialized XAIExplainer for {model_type} model"
                )
            except Exception as e:
                logger.error(f"Failed to initialize XAIExplainer: {str(e)}")

    def has_explainer(self) -> bool:
        """
        Check if an explainer is available.

        Returns:
            bool: True if an explainer is available, False otherwise
        """
        return HAS_XAI and self.explainer is not None

    def explain_prediction(self, X: np.ndarray, method: str = "shap") -> Dict:
        """
        Explain a prediction using the specified method.

        Args:
            X: Input data to explain
            method: Explanation method (shap, lime, integrated_gradients, etc.)

        Returns:
            Dict: Explanation results
        """
        if not self.has_explainer():
            return {"error": "XAI explainer not available"}

        try:
            if method == "shap":
                return self.explainer.explain_with_shap(X, plot=True)
            elif method == "lime":
                return self.explainer.explain_with_lime(X, plot=True)
            elif method == "integrated_gradients":
                return self.explainer.explain_with_integrated_gradients(X, plot=True)
            elif method == "permutation":
                # Need target values for permutation
                y_pred = self.explainer.predict_wrapper(X).flatten()
                return self.explainer.explain_with_permutation_importance(
                    X, y_pred, plot=True
                )
            else:
                return {"error": f"Unsupported explanation method: {method}"}
        except Exception as e:
            logger.error(f"Error explaining prediction with {method}: {str(e)}")
            return {"error": str(e)}

    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance from the model.

        Args:
            X: Input data

        Returns:
            Dict[str, float]: Mapping of feature names to importance values
        """
        if not self.has_explainer():
            # Return mock data if explainer not available
            if self.feature_names:
                import random

                return {name: random.random() for name in self.feature_names}
            return {"error": "XAI explainer not available"}

        try:
            return get_feature_importance(
                self.model, X, self.feature_names, self.model_type
            )
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {"error": str(e)}

    def create_pdp_plot(self, X: np.ndarray, feature_idx: int) -> Dict:
        """
        Create a partial dependence plot for a feature.

        Args:
            X: Input data
            feature_idx: Index of the feature to analyze

        Returns:
            Dict: PDP plot data
        """
        if not self.has_explainer():
            return {"error": "XAI explainer not available"}

        try:
            return self.explainer.create_partial_dependence_plot(
                X, feature_idx, ice=True
            )
        except Exception as e:
            logger.error(f"Error creating PDP plot: {str(e)}")
            return {"error": str(e)}

    def create_counterfactual(self, X: np.ndarray, target_value: float) -> Dict:
        """
        Create a counterfactual example.

        Args:
            X: Input data (single instance)
            target_value: Target prediction value

        Returns:
            Dict: Counterfactual data
        """
        if not self.has_explainer():
            return {"error": "XAI explainer not available"}

        try:
            return self.explainer.create_counterfactual_example(X, target_value)
        except Exception as e:
            logger.error(f"Error creating counterfactual: {str(e)}")
            return {"error": str(e)}


# ===== STREAMLIT INTEGRATION =====


def create_xai_explorer(st, model, X_sample, feature_names=None, key_prefix="xai"):
    """
    Create an interactive XAI explorer for Streamlit dashboards.

    Args:
        st: Streamlit module
        model: Model to explain
        X_sample: Sample data for explanations
        feature_names: Names of the features
        key_prefix: Prefix for Streamlit widget keys

    Returns:
        XAI explorer components
    """
    # Create wrapper
    wrapper = XAIWrapper(model, "tensorflow", feature_names)

    # Create expander for XAI
    with st.expander("🔍 Explainable AI Analysis", expanded=False):
        if not wrapper.has_explainer():
            st.warning(
                "XAI functionality is not available. Make sure explainable_ai.py is in your path."
            )
            return None

        # Create tabs for different explanations
        xai_tabs = st.tabs(["SHAP", "Feature Importance", "PDP", "What-If Analysis"])

        # SHAP Tab
        with xai_tabs[0]:
            st.subheader("SHAP Explanation")

            if st.button("Generate SHAP Analysis", key=f"{key_prefix}_shap_btn"):
                with st.spinner("Generating SHAP explanation..."):
                    shap_results = wrapper.explain_prediction(X_sample, method="shap")

                    if "error" in shap_results:
                        st.error(
                            f"Error generating SHAP explanation: {shap_results['error']}"
                        )
                    elif "figure" in shap_results:
                        st.pyplot(shap_results["figure"])
                        plt.close(shap_results["figure"])
                    else:
                        st.info(
                            "SHAP analysis completed, but no visualization available"
                        )

            st.markdown(
                """
            **What is SHAP?**
            
            SHAP (SHapley Additive exPlanations) shows how each feature contributes to pushing
            the prediction higher or lower from the baseline.
            """
            )

        # Feature Importance Tab
        with xai_tabs[1]:
            st.subheader("Feature Importance")

            if st.button(
                "Generate Feature Importance", key=f"{key_prefix}_importance_btn"
            ):
                with st.spinner("Calculating feature importance..."):
                    importance = wrapper.get_feature_importance(X_sample)

                    if "error" in importance:
                        st.error(
                            f"Error calculating feature importance: {importance['error']}"
                        )
                    else:
                        # Sort by importance
                        sorted_importance = {
                            k: v
                            for k, v in sorted(
                                importance.items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                        }

                        # Display as bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(
                            list(sorted_importance.keys()),
                            list(sorted_importance.values()),
                        )
                        ax.set_xlabel("Importance")
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)
                        plt.close(fig)

        # PDP Tab
        with xai_tabs[2]:
            st.subheader("Partial Dependence Plots")

            # Select feature for PDP
            if feature_names and len(feature_names) > 0:
                selected_feature = st.selectbox(
                    "Select feature for PDP",
                    options=feature_names,
                    key=f"{key_prefix}_pdp_feature",
                )
                feature_idx = feature_names.index(selected_feature)

                if st.button("Generate PDP", key=f"{key_prefix}_pdp_btn"):
                    with st.spinner(f"Generating PDP for {selected_feature}..."):
                        pdp_result = wrapper.create_pdp_plot(X_sample, feature_idx)

                        if "error" in pdp_result:
                            st.error(f"Error generating PDP: {pdp_result['error']}")
                        elif "figure" in pdp_result:
                            st.pyplot(pdp_result["figure"])
                            plt.close(pdp_result["figure"])
                        else:
                            st.info(
                                "PDP analysis completed, but no visualization available"
                            )
            else:
                st.warning("Feature names not available for PDP")

            st.markdown(
                """
            **What are Partial Dependence Plots?**
            
            Partial Dependence Plots show how the prediction changes when a single feature varies
            while all other features remain constant.
            """
            )

        # What-If Tab
        with xai_tabs[3]:
            st.subheader("Counterfactual Analysis")

            # Get current prediction
            if X_sample is not None and X_sample.shape[0] > 0:
                try:
                    current_pred = model.predict(X_sample[:1])[0][0]
                    st.write(f"Current prediction: {current_pred:.4f}")

                    # Set target prediction
                    target_pred = st.slider(
                        "Target prediction",
                        min_value=float(current_pred * 0.5),
                        max_value=float(current_pred * 1.5),
                        value=float(current_pred * 1.1),
                        step=0.01,
                        key=f"{key_prefix}_cf_target",
                    )

                    if st.button("Generate Counterfactual", key=f"{key_prefix}_cf_btn"):
                        with st.spinner("Generating counterfactual example..."):
                            cf_result = wrapper.create_counterfactual(
                                X_sample[:1], target_pred
                            )

                            if "error" in cf_result:
                                st.error(
                                    f"Error generating counterfactual: {cf_result['error']}"
                                )
                            elif "figure" in cf_result:
                                st.pyplot(cf_result["figure"])
                                plt.close(cf_result["figure"])

                                if "counterfactual_data" in cf_result:
                                    data = cf_result["counterfactual_data"]
                                    st.write(
                                        f"Original prediction: {data['original_prediction']:.4f}"
                                    )
                                    st.write(
                                        f"Counterfactual prediction: {data['counterfactual_prediction']:.4f}"
                                    )
                            else:
                                st.info(
                                    "Counterfactual analysis completed, but no visualization available"
                                )
                except Exception as e:
                    st.error(f"Error in counterfactual analysis: {str(e)}")
            else:
                st.warning("Sample data not available for counterfactual analysis")

            st.markdown(
                """
            **What is Counterfactual Analysis?**
            
            Counterfactual analysis shows what changes would be needed in the input features
            to achieve a different prediction.
            """
            )

    return wrapper


# ===== MAIN FUNCTION =====


def main():
    """Main function for demonstration"""
    import numpy as np
    from tensorflow.keras.layers import Dense  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore

    # Create a simple model
    model = Sequential(
        [
            Dense(10, activation="relu", input_shape=(4,)),
            Dense(5, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    # Create sample data
    X = np.random.rand(100, 4)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(100) * 0.1

    # Train the model
    model.fit(X, y, epochs=50, verbose=0)

    # Create sample for explanation
    X_sample = X[:10]

    # Define feature names
    feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]

    # Create XAI wrapper
    wrapper = XAIWrapper(model, "tensorflow", feature_names)

    # Check if explainer is available
    if wrapper.has_explainer():
        print("XAI explainer is available")

        # Generate explanations
        shap_result = wrapper.explain_prediction(X_sample, method="shap")
        importance = wrapper.get_feature_importance(X_sample)

        # Display results
        print("\nFeature Importance:")
        for feature, value in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feature}: {value:.4f}")
    else:
        print("XAI explainer is not available")


if __name__ == "__main__":
    main()
