"""
Explainable AI tab for the dashboard.
This module provides the UI components for the XAI tab in the Streamlit dashboard.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)

# Try to import XAI components
try:
    from src.dashboard.explainable_ai import XAIExplainer, explain_model_prediction, get_feature_importance
    from src.dashboard.xai_integration import XAIWrapper, create_xai_explorer
    HAS_XAI = True
except ImportError as e:
    logger.error(f"Error importing XAI components: {e}")
    HAS_XAI = False

def render_explainable_ai_tab():
    """
    Render the Explainable AI tab in the dashboard.
    This function creates a user interface for exploring model explanations.
    """
    st.title("🔍 Explainable AI")
    
    if not HAS_XAI:
        st.error("XAI components could not be loaded. Please check your installation.")
        st.info("Make sure both explainable_ai.py and xai_integration.py are properly installed.")
        return
        
    # Section for model selection
    st.subheader("Model Selection")
    
    # Check if we have a trained model in the session state
    if "current_model" not in st.session_state:
        st.warning("No trained model available. Please train a model first.")
        return
    
    model = st.session_state["current_model"]
    
    # Get sample data for explanations
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        
        # Get feature columns
        feature_cols = []
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']:
                feature_cols.append(col)
        
        if not feature_cols:
            # If no feature columns found, use basic features
            feature_cols = ['Open', 'High', 'Low', 'Volume']
            
        # Create sample data for explanation
        X_sample = df[feature_cols].tail(30).values
        
        # Create tabs for different explanations
        xai_tabs = st.tabs(["Feature Importance", "SHAP Analysis", "What-If Analysis", "PDP Plots"])
        
        # Feature Importance tab
        with xai_tabs[0]:
            st.subheader("Feature Importance")
            
            if st.button("Generate Feature Importance"):
                with st.spinner("Calculating feature importance..."):
                    try:
                        # Use XAIWrapper to get feature importance
                        wrapper = XAIWrapper(model, "tensorflow", feature_cols)
                        importance = wrapper.get_feature_importance(X_sample)
                        
                        if "error" in importance:
                            st.error(f"Error calculating feature importance: {importance['error']}")
                        else:
                            # Sort by importance
                            sorted_importance = {k: v for k, v in sorted(
                                importance.items(), key=lambda item: item[1], reverse=True)}
                            
                            # Display as bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
                            ax.set_xlabel("Importance")
                            ax.set_title("Feature Importance")
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Display table of values
                            importance_df = pd.DataFrame({
                                'Feature': list(sorted_importance.keys()),
                                'Importance': list(sorted_importance.values())
                            })
                            st.dataframe(importance_df)
                    except Exception as e:
                        st.error(f"Error generating feature importance: {e}")
            
            st.markdown("""
            **What is Feature Importance?**
            
            Feature importance identifies which features (input variables) most influence the model's predictions.
            Higher values indicate a stronger influence on the model output.
            """)
        
        # SHAP Analysis tab
        with xai_tabs[1]:
            st.subheader("SHAP Analysis")
            
            if st.button("Generate SHAP Explanation"):
                with st.spinner("Generating SHAP values..."):
                    try:
                        # Use XAIWrapper to explain with SHAP
                        wrapper = XAIWrapper(model, "tensorflow", feature_cols)
                        shap_results = wrapper.explain_prediction(X_sample, method="shap")
                        
                        if "error" in shap_results:
                            st.error(f"Error generating SHAP explanation: {shap_results['error']}")
                        elif "figure" in shap_results:
                            st.pyplot(shap_results["figure"])
                            plt.close(shap_results["figure"])
                        else:
                            st.info("SHAP analysis completed, but no visualization available")
                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {e}")
            
            st.markdown("""
            **What is SHAP?**
            
            SHAP (SHapley Additive exPlanations) shows how each feature contributes to pushing
            the prediction higher or lower from the baseline. It helps explain individual predictions
            by computing the contribution of each feature to the prediction.
            """)
        
        # What-If Analysis tab
        with xai_tabs[2]:
            st.subheader("What-If Analysis (Counterfactual Explanations)")
            
            if st.button("Generate Counterfactual Example"):
                with st.spinner("Generating counterfactual example..."):
                    try:
                        # Use XAIWrapper to create counterfactual
                        wrapper = XAIWrapper(model, "tensorflow", feature_cols)
                        
                        # Get current prediction
                        current_pred = model.predict(X_sample[:1])[0][0]
                        st.write(f"Current prediction: {current_pred:.4f}")
                        
                        # Set target prediction (10% higher)
                        target_pred = current_pred * 1.1
                        
                        cf_result = wrapper.create_counterfactual(X_sample[:1], target_pred)
                        
                        if "error" in cf_result:
                            st.error(f"Error generating counterfactual: {cf_result['error']}")
                        elif "figure" in cf_result:
                            st.pyplot(cf_result["figure"])
                            plt.close(cf_result["figure"])
                            
                            if "counterfactual_data" in cf_result:
                                data = cf_result["counterfactual_data"]
                                st.write(f"Original prediction: {data['original_prediction']:.4f}")
                                st.write(f"Counterfactual prediction: {data['counterfactual_prediction']:.4f}")
                                
                                # Show feature changes
                                changes_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Original': data['original'],
                                    'Counterfactual': data['counterfactual'],
                                    'Change': data['changes']
                                })
                                st.dataframe(changes_df.sort_values('Change', key=abs, ascending=False))
                        else:
                            st.info("Counterfactual analysis completed, but no visualization available")
                    except Exception as e:
                        st.error(f"Error in counterfactual analysis: {str(e)}")
            
            st.markdown("""
            **What is Counterfactual Analysis?**
            
            Counterfactual analysis shows what changes would be needed in the input features
            to achieve a different prediction. It answers questions like "What would I need to change
            to get a 10% higher prediction?"
            """)
        
        # PDP Plots tab
        with xai_tabs[3]:
            st.subheader("Partial Dependence Plots")
            
            # Select feature for PDP
            if feature_cols and len(feature_cols) > 0:
                selected_feature = st.selectbox(
                    "Select feature for PDP", 
                    options=feature_cols
                )
                feature_idx = feature_cols.index(selected_feature)
                
                if st.button("Generate PDP"):
                    with st.spinner(f"Generating PDP for {selected_feature}..."):
                        try:
                            # Use XAIWrapper to create PDP
                            wrapper = XAIWrapper(model, "tensorflow", feature_cols)
                            pdp_result = wrapper.create_pdp_plot(X_sample, feature_idx)
                            
                            if "error" in pdp_result:
                                st.error(f"Error generating PDP: {pdp_result['error']}")
                            elif "figure" in pdp_result:
                                st.pyplot(pdp_result["figure"])
                                plt.close(pdp_result["figure"])
                            else:
                                st.info("PDP analysis completed, but no visualization available")
                        except Exception as e:
                            st.error(f"Error generating PDP: {str(e)}")
            else:
                st.warning("Feature names not available for PDP")
            
            st.markdown("""
            **What are Partial Dependence Plots?**
            
            Partial Dependence Plots show how the prediction changes when a single feature varies
            while all other features remain constant. They help understand the relationship between
            a feature and the target variable, accounting for the average effects of all other features.
            """)
            
    else:
        st.warning("No data available. Please load data first.")

    # Add a comprehensive XAI explorer as an alternative interface
    st.subheader("Interactive XAI Explorer")
    
    if "current_model" in st.session_state and "df" in st.session_state and st.session_state["df"] is not None:
        model = st.session_state["current_model"]
        df = st.session_state["df"]
        
        # Get feature columns
        feature_cols = []
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']:
                feature_cols.append(col)
        
        if not feature_cols:
            # If no feature columns found, use basic features
            feature_cols = ['Open', 'High', 'Low', 'Volume']
            
        # Create sample data for explanation
        X_sample = df[feature_cols].tail(30).values
        
        # Create XAI explorer
        create_xai_explorer(st, model, X_sample, feature_cols, key_prefix="explorer")
    else:
        st.info("Load data and train a model to use the interactive XAI explorer.")

if __name__ == "__main__":
    # For testing the module independently
    import streamlit as st
    st.set_page_config(page_title="XAI Test", layout="wide")
    render_explainable_ai_tab()
