# drift_dashboard.py
"""
Dashboard component for visualizing concept drift detection.
"""

import os
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

def show_drift_visualization():
    """Display drift detection visualization in the dashboard"""
    st.subheader("Concept Drift Detection")
    
    # Check if we have drift visualization data
    if "drift_visualization" not in st.session_state:
        st.info("No drift detection data available yet. Run model training to collect drift data.")
        return
    
    viz_data = st.session_state["drift_visualization"]
    
    # Create dataframe from visualization data
    if viz_data and "timestamps" in viz_data and viz_data["timestamps"]:
        drift_df = pd.DataFrame({
            "timestamp": viz_data["timestamps"],
            "score": viz_data["drift_scores"],
            "type": viz_data["drift_types"]
        })
        
        # Make timestamps readable
        drift_df["time"] = [datetime.fromtimestamp(ts) for ts in drift_df["timestamp"]]
        
        # Show drift score chart
        st.write("### Drift Score Over Time")
        
        chart = alt.Chart(drift_df).mark_line(point=True).encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("score:Q", title="Drift Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("type:N", title="Drift Type"),
            tooltip=["time:T", "score:Q", "type:N"]
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show recent drift events
        st.write("### Recent Drift Events")
        
        if "drift_events" in st.session_state and st.session_state["drift_events"]:
            events = st.session_state["drift_events"]
            
            # Create table of most recent events
            events_df = pd.DataFrame([
                {
                    "Time": e["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Type": e["type"],
                    "Score": f"{e['score']:.4f}",
                    "Adaptation": ", ".join([f"{k}: {v}" for k, v in e["adaptation"].items() 
                                            if k not in ["drift_type", "drift_score", "timestamp"]])
                }
                for e in events[-10:]  # Show last 10 events
            ])
            
            if not events_df.empty:
                st.dataframe(events_df, use_container_width=True)
            else:
                st.info("No drift events recorded yet.")
        else:
            st.info("No drift events recorded yet.")
    else:
        st.info("Insufficient drift detection data available. Continue model training to collect more data.")
    
    # Show drift settings and controls
    with st.expander("Drift Detection Settings"):
        st.write("### Detection Thresholds")
        
        # Allow adjusting sensitivity
        sensitivity = st.slider(
            "Drift Detection Sensitivity", 
            min_value=0.1, 
            max_value=2.0, 
            value=st.session_state.get("drift_sensitivity", 1.0),
            step=0.1
        )
        
        # Store sensitivity setting
        if "drift_sensitivity" not in st.session_state or st.session_state["drift_sensitivity"] != sensitivity:
            st.session_state["drift_sensitivity"] = sensitivity
            
            # Update detector if it exists
            if "drift_detector" in st.session_state:
                detector = st.session_state["drift_detector"]
                
                # Update thresholds based on sensitivity
                detector.thresholds["statistical"] = 2.0 / sensitivity
                detector.thresholds["performance"] = 0.15 * sensitivity
                detector.thresholds["distribution"] = 0.05 * sensitivity
                
                st.success("Updated drift detection sensitivity")

        # Add reset button
        if st.button("Reset Drift Detection"):
            if "drift_detector" in st.session_state:
                del st.session_state["drift_detector"]
            if "drift_visualization" in st.session_state:
                del st.session_state["drift_visualization"]
            if "drift_events" in st.session_state:
                del st.session_state["drift_events"]
            st.success("Drift detection has been reset")
            st.experimental_rerun()