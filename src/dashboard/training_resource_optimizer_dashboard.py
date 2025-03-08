# training_resource_optimizer_dashboard.py
"""
Dashboard tab for monitoring model training performance and resource allocation.
Simply import and call the function from your main dashboard.
"""

import os
import time
from typing import Dict, Any, List, Optional
import threading

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.utils.training_optimizer import get_training_optimizer, TrainingOptimizer

def render_training_optimizer_tab():
    """
    Renders the complete training optimizer dashboard tab.
    Call this function from your main dashboard app.
    """
    st.title("Training Performance & Resource Allocation")
    
    # Get or create training optimizer
    if "training_optimizer" not in st.session_state:
        st.session_state["training_optimizer"] = get_training_optimizer()
    
    optimizer = st.session_state["training_optimizer"]
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Hardware Resources", 
        "Model Performance", 
        "Resource Allocation",
        "Advanced Settings"
    ])
    
    # Tab 1: Hardware Resource Detection & Usage
    with tab1:
        render_hardware_resources_section(optimizer)
    
    # Tab 2: Model Training Performance
    with tab2:
        render_model_performance_section()
    
    # Tab 3: Resource Allocation
    with tab3:
        render_resource_allocation_section(optimizer)
    
    # Tab 4: Advanced Settings
    with tab4:
        render_advanced_settings_section(optimizer)


def render_hardware_resources_section(optimizer: TrainingOptimizer):
    """Render hardware resources section with detected and used resources."""
    st.header("Hardware Resources")
    st.write("The system has automatically detected the following hardware resources:")
    
    # Hardware specs
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Cores", f"{optimizer.cpu_count}")
        st.metric("System Memory", f"{optimizer.system_memory_gb:.1f} GB")
        
        # Show CPU utilization if available
        try:
            import psutil
            process = psutil.Process()
            with st.expander("CPU Details", expanded=False):
                # Refresh button for real-time updates
                if st.button("Refresh CPU Info", key="refresh_cpu"):
                    st.session_state["last_cpu_refresh"] = datetime.now()
                
                # CPU usage by core
                cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
                
                # Create a DataFrame for the chart
                cpu_df = pd.DataFrame({
                    "Core": [f"Core {i}" for i in range(len(cpu_percent))],
                    "Usage": cpu_percent
                })
                
                # Create chart
                cpu_chart = alt.Chart(cpu_df).mark_bar().encode(
                    x=alt.X('Usage:Q', title='Usage %', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Core:N', title=None),
                    color=alt.Color('Usage:Q', scale=alt.Scale(scheme='blues'))
                )
                
                st.altair_chart(cpu_chart, use_container_width=True)
                
                # Process info
                st.write("##### Process Information")
                st.write(f"Process CPU: {process.cpu_percent()}%")
                st.write(f"Thread Count: {process.num_threads()}")
        except Exception as e:
            st.write(f"CPU utilization details unavailable: {e}")
    
    with col2:
        st.metric("GPUs", f"{optimizer.gpu_count}")
        if optimizer.has_gpu:
            st.metric("GPU Memory", f"{optimizer.gpu_memory_gb:.1f} GB")
            
            # Show GPU utilization if available
            with st.expander("GPU Details", expanded=False):
                # Refresh button for real-time updates
                if st.button("Refresh GPU Info", key="refresh_gpu"):
                    st.session_state["last_gpu_refresh"] = datetime.now()
                
                try:
                    import subprocess
                    # Try nvidia-smi for NVIDIA GPUs
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total', 
                             '--format=csv,noheader'],
                            stdout=subprocess.PIPE, check=True, text=True
                        )
                        
                        # Parse the output
                        gpu_info = []
                        for line in result.stdout.strip().split('\n'):
                            parts = [part.strip() for part in line.split(',')]
                            if len(parts) >= 6:
                                gpu_info.append({
                                    "index": parts[0],
                                    "name": parts[1],
                                    "gpu_util": parts[2],
                                    "mem_util": parts[3],
                                    "mem_used": parts[4],
                                    "mem_total": parts[5]
                                })
                        
                        # Display GPU info
                        for gpu in gpu_info:
                            st.write(f"##### {gpu['name']} (GPU {gpu['index']})")
                            
                            # Parse memory values
                            mem_used = float(gpu['mem_used'].split()[0])
                            mem_total = float(gpu['mem_total'].split()[0])
                            mem_pct = mem_used / mem_total * 100
                            
                            # Usage bars
                            st.write(f"Compute: {gpu['gpu_util']}")
                            st.progress(float(gpu['gpu_util'].rstrip('%')) / 100)
                            
                            st.write(f"Memory: {gpu['mem_util']} ({mem_used} / {mem_total} MiB)")
                            st.progress(mem_pct / 100)
                    
                    except Exception:
                        # If nvidia-smi fails, try TensorFlow API
                        import tensorflow as tf
                        gpus = tf.config.list_physical_devices('GPU')
                        for i, gpu in enumerate(gpus):
                            st.write(f"##### GPU {i}: {gpu.name}")
                            try:
                                mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                                if 'current' in mem_info and 'peak' in mem_info:
                                    current_mb = mem_info['current'] / (1024 * 1024)
                                    peak_mb = mem_info['peak'] / (1024 * 1024)
                                    st.write(f"Current Memory: {current_mb:.1f} MB")
                                    st.write(f"Peak Memory: {peak_mb:.1f} MB")
                            except:
                                st.write("Memory information not available")
                
                except Exception as e:
                    st.write(f"GPU details unavailable: {e}")
        else:
            st.warning("No GPUs detected. Models will run on CPU only.")
    
    # Resource allocation summary
    st.subheader("Resource Allocation for Models")
    
    # Create a table showing resource allocation for each model type
    model_resources = []
    for model_type, resources in optimizer.config["model_resource_profiles"].items():
        model_resources.append({
            "Model Type": model_type.upper(),
            "GPU Usage": f"{resources['gpu_memory_fraction']*100:.1f}%" if resources['gpu_memory_fraction'] > 0 else "None",
            "CPU Weight": f"{resources['cpu_weight']:.1f}x",
            "RAM": f"{resources['ram_gb']:.1f} GB",
            "TensorFlow": "Yes" if resources['tf_model'] else "No"
        })
    
    # Display as DataFrame
    model_df = pd.DataFrame(model_resources)
    st.dataframe(model_df, use_container_width=True)
    
    # Show the current configuration
    with st.expander("Current Configuration", expanded=False):
        st.write("Current hardware configuration settings:")
        
        # Pretty-print the config
        for key, value in optimizer.config.items():
            if key != "model_resource_profiles":  # Already showing this above
                st.write(f"**{key}:** {value}")


def render_model_performance_section():
    """Render model performance comparison section with training metrics."""
    st.header("Model Training Performance")
    
    # Initialize model performance tracking if not present
    if "model_training_metrics" not in st.session_state:
        st.session_state["model_training_metrics"] = {}
    
    if "model_training_times" not in st.session_state:
        st.session_state["model_training_times"] = {}
    
    metrics = st.session_state["model_training_metrics"]
    times = st.session_state["model_training_times"]
    
    if not metrics or not times:
        st.info("No training data available yet. Train models to see performance metrics.")
        
        # Sample data for visualization
        st.write("#### Sample Visualization (with dummy data)")
        
        # Create sample data
        sample_times = {
            "lstm": 45.2,
            "rnn": 38.7,
            "xgboost": 12.3,
            "random_forest": 8.5,
            "tabnet": 67.8,
            "cnn": 52.1,
            "ltc": 41.3,
            "tft": 73.9
        }
        
        # Visualize sample training times
        time_df = pd.DataFrame([
            {"model": model, "seconds": time} 
            for model, time in sample_times.items()
        ])
        
        # Sort by time
        time_df = time_df.sort_values("seconds", ascending=False)
        
        # Create chart
        chart = alt.Chart(time_df).mark_bar().encode(
            x=alt.X('seconds:Q', title='Training Time (seconds)'),
            y=alt.Y('model:N', title='Model Type', sort='-x'),
            color=alt.Color('model:N', legend=None)
        ).properties(
            title='Example: Training Time by Model Type'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Sample metrics
        sample_metrics = {
            "lstm": {"rmse": 0.0412, "mape": 3.45},
            "rnn": {"rmse": 0.0437, "mape": 3.78},
            "xgboost": {"rmse": 0.0385, "mape": 3.12},
            "random_forest": {"rmse": 0.0401, "mape": 3.34},
            "tabnet": {"rmse": 0.0390, "mape": 3.21},
            "cnn": {"rmse": 0.0424, "mape": 3.56},
            "ltc": {"rmse": 0.0429, "mape": 3.62},
            "tft": {"rmse": 0.0378, "mape": 3.05}
        }
        
        # Create metric visualization
        metric_data = []
        for model, model_metrics in sample_metrics.items():
            for metric_name, value in model_metrics.items():
                metric_data.append({
                    "model": model,
                    "metric": metric_name,
                    "value": value
                })
        
        metric_df = pd.DataFrame(metric_data)
        
        # Create faceted chart for metrics
        metric_chart = alt.Chart(metric_df).mark_bar().encode(
            x=alt.X('value:Q', title='Value'),
            y=alt.Y('model:N', title='Model', sort='x'),
            color='model:N',
            column='metric:N'
        ).properties(
            width=300,
            title='Example: Model Performance Metrics'
        )
        
        st.altair_chart(metric_chart)
        
        return
    
    # Real data visualization
    st.subheader("Training Times")
    
    # Create DataFrame for training times
    time_df = pd.DataFrame([
        {"model": model, "seconds": time} 
        for model, time in times.items()
    ])
    
    # Sort by time
    time_df = time_df.sort_values("seconds", ascending=False)
    
    # Create chart
    time_chart = alt.Chart(time_df).mark_bar().encode(
        x=alt.X('seconds:Q', title='Training Time (seconds)'),
        y=alt.Y('model:N', title='Model Type', sort='-x'),
        color=alt.Color('model:N', legend=None)
    ).properties(
        title='Training Time by Model Type'
    )
    
    st.altair_chart(time_chart, use_container_width=True)
    
    # Check for imbalances
    if len(time_df) > 1:
        max_time = time_df['seconds'].max()
        min_time = time_df['seconds'].min()
        
        if max_time > min_time * 3:
            slowest = time_df.iloc[0]['model']
            fastest = time_df.iloc[-1]['model']
            
            st.warning(f"⚠️ **Training Imbalance Detected**: {slowest} is {max_time/min_time:.1f}x slower than {fastest}")
            
            # Resource allocation suggestion
            st.info("""
            **Suggestion**: Consider adjusting resource allocation in Advanced Settings 
            to give more resources to slower models.
            """)
    
    # Performance metrics
    st.subheader("Model Performance Metrics")
    
    # Create metric visualization if any metrics available
    if any(metrics.values()):
        # Gather all metric types
        all_metrics = set()
        for model_metrics in metrics.values():
            all_metrics.update(model_metrics.keys())
        
        # Create metric data
        metric_data = []
        for model, model_metrics in metrics.items():
            for metric_name in all_metrics:
                if metric_name in model_metrics:
                    metric_data.append({
                        "model": model,
                        "metric": metric_name,
                        "value": model_metrics[metric_name]
                    })
        
        if metric_data:
            metric_df = pd.DataFrame(metric_data)
            
            # Create tabs for different metrics
            metric_tabs = st.tabs(list(all_metrics))
            
            for i, metric_name in enumerate(all_metrics):
                with metric_tabs[i]:
                    # Filter data for this metric
                    this_metric = metric_df[metric_df['metric'] == metric_name]
                    
                    if not this_metric.empty:
                        # Sort appropriately (lower is better for most metrics)
                        this_metric = this_metric.sort_values('value', ascending=True)
                        
                        # Create chart
                        chart = alt.Chart(this_metric).mark_bar().encode(
                            x=alt.X('value:Q', title=f'{metric_name.upper()}'),
                            y=alt.Y('model:N', title='Model Type', sort='x'),
                            color=alt.Color('model:N', legend=None)
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Show best model
                        best_model = this_metric.iloc[0]['model']
                        best_value = this_metric.iloc[0]['value']
                        
                        st.success(f"Best model for {metric_name.upper()}: **{best_model}** ({best_value:.6f})")
    else:
        st.info("No performance metrics available yet.")


def render_resource_allocation_section(optimizer: TrainingOptimizer):
    """Render resource allocation controls for all model types."""
    st.header("Resource Allocation")
    st.write("""
    Fine-tune resource allocation for each model type. These settings will be applied
    the next time you train models.
    """)
    
    # Ensure config has model_resource_profiles
    if "model_resource_profiles" not in optimizer.config:
        st.error("Resource profiles not available in optimizer configuration.")
        return
    
    # Store current profiles
    current_profiles = optimizer.config["model_resource_profiles"]
    
    # Deep copy for modification
    if "edited_resource_profiles" not in st.session_state:
        st.session_state["edited_resource_profiles"] = {
            model: resources.copy() 
            for model, resources in current_profiles.items()
        }
    
    edited_profiles = st.session_state["edited_resource_profiles"]
    
    # Create tabs for each model type
    model_tabs = st.tabs(list(edited_profiles.keys()))
    
    for i, model_type in enumerate(edited_profiles.keys()):
        with model_tabs[i]:
            st.subheader(f"{model_type.upper()} Resource Settings")
            
            # Get current resources for this model
            resources = edited_profiles[model_type]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # GPU allocation (if GPU available)
                if optimizer.has_gpu:
                    resources["gpu_memory_fraction"] = st.slider(
                        "GPU Memory Allocation",
                        min_value=0.0,
                        max_value=0.8,
                        value=float(resources["gpu_memory_fraction"]),
                        step=0.05,
                        help="Fraction of GPU memory to allocate (0.0 = CPU only)",
                        key=f"{model_type}_gpu"
                    )
                
                # CPU allocation
                resources["cpu_weight"] = st.slider(
                    "CPU Weight",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(resources["cpu_weight"]),
                    step=0.5,
                    help="Relative CPU allocation (higher = more CPU cores)",
                    key=f"{model_type}_cpu"
                )
            
            with col2:
                # RAM allocation
                resources["ram_gb"] = st.slider(
                    "RAM Allocation (GB)",
                    min_value=0.5,
                    max_value=min(8.0, optimizer.system_memory_gb * 0.5),
                    value=float(resources["ram_gb"]),
                    step=0.5,
                    help="RAM allocation in GB",
                    key=f"{model_type}_ram"
                )
                
                # Batch size factor
                resources["batch_size_factor"] = st.slider(
                    "Batch Size Factor",
                    min_value=0.2,
                    max_value=2.0,
                    value=float(resources["batch_size_factor"]),
                    step=0.1,
                    help="Relative batch size (higher = larger batches)",
                    key=f"{model_type}_batch"
                )
            
            # Additional settings for neural network models
            if resources.get("tf_model", False):
                st.write("##### Neural Network Specific Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Mixed precision
                    use_mixed = st.checkbox(
                        "Use Mixed Precision",
                        value=optimizer.config.get("use_mixed_precision", True) and optimizer.has_gpu,
                        disabled=not optimizer.has_gpu,
                        help="Use mixed precision for faster training (requires GPU)",
                        key=f"{model_type}_mixed"
                    )
                    
                with col2:
                    # XLA compilation
                    use_xla = st.checkbox(
                        "Use XLA Compilation",
                        value=optimizer.config.get("use_xla", True) and optimizer.has_gpu,
                        disabled=not optimizer.has_gpu,
                        help="Use XLA compilation for faster training (requires GPU)",
                        key=f"{model_type}_xla"
                    )
    
    # Apply button for resource changes
    if st.button("Apply Resource Allocation Changes"):
        # Update optimizer config with edited profiles
        optimizer.config["model_resource_profiles"] = st.session_state["edited_resource_profiles"]
        
        # Update mixed precision and XLA settings based on first neural network model
        for model_type, resources in edited_profiles.items():
            if resources.get("tf_model", False):
                # Use settings from this model for global config
                optimizer.config["use_mixed_precision"] = st.session_state.get(f"{model_type}_mixed", False)
                optimizer.config["use_xla"] = st.session_state.get(f"{model_type}_xla", False)
                break
        
        # Reconfigure TensorFlow based on new settings
        optimizer._configure_tensorflow()
        
        st.success("Resource allocation updated successfully!")
        
        # Reset training metrics since they're no longer valid with new allocations
        if "model_training_metrics" in st.session_state:
            st.session_state["model_training_metrics"] = {}
        
        if "model_training_times" in st.session_state:
            st.session_state["model_training_times"] = {}


def render_advanced_settings_section(optimizer: TrainingOptimizer):
    """Render advanced settings for the training optimizer."""
    st.header("Advanced Settings")
    
    st.write("""
    These settings control how the training optimizer allocates resources and 
    manages parallel execution. Adjust with caution as they can significantly impact
    training performance.
    """)
    
    # Create copy of config for editing
    if "edited_advanced_config" not in st.session_state:
        st.session_state["edited_advanced_config"] = {
            k: v for k, v in optimizer.config.items() 
            if k not in ["model_resource_profiles"]
        }
    
    edited_config = st.session_state["edited_advanced_config"]
    
    # Parallel execution settings
    st.subheader("Parallel Execution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edited_config["max_parallel_models"] = st.slider(
            "Maximum Parallel Models",
            min_value=1,
            max_value=8,
            value=int(edited_config.get("max_parallel_models", min(8, max(1, optimizer.cpu_count // 2)))),
            help="Maximum number of models to train in parallel",
            key="max_parallel"
        )
        
        edited_config["max_gpu_models"] = st.slider(
            "Maximum GPU Models",
            min_value=0,
            max_value=optimizer.gpu_count * 4,
            value=int(edited_config.get("max_gpu_models", optimizer.gpu_count * 2)),
            help="Maximum number of models using GPU simultaneously",
            key="max_gpu"
        )
    
    with col2:
        edited_config["max_cpu_models"] = st.slider(
            "Maximum CPU-Only Models",
            min_value=1,
            max_value=optimizer.cpu_count,
            value=int(edited_config.get("max_cpu_models", max(1, optimizer.cpu_count // 2))),
            help="Maximum number of CPU-only models to run in parallel",
            key="max_cpu"
        )
        
        # Resource headroom 
        edited_config["gpu_memory_headroom_pct"] = st.slider(
            "GPU Memory Headroom %",
            min_value=5,
            max_value=30,
            value=int(edited_config.get("gpu_memory_headroom_pct", 15)),
            help="Percentage of GPU memory to keep free",
            key="gpu_headroom"
        )
    
    # Thread settings
    st.subheader("Threading Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edited_config["threads_per_model"] = st.slider(
            "Threads Per Model",
            min_value=1,
            max_value=max(1, optimizer.cpu_count // 2),
            value=int(edited_config.get("threads_per_model", max(1, optimizer.cpu_count // 8))),
            help="Number of threads to allocate per model",
            key="threads_per_model"
        )
    
    with col2:
        edited_config["inter_op_threads"] = st.slider(
            "Inter-Op Threads",
            min_value=1,
            max_value=8,
            value=int(edited_config.get("inter_op_threads", min(4, optimizer.cpu_count // 4))),
            help="TensorFlow inter-op parallelism threads",
            key="inter_op"
        )
        
        edited_config["intra_op_threads"] = st.slider(
            "Intra-Op Threads",
            min_value=1,
            max_value=8,
            value=int(edited_config.get("intra_op_threads", min(4, optimizer.cpu_count // 4))),
            help="TensorFlow intra-op parallelism threads",
            key="intra_op"
        )
    
    # Batch size settings
    st.subheader("Batch Size Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edited_config["base_batch_size"] = st.slider(
            "Base Batch Size",
            min_value=16,
            max_value=256,
            value=int(edited_config.get("base_batch_size", 64 if optimizer.has_gpu else 32)),
            step=16,
            help="Base batch size for all models (adjusted by model-specific factors)",
            key="base_batch"
        )
    
    # Apply button
    if st.button("Apply Advanced Settings"):
        # Update optimizer config with edited values
        for key, value in edited_config.items():
            optimizer.config[key] = value
        
        # Reconfigure TensorFlow
        optimizer._configure_tensorflow()
        
        st.success("Advanced settings updated successfully!")
        
        # Reset training metrics since they're no longer valid with new settings
        if "model_training_metrics" in st.session_state:
            st.session_state["model_training_metrics"] = {}
        
        if "model_training_times" in st.session_state:
            st.session_state["model_training_times"] = {}
    
    # Reset button
    if st.button("Reset to Default Settings"):
        # Reset to auto-configured defaults
        default_config = optimizer._auto_configure()
        
        # Update session state
        st.session_state["edited_advanced_config"] = {
            k: v for k, v in default_config.items() 
            if k not in ["model_resource_profiles"]
        }
        
        st.session_state["edited_resource_profiles"] = default_config["model_resource_profiles"]
        
        # Update optimizer config
        optimizer.config = default_config
        
        # Reconfigure TensorFlow
        optimizer._configure_tensorflow()
        
        st.success("Settings reset to default values!")
        st.experimental_rerun()


# Function to track model performance during training
def track_model_performance(model_type, runtime, metrics=None):
    """
    Track model performance and training time.
    Call this during model training to populate dashboard data.
    
    Args:
        model_type: Type of model being trained
        runtime: Training runtime in seconds
        metrics: Optional dictionary of performance metrics
    """
    # Initialize tracking dictionaries if needed
    if "model_training_metrics" not in st.session_state:
        st.session_state["model_training_metrics"] = {}
    
    if "model_training_times" not in st.session_state:
        st.session_state["model_training_times"] = {}
    
    # Update training time
    st.session_state["model_training_times"][model_type] = runtime
    
    # Update metrics if provided
    if metrics:
        st.session_state["model_training_metrics"][model_type] = metrics


# Add this to your walk_forward.py to integrate the tracking
def track_training_performance(model_type, start_time, result):
    """
    Track performance of a model after training.
    
    Args:
        model_type: Type of model that was trained
        start_time: Start time of training (from time.time())
        result: Result dictionary from training
    """
    from training_resource_optimizer_dashboard import track_model_performance
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Extract metrics if available
    metrics = None
    if isinstance(result, dict):
        metrics = {}
        for key, value in result.items():
            if key in ["mse", "rmse", "mape", "mae"]:
                metrics[key] = value
    
    # Track the performance
    track_model_performance(model_type, runtime, metrics)


# Add this to your main dashboard.py
if __name__ == "__main__":
    # Sample usage in streamlit dashboard
    import streamlit as st
    
    st.set_page_config(page_title="Training Optimizer Dashboard", layout="wide")
    
    render_training_optimizer_tab()