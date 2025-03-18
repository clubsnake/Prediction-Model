# training_resource_optimizer_dashboard.py
"""
Dashboard tab for monitoring model training performance and resource allocation.
Simply import and call the function from your main dashboard.
"""

import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from src.utils.training_optimizer import TrainingOptimizer, get_training_optimizer


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
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Hardware Resources",
            "Model Performance",
            "Resource Allocation",
            "Advanced Settings",
        ]
    )

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
        st.metric("Threads", f"{optimizer.cpu_count}")  # renamed from CPU Cores
        st.metric("System Memory", f"{optimizer.system_memory_gb:.1f} GB")

    with col2:
        st.metric("GPUs", f"{optimizer.gpu_count}")
        if optimizer.has_gpu:
            st.metric("GPU Memory", f"{optimizer.gpu_memory_gb:.1f} GB")
        else:
            st.warning("No GPUs detected. Models will run on CPU only.")
    
    # Auto-refresh checkbox instead of toggle
    auto_refresh = st.checkbox("Auto-refresh metrics", value=True, key="hardware_auto_refresh")
    if "last_refresh_time" not in st.session_state:
        st.session_state["last_refresh_time"] = time.time()
        
    # Add time interval selector when auto-refresh is enabled
    refresh_interval = 5
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", min_value=1, max_value=30, value=5)
        
        # Check if it's time to refresh
        current_time = time.time()
        if current_time - st.session_state["last_refresh_time"] >= refresh_interval:
            st.session_state["last_refresh_time"] = current_time
            # Trigger rerun to refresh data
            st.expiremental_rerun()
    else:
        # Manual refresh button when auto-refresh is off
        if st.button("Refresh Metrics Now"):
            st.session_state["last_refresh_time"] = time.time()
            # Will rerun automatically when button is clicked
    
    # Display last refresh time
    st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state['last_refresh_time']).strftime('%H:%M:%S')}")
    
    # Create metrics container
    metrics_container = st.container()
    
    # Update resource metrics
    with metrics_container:
        # Combined Resource Usage Chart Section
        st.subheader("System Resource Utilization")
        
        # Create data for the combined chart
        try:
            import psutil
            import pandas as pd
            import altair as alt
            
            # Function to get current disk IO
            def get_disk_io():
                try:
                    io_counters = psutil.disk_io_counters()
                    return {
                        "read_bytes": io_counters.read_bytes / 1024 / 1024,  # MB
                        "write_bytes": io_counters.write_bytes / 1024 / 1024,  # MB
                        "read_count": io_counters.read_count,
                        "write_count": io_counters.write_count
                    }
                except:
                    return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}
            
            # Gather resource data
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk_io = get_disk_io()
            
            # GPU data if available
            gpu_compute_percent = 0
            gpu_memory_percent = 0
            
            if optimizer.has_gpu:
                try:
                    import subprocess
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory",
                            "--format=csv,noheader",
                        ],
                        stdout=subprocess.PIPE,
                        check=True,
                        text=True,
                    )
                    parts = [part.strip() for part in result.stdout.split(",")]
                    if len(parts) >= 2:
                        gpu_compute_percent = float(parts[0].rstrip("%"))
                        gpu_memory_percent = float(parts[1].rstrip("%"))
                except:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU for summary
                            gpu_compute_percent = gpu.load * 100
                            gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal else 0
                    except:
                        pass
            
            # Create DataFrame for unified resources chart
            resource_data = pd.DataFrame([
                {"Resource": "CPU", "Utilization": cpu_percent, "Type": "Processing"},
                {"Resource": "RAM", "Utilization": memory_percent, "Type": "Memory"},
                {"Resource": "Disk Write", "Utilization": min(100, disk_io["write_bytes"]), "Type": "I/O"},
                {"Resource": "Disk Read", "Utilization": min(100, disk_io["read_bytes"]), "Type": "I/O"}
            ])
            
            if optimizer.has_gpu:
                resource_data = pd.concat([resource_data, pd.DataFrame([
                    {"Resource": "GPU", "Utilization": gpu_compute_percent, "Type": "Processing"},
                    {"Resource": "VRAM", "Utilization": gpu_memory_percent, "Type": "Memory"}
                ])])
            
            # Create chart for all resources
            resource_chart = (
                alt.Chart(resource_data)
                .mark_bar()
                .encode(
                    x=alt.X("Utilization:Q", title="Utilization (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Resource:N", title=None),
                    color=alt.Color("Type:N", scale=alt.Scale(domain=["Processing", "Memory", "I/O"],
                                                            range=["#1f77b4", "#ff7f0e", "#2ca02c"])),
                    tooltip=["Resource:N", "Utilization:Q", "Type:N"]
                )
                .properties(title="Current System Resource Utilization")
            )
            
            st.altair_chart(resource_chart, use_container_width=True)
            
            # Time series for resource history
            if "resource_history" not in st.session_state:
                st.session_state["resource_history"] = []
            
            # Update history with current values
            current_timestamp = time.time()
            
            # Add new datapoint
            st.session_state["resource_history"].append({
                "time": current_timestamp,
                "cpu": cpu_percent,
                "ram": memory_percent,
                "gpu": gpu_compute_percent,
                "vram": gpu_memory_percent,
                "disk_read": disk_io["read_bytes"],
                "disk_write": disk_io["write_bytes"]
            })
            
            # Keep only last 60 datapoints (5 minutes with 5-second refresh)
            max_history = 60
            if len(st.session_state["resource_history"]) > max_history:
                st.session_state["resource_history"] = st.session_state["resource_history"][-max_history:]
            
            # Create time series chart if we have enough data points
            if len(st.session_state["resource_history"]) > 1:
                # Convert to DataFrame for Altair
                history_df = pd.DataFrame(st.session_state["resource_history"])
                
                # Calculate relative time in seconds
                min_time = history_df["time"].min()
                history_df["relative_time"] = history_df["time"] - min_time
                
                # Melt for easier charting
                melted_df = pd.melt(
                    history_df,
                    id_vars=["time", "relative_time"],
                    value_vars=["cpu", "ram", "gpu", "vram", "disk_read", "disk_write"],
                    var_name="resource",
                    value_name="utilization"
                )
                
                # Map resource types
                resource_type_map = {
                    "cpu": "Processing", "gpu": "Processing",
                    "ram": "Memory", "vram": "Memory",
                    "disk_read": "I/O", "disk_write": "I/O"
                }
                melted_df["type"] = melted_df["resource"].map(resource_type_map)
                
                # Create time series chart
                time_chart = (
                    alt.Chart(melted_df)
                    .mark_line()
                    .encode(
                        x=alt.X("relative_time:Q", title="Time (seconds)"),
                        y=alt.Y("utilization:Q", title="Utilization (%)", scale=alt.Scale(domain=[0, 100])),
                        color="resource:N",
                        strokeDash=alt.StrokeDash("type:N"),
                        tooltip=["resource:N", "utilization:Q"]
                    )
                    .properties(title="Resource Utilization Over Time")
                )
                
                st.altair_chart(time_chart, use_container_width=True)
            
            # Display CPU Details
            st.subheader("CPU Details")
            
            # CPU usage by core
            cpu_percent_by_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Create a DataFrame for the chart
            cpu_df = pd.DataFrame(
                {
                    "Core": [f"Core {i}" for i in range(len(cpu_percent_by_core))],
                    "Usage": cpu_percent_by_core,
                }
            )
            
            # Create chart
            cpu_chart = (
                alt.Chart(cpu_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Usage:Q", title="Usage %", scale=alt.Scale(domain=[0, 100])
                    ),
                    y=alt.Y("Core:N", title=None),
                    color=alt.Color("Usage:Q", scale=alt.Scale(scheme="blues")),
                )
            )
            
            st.altair_chart(cpu_chart, use_container_width=True)
            
            # Process info
            process = psutil.Process()
            st.write("##### Process Information")
            st.write(f"Process CPU: {process.cpu_percent()}%")
            st.write(f"Thread Count: {process.num_threads()}")
            
            # Display GPU Details if available
            if optimizer.has_gpu:
                st.subheader("GPU Details")
                try:
                    import subprocess
                    # Try using nvidia-smi
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader",
                        ],
                        stdout=subprocess.PIPE,
                        check=True,
                        text=True,
                    )
                    gpu_info = []
                    for line in result.stdout.strip().split("\n"):
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) >= 7:
                            gpu_info.append(
                                {
                                    "index": parts[0],
                                    "name": parts[1],
                                    "gpu_util": float(parts[2].rstrip("%")),
                                    "mem_util": float(parts[3].rstrip("%")),
                                    "mem_used": float(parts[4].split()[0]),
                                    "mem_total": float(parts[5].split()[0]),
                                    "temperature": float(parts[6].rstrip("°C")),
                                }
                            )
                except Exception as e:
                    # Fallback for non-NVIDIA GPUs using GPUtil if available.
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        gpu_info = []
                        for gpu in gpus:
                            gpu_info.append({
                                "index": gpu.id,
                                "name": gpu.name,
                                "gpu_util": gpu.load * 100,
                                "mem_util": (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal else 0,
                                "mem_used": gpu.memoryUsed,
                                "mem_total": gpu.memoryTotal,
                                "temperature": gpu.temperature,
                            })
                    except Exception as e2:
                        st.write("Enhanced GPU details unavailable")

                # Display individual GPU details
                if gpu_info:
                    for i, gpu in enumerate(gpu_info):
                        st.markdown(f"### {gpu['name']} (GPU {gpu['index']})")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Compute Utilization", f"{gpu['gpu_util']:.1f}%")
                            st.progress(gpu["gpu_util"] / 100)
                        with c2:
                            mem_pct = gpu["mem_used"] / gpu["mem_total"] * 100 if gpu["mem_total"] else 0
                            st.metric("Memory Usage", f"{gpu['mem_util']:.1f}% ({gpu['mem_used']} / {gpu['mem_total']} MiB)")
                            st.progress(mem_pct / 100)
                        with c3:
                            st.metric("Temperature", f"{gpu['temperature']:.1f}°C")
                    # Combined chart for GPU metrics
                    if gpu_info:
                        df = pd.DataFrame(gpu_info)
                        df_melt = df.melt(id_vars=["index", "name"], value_vars=["gpu_util", "mem_util"],
                                          var_name="Metric", value_name="Value")
                        chart = (
                            alt.Chart(df_melt)
                            .mark_bar()
                            .encode(
                                x=alt.X("name:N", title="GPU"),
                                y=alt.Y("Value:Q", title="Utilization (%)", scale=alt.Scale(domain=[0, 100])),
                                color=alt.Color("Metric:N", scale=alt.Scale(scheme="category10")),
                                column=alt.Column("Metric:N", header=alt.Header(title="")),
                            )
                            .properties(title="GPU Utilization Metrics")
                        )
                        st.altair_chart(chart, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error displaying resource metrics: {e}")
            
    # Display model resource usage if data is available
    st.subheader("Resource Usage By Model")
    if "model_resource_usage" in st.session_state and st.session_state["model_resource_usage"]:
        try:
            model_resources = st.session_state["model_resource_usage"]
            model_df = pd.DataFrame(model_resources)
            
            # Create a bar chart for each resource type
            resource_types = ["cpu_percent", "memory_mb", "gpu_percent", "gpu_memory_mb"]
            labels = {"cpu_percent": "CPU %", "memory_mb": "RAM (MB)", 
                      "gpu_percent": "GPU %", "gpu_memory_mb": "VRAM (MB)"}
            
            # Filter only columns that exist in the dataframe
            resource_types = [r for r in resource_types if r in model_df.columns]
            
            # Create selector for resource type
            resource_type = st.selectbox("Resource Metric", 
                                         options=resource_types,
                                         format_func=lambda x: labels.get(x, x))
            
            if resource_type in model_df.columns:
                chart = (
                    alt.Chart(model_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{resource_type}:Q", title=labels.get(resource_type, resource_type)),
                        y=alt.Y("model:N", title="Model", sort=f"-{resource_type}"),
                        color=alt.Color(f"{resource_type}:Q", scale=alt.Scale(scheme="viridis"))
                    )
                    .properties(title=f"Model {labels.get(resource_type, resource_type)} Usage")
                )
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(f"No data available for {labels.get(resource_type, resource_type)}")
        except Exception as e:
            st.error(f"Error displaying model resource usage: {e}")
    else:
        st.info("No model resource usage data available. Start training models to see resource usage.")

    # Resource allocation summary
    st.subheader("Resource Allocation for Models")

    # Create a table showing resource allocation for each model type
    model_resources = []
    for model_type, resources in optimizer.config["model_resource_profiles"].items():
        model_resources.append(
            {
                "Model Type": model_type.upper(),
                "GPU Usage": (
                    f"{resources['gpu_memory_fraction']*100:.1f}%"
                    if resources["gpu_memory_fraction"] > 0
                    else "None"
                ),
                "CPU Weight": f"{resources['cpu_weight']:.1f}x",
                "RAM": f"{resources['ram_gb']:.1f} GB",
                "TensorFlow": "Yes" if resources["tf_model"] else "No",
            }
        )

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
        st.info("No training data available yet. Start training models to see performance metrics.")
        return

    # Real data visualization
    st.subheader("Training Times")

    # Create DataFrame for training times
    time_df = pd.DataFrame(
        [{"model": model, "seconds": time} for model, time in times.items()]
    )

    # Sort by time
    time_df = time_df.sort_values("seconds", ascending=False)

    # Create chart
    time_chart = (
        alt.Chart(time_df)
        .mark_bar()
        .encode(
            x=alt.X("seconds:Q", title="Training Time (seconds)"),
            y=alt.Y("model:N", title="Model Type", sort="-x"),
            color=alt.Color(
                "seconds:Q", 
                scale=alt.Scale(scheme="redyellowgreen", domain=[time_df["seconds"].min(), time_df["seconds"].max()], reverse=True),
                legend=alt.Legend(title="Training Time")
            ),
            tooltip=["model:N", "seconds:Q"]
        )
        .properties(title="Training Time by Model Type")
    )

    st.altair_chart(time_chart, use_container_width=True)

    # Check for imbalances and identify bottlenecks
    if len(time_df) > 1:
        max_time = time_df["seconds"].max()
        min_time = time_df["seconds"].min()
        avg_time = time_df["seconds"].mean()

        if max_time > min_time * 3:
            slowest = time_df.iloc[0]["model"]
            fastest = time_df.iloc[-1]["model"]

            st.warning(
                f"⚠️ **Performance Bottleneck Detected**: {slowest} is {max_time/min_time:.1f}x slower than {fastest}"
            )

            # Enhanced resource allocation suggestion
            st.info(
                f"""
            **Optimization Suggestion**: Consider adjusting resource allocation for {slowest}:
            - Increase GPU memory fraction if it's a neural network model
            - Increase CPU weight if it's running on CPU
            - Adjust batch size for better throughput
            
            Go to the Resource Allocation tab to make these adjustments.
            """
            )

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
                    metric_data.append(
                        {
                            "model": model,
                            "metric": metric_name,
                            "value": model_metrics[metric_name],
                        }
                    )

        if metric_data:
            metric_df = pd.DataFrame(metric_data)

            # Create tabs for different metrics
            metric_tabs = st.tabs(list(all_metrics))

            for i, metric_name in enumerate(all_metrics):
                with metric_tabs[i]:
                    # Filter data for this metric
                    this_metric = metric_df[metric_df["metric"] == metric_name]

                    if not this_metric.empty:
                        # Sort appropriately (lower is better for most metrics)
                        this_metric = this_metric.sort_values("value", ascending=True)

                        # Create enhanced chart with color gradient
                        chart = (
                            alt.Chart(this_metric)
                            .mark_bar()
                            .encode(
                                x=alt.X("value:Q", title=f"{metric_name.upper()}"),
                                y=alt.Y("model:N", title="Model Type", sort="x"),
                                color=alt.Color(
                                    "value:Q", 
                                    scale=alt.Scale(scheme="blueorange", domain=[this_metric["value"].min(), this_metric["value"].max()]),
                                    legend=alt.Legend(title=metric_name.upper())
                                ),
                                tooltip=["model:N", "value:Q"]
                            )
                        )

                        st.altair_chart(chart, use_container_width=True)

                        # Performance correlation analysis
                        if "model_training_times" in st.session_state:
                            times = st.session_state["model_training_times"]
                            metric_times = []
                            metric_values = []
                            
                            for model in this_metric["model"]:
                                if model in times:
                                    model_value = this_metric[this_metric["model"] == model]["value"].values[0]
                                    metric_times.append(times[model])
                                    metric_values.append(model_value)
                            
                            if len(metric_times) > 1:
                                import numpy as np
                                correlation = np.corrcoef(metric_times, metric_values)[0, 1]
                                if abs(correlation) > 0.5:
                                    corr_message = "strong positive" if correlation > 0 else "strong negative"
                                    st.info(f"There is a {corr_message} correlation ({correlation:.2f}) between training time and {metric_name}.")
                        
                        # Show best model
                        best_model = this_metric.iloc[0]["model"]
                        best_value = this_metric.iloc[0]["value"]

                        st.success(
                            f"Best model for {metric_name.upper()}: **{best_model}** ({best_value:.6f})"
                        )
    else:
        st.info("No performance metrics available yet.")


def render_resource_allocation_section(optimizer: TrainingOptimizer):
    """Render resource allocation controls for all model types."""
    st.header("Resource Allocation")
    st.write(
        """
    Fine-tune resource allocation for each model type. These settings will be applied
    the next time you train models.
    """
    )

    # Ensure config has model_resource_profiles
    if "model_resource_profiles" not in optimizer.config:
        st.error("Resource profiles not available in optimizer configuration.")
        return

    # Store current profiles
    current_profiles = optimizer.config["model_resource_profiles"]

    # Deep copy for modification
    if "edited_resource_profiles" not in st.session_state:
        st.session_state["edited_resource_profiles"] = {
            model: resources.copy() for model, resources in current_profiles.items()
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
                        key=f"{model_type}_gpu",
                    )

                # CPU allocation
                resources["cpu_weight"] = st.slider(
                    "CPU Weight",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(resources["cpu_weight"]),
                    step=0.5,
                    help="Relative CPU allocation (higher = more CPU cores)",
                    key=f"{model_type}_cpu",
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
                    key=f"{model_type}_ram",
                )

                # Batch size factor - Add default value if key doesn't exist
                if "batch_size_factor" not in resources:
                    resources["batch_size_factor"] = 1.0  # Default value
                
                resources["batch_size_factor"] = st.slider(
                    "Batch Size Factor",
                    min_value=0.2,
                    max_value=2.0,
                    value=float(resources["batch_size_factor"]),
                    step=0.1,
                    help="Relative batch size (higher = larger batches)",
                    key=f"{model_type}_batch",
                )

            # Additional settings for neural network models
            if resources.get("tf_model", False):
                st.write("##### Neural Network Specific Settings")

                col1, col2 = st.columns(2)
                with col1:
                    # Mixed precision
                    use_mixed = st.checkbox(
                        "Use Mixed Precision",
                        value=optimizer.config.get("use_mixed_precision", True)
                        and optimizer.has_gpu,
                        disabled=not optimizer.has_gpu,
                        help="Use mixed precision for faster training (requires GPU)",
                        key=f"{model_type}_mixed",
                    )

                with col2:
                    # XLA compilation
                    use_xla = st.checkbox(
                        "Use XLA Compilation",
                        value=optimizer.config.get("use_xla", True)
                        and optimizer.has_gpu,
                        disabled=not optimizer.has_gpu,
                        help="Use XLA compilation for faster training (requires GPU)",
                        key=f"{model_type}_xla",
                    )

    # Apply button for resource changes
    if st.button("Apply Resource Allocation Changes"):
        # Update optimizer config with edited profiles
        optimizer.config["model_resource_profiles"] = st.session_state[
            "edited_resource_profiles"
        ]

        # Update mixed precision and XLA settings based on first neural network model
        for model_type, resources in edited_profiles.items():
            if resources.get("tf_model", False):
                # Use settings from this model for global config
                optimizer.config["use_mixed_precision"] = st.session_state.get(
                    f"{model_type}_mixed", False
                )
                optimizer.config["use_xla"] = st.session_state.get(
                    f"{model_type}_xla", False
                )
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

    st.write(
        """
    These settings control how the training optimizer allocates resources and 
    manages parallel execution. Adjust with caution as they can significantly impact
    training performance.
    """
    )

    # Create copy of config for editing
    if "edited_advanced_config" not in st.session_state:
        st.session_state["edited_advanced_config"] = {
            k: v
            for k, v in optimizer.config.items()
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
            value=int(
                edited_config.get(
                    "max_parallel_models", min(8, max(1, optimizer.cpu_count // 2))
                )
            ),
            help="Maximum number of models to train in parallel",
            key="max_parallel",
        )

        edited_config["max_gpu_models"] = st.slider(
            "Maximum GPU Models",
            min_value=0,
            max_value=optimizer.gpu_count * 4,
            value=int(edited_config.get("max_gpu_models", optimizer.gpu_count * 2)),
            help="Maximum number of models using GPU simultaneously",
            key="max_gpu",
        )

    with col2:
        edited_config["max_cpu_models"] = st.slider(
            "Maximum CPU-Only Models",
            min_value=1,
            max_value=optimizer.cpu_count,
            value=int(
                edited_config.get("max_cpu_models", max(1, optimizer.cpu_count // 2))
            ),
            help="Maximum number of CPU-only models to run in parallel",
            key="max_cpu",
        )

        # Resource headroom
        edited_config["gpu_memory_headroom_pct"] = st.slider(
            "GPU Memory Headroom %",
            min_value=5,
            max_value=30,
            value=int(edited_config.get("gpu_memory_headroom_pct", 15)),
            help="Percentage of GPU memory to keep free",
            key="gpu_headroom",
        )

    # Thread settings
    st.subheader("Threading Configuration")

    col1, col2 = st.columns(2)

    with col1:
        edited_config["threads_per_model"] = st.slider(
            "Threads Per Model",
            min_value=1,
            max_value=max(1, optimizer.cpu_count // 2),
            value=int(
                edited_config.get("threads_per_model", max(1, optimizer.cpu_count // 8))
            ),
            help="Number of threads to allocate per model",
            key="threads_per_model",
        )

    with col2:
        edited_config["inter_op_threads"] = st.slider(
            "Inter-Op Threads",
            min_value=1,
            max_value=8,
            value=int(
                edited_config.get("inter_op_threads", min(4, optimizer.cpu_count // 4))
            ),
            help="TensorFlow inter-op parallelism threads",
            key="inter_op",
        )

        edited_config["intra_op_threads"] = st.slider(
            "Intra-Op Threads",
            min_value=1,
            max_value=8,
            value=int(
                edited_config.get("intra_op_threads", min(4, optimizer.cpu_count // 4))
            ),
            help="TensorFlow intra-op parallelism threads",
            key="intra_op",
        )

    # Batch size settings
    st.subheader("Batch Size Configuration")

    col1, col2 = st.columns(2)

    with col1:
        edited_config["base_batch_size"] = st.slider(
            "Base Batch Size",
            min_value=16,
            max_value=256,
            value=int(
                edited_config.get("base_batch_size", 64 if optimizer.has_gpu else 32)
            ),
            step=16,
            help="Base batch size for all models (adjusted by model-specific factors)",
            key="base_batch",
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
            k: v
            for k, v in default_config.items()
            if k not in ["model_resource_profiles"]
        }

        st.session_state["edited_resource_profiles"] = default_config[
            "model_resource_profiles"
        ]

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
    from src.dashboard.training_resource_optimizer_dashboard import track_model_performance
    
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


# Function to track model resource usage
def track_model_resource_usage(model_type, cpu_percent=0, memory_mb=0, gpu_percent=0, gpu_memory_mb=0):
    """
    Track resource usage for a model during training.
    Call this periodically during model training to collect resource usage data.
    
    Args:
        model_type: Type of model being tracked
        cpu_percent: CPU usage percentage
        memory_mb: Memory usage in MB
        gpu_percent: GPU usage percentage (0 if not using GPU)
        gpu_memory_mb: GPU memory usage in MB (0 if not using GPU)
    """
    if "model_resource_usage" not in st.session_state:
        st.session_state["model_resource_usage"] = []
        
    # Find if we already have an entry for this model
    found = False
    for i, entry in enumerate(st.session_state["model_resource_usage"]):
        if entry["model"] == model_type:
            # Update existing entry
            st.session_state["model_resource_usage"][i].update({
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "gpu_percent": gpu_percent,
                "gpu_memory_mb": gpu_memory_mb,
                "last_updated": time.time()
            })
            found = True
            break
            
    # Add new entry if not found
    if not found:
        st.session_state["model_resource_usage"].append({
            "model": model_type,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb, 
            "gpu_percent": gpu_percent,
            "gpu_memory_mb": gpu_memory_mb,
            "last_updated": time.time()
        })
        
    # Remove old entries (older than 5 minutes)
    current_time = time.time()
    st.session_state["model_resource_usage"] = [
        entry for entry in st.session_state["model_resource_usage"] 
        if current_time - entry.get("last_updated", 0) < 300
    ]
