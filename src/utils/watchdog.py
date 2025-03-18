"""
ML Tuning Watchdog

Features:
- Primary monitoring of tested_models.yaml for trial progress
- Secondary monitoring of progress.yaml for overall state
- Learning phase tolerance for high RMSE/MAPE and NaN values
- Trial-based error thresholds that become stricter as training progresses
- Specific tracking of individual models in the ensemble
- Detailed but clean error logging
- Automatic recovery of stalled processes
- Handles partial ensemble failures
"""

import json
import logging
import math
import os
import shutil  # For file backups
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("tuning_watchdog.log"), logging.StreamHandler()],
)
logger = logging.getLogger("TuningWatchdog")

# Optional imports with fallbacks
try:
    import psutil
except ImportError:
    logger.warning("psutil not installed. Resource monitoring will be limited.")
    psutil = None

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    logger.warning("GPUtil not installed. GPU monitoring will be disabled.")
    HAS_GPUTIL = False


class TuningWatchdog:
    """
    Watchdog for monitoring and automatically restarting ML tuning processes.
    
    This revised watchdog implementation is more tolerant of high loss values 
    and unstable metrics during early learning phases, allowing models with wide
    hyperparameter ranges to explore without being prematurely flagged as problematic.
    """

    def __init__(
        self,
        data_dir=None,
        tested_models_file=None,
        progress_file=None,
        status_file=None,
        ticker=None,  # Changed to None to use config default
        timeframe=None,  # Changed to None to use config default
        check_interval=600,  # 10 minutes
        max_stale_time=10800,  # 180 minutes
        startup_grace_period=120,  # 2 minute
        shutdown_grace_period=120,  # 2 minutes
        auto_restart=True,
        max_consecutive_restarts=10,
        restart_cooldown=3600,  # 1 hour
        monitor_resources=True,
        create_backups=True,
        notification_level="ERROR",  # "INFO", "WARNING", "ERROR"
        dashboard_update_interval=60,  # seconds
        log_rotation_size=10_000_000,  # 10MB
        # New parameters for learning phase tolerance
        min_trials_before_error_flagging=20000,  # Minimum trials before flagging errors
        rmse_threshold_start=100000,  # High initial RMSE threshold
        rmse_threshold_end=1000,  # Lower RMSE threshold after sufficient trials
        max_trials_for_full_error_checking=20000,  # Trials before full error checking
        ignore_nan_metrics_before_trials=30000,  # Ignore NaN metrics in early trials
        error_logging_enabled=True  # New parameter for toggling error logging
    ):
        """
        Initialize the tuning watchdog with enhanced features and learning phase tolerance.

        Args:
            data_dir: Path to data directory (will try to auto-detect if None)
            tested_models_file: Path to tested_models.yaml (will construct from data_dir if None)
            progress_file: Path to progress.yaml (will construct from data_dir if None)
            status_file: Path to tuning_status.txt (will construct from data_dir if None)
            ticker: Default ticker to use when restarting tuning (uses config default if None)
            timeframe: Default timeframe to use when restarting tuning (uses config default if None)
            check_interval: How often to check for stalls (seconds)
            max_stale_time: Maximum time without progress before restarting (seconds)
            startup_grace_period: Time to wait after starting before monitoring (seconds)
            shutdown_grace_period: Time to wait after stopping before confirming (seconds)
            auto_restart: Whether to automatically restart failed tuning
            max_consecutive_restarts: Maximum number of consecutive restarts before cooling down
            restart_cooldown: Time to wait after max consecutive restarts (seconds)
            monitor_resources: Whether to monitor system resources
            create_backups: Whether to create backups before restart attempts
            notification_level: Minimum level for notifications ("INFO", "WARNING", "ERROR")
            dashboard_update_interval: How often to update the dashboard (seconds)
            log_rotation_size: Size threshold for log rotation (bytes)
            min_trials_before_error_flagging: Minimum trials before flagging model errors
            rmse_threshold_start: Initial high RMSE threshold for early trials
            rmse_threshold_end: RMSE threshold after sufficient trials
            max_trials_for_full_error_checking: Number of trials before applying full error checks
            ignore_nan_metrics_before_trials: Ignore NaN metrics in early trials
            error_logging_enabled: Whether to enable error logging
        """
        # Import config to get default values
        try:
            # Try to import from config
            from config.config_loader import INTERVAL, TICKER

            default_ticker = TICKER
            default_timeframe = INTERVAL
        except (ImportError, NameError):
            logger.warning("Could not import config_loader, using hardcoded defaults")
            default_ticker = "ETH-USD"
            default_timeframe = "1d"

        # Setup paths
        self.data_dir = self._find_data_dir() if data_dir is None else data_dir
        self.tested_models_file = (
            os.path.join(self.data_dir, "tested_models.yaml")
            if tested_models_file is None
            else tested_models_file
        )
        self.progress_file = (
            os.path.join(self.data_dir, "progress.yaml")
            if progress_file is None
            else progress_file
        )
        self.status_file = (
            os.path.join(self.data_dir, "tuning_status.txt")
            if status_file is None
            else status_file
        )

        # New directories
        self.log_dir = os.path.join(self.data_dir, "logs")
        self.backup_dir = os.path.join(self.data_dir, "backups")
        self.dashboard_dir = os.path.join(self.data_dir, "dashboard")

        # Create directories if they don't exist
        for directory in [self.log_dir, self.backup_dir, self.dashboard_dir]:
            os.makedirs(directory, exist_ok=True)

        # Error log and dashboard paths
        self.error_log_file = os.path.join(self.log_dir, "watchdog_errors.log")
        self.resource_log_file = os.path.join(self.log_dir, "resource_usage.log")
        self.dashboard_file = os.path.join(self.dashboard_dir, "status_dashboard.html")

        # Configuration
        self.ticker = ticker if ticker is not None else default_ticker
        self.timeframe = timeframe if timeframe is not None else default_timeframe
        self.check_interval = check_interval
        self.max_stale_time = max_stale_time
        self.startup_grace_period = startup_grace_period
        self.shutdown_grace_period = shutdown_grace_period
        self.auto_restart = auto_restart
        self.max_consecutive_restarts = max_consecutive_restarts
        self.restart_cooldown = restart_cooldown

        # Feature flags
        self.monitor_resources = monitor_resources
        self.create_backups = create_backups
        self.notification_level = notification_level
        self.dashboard_update_interval = dashboard_update_interval
        self.log_rotation_size = log_rotation_size

        # Learning phase tolerance parameters - expanded to allow for massive exploration
        self.min_trials_before_error_flagging = 100000  # Massively increased - practically disabled
        self.rmse_threshold_for_convergence = 50.0  # Only consider converged when RMSE drops below this
        self.consecutive_improvements_required = 5  # Require consecutive improvements before considering convergence
        self.convergence_by_model_type = {}  # Track which models have converged
        self.best_metric_by_model = {}  # Track best metric to identify improvements

        # Setup learning phase tolerance parameters using constructor values
        self.min_trials_before_error_flagging = min_trials_before_error_flagging
        self.rmse_threshold_start = rmse_threshold_start
        self.rmse_threshold_end = rmse_threshold_end
        self.error_logging_enabled = error_logging_enabled

        # State tracking
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.consecutive_restarts = 0
        self.last_restart_time = 0
        self.last_models_update_time = 0
        self.last_progress_update_time = 0
        self.last_models_count = 0
        self.model_performance_history = {}  # Track performance by model type
        self.model_error_count = {}  # Count errors by model type
        self.model_trial_counts = {}  # Track number of trials per model type

        # Additional tracking data
        self.resource_history = []
        self.trial_history_by_model = defaultdict(list)
        self.error_count_by_category = Counter()
        self.last_dashboard_update = 0

        # Initialize dashboard data structure
        self.dashboard_data = {
            "recent_activity": [],
            "errors": [],
            "resource_usage": [],
            "model_performance": {},
        }

        # Initialize error log
        self._init_error_log()

        # Initialize dashboard
        self._init_dashboard()

        logger.info(f"Watchdog initialized with data_dir: {self.data_dir}")
        logger.info(f"Primary monitoring: {self.tested_models_file}")
        logger.info(f"Secondary monitoring: {self.progress_file}")
        logger.info(f"Status file: {self.status_file}")
        logger.info(f"Error log: {self.error_log_file}")
        logger.info(f"Learning phase tolerance enabled: min_trials={min_trials_before_error_flagging}, RMSE threshold range: {rmse_threshold_start}-{rmse_threshold_end}")

    def _init_error_log(self):
        """Initialize the error log file with header."""
        try:
            # Check if we need to rotate logs
            if (
                os.path.exists(self.error_log_file)
                and os.path.getsize(self.error_log_file) > self.log_rotation_size
            ):
                # Rotate log file
                backup_name = (
                    f"{self.error_log_file}.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
                shutil.copy2(self.error_log_file, backup_name)
                logger.info(f"Rotated error log to {backup_name}")

            # Create or truncate the log file
            with open(self.error_log_file, "w") as f:
                f.write("# Tuning Watchdog Error Log\n")
                f.write(f"# Started: {datetime.now().isoformat()}\n")
                f.write(f"# Monitoring data directory: {self.data_dir}\n")
                f.write(f"# Ticker: {self.ticker} | Timeframe: {self.timeframe}\n")
                f.write(
                    f"# Auto-restart: {'Enabled' if self.auto_restart else 'Disabled'}\n\n"
                )
                f.write(f"# Learning phase tolerance enabled: min_trials={self.min_trials_before_error_flagging}, "
                      f"RMSE threshold range: {self.rmse_threshold_start}-{self.rmse_threshold_end}\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize error log: {e}")

    def _init_dashboard(self):
        """Initialize the status dashboard file."""
        try:
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="refresh" content="{self.dashboard_update_interval}">
                <title>Tuning Watchdog Dashboard</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .header {{
                        background-color: #2c3e50;
                        color: white;
                        padding: 10px 20px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .status-card {{
                        background-color: white;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        padding: 15px;
                        margin-bottom: 20px;
                    }}
                    .status-healthy {{
                        background-color: #27ae60;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }}
                    .status-warning {{
                        background-color: #f39c12;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }}
                    .status-error {{
                        background-color: #e74c3c;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }}
                    .status-unknown {{
                        background-color: #95a5a6;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }}
                    .status-learning {{
                        background-color: #3498db;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }}
                    .grid-container {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        text-align: left;
                        padding: 8px;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        color: #7f8c8d;
                        font-size: 0.8em;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Tuning Watchdog Dashboard</h1>
                    <p>Initializing monitoring for {self.ticker}/{self.timeframe}...</p>
                    <p><small>Massive learning phase enabled: min_trials={self.min_trials_before_error_flagging}, 
                    convergence RMSE threshold: {self.rmse_threshold_for_convergence}</small></p>
                </div>
                
                <div class="status-card">
                    <h2>System Status</h2>
                    <p>Dashboard is being initialized. Please refresh in a few moments.</p>
                </div>
                
                <div class="footer">
                    <p>Last updated: {datetime.now().isoformat()}</p>
                    <p>Tuning Watchdog v1.1 (Learning Phase Tolerance)</p>
                </div>
            </body>
            </html>
            """

            with open(self.dashboard_file, "w") as f:
                f.write(dashboard_html)

            logger.info(f"Initialized dashboard at {self.dashboard_file}")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")

    def _find_data_dir(self):
        """
        Attempt to find the data directory based on project structure.

        Returns:
            str: Absolute path to the detected data directory
        """
        # Try different relative paths that might lead to the data directory
        potential_paths = [
            "data",  # Current directory
            "../data",  # One level up
            "../../data",  # Two levels up
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data"
            ),  # Same dir as script
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
            ),  # One up from script
        ]

        for path in potential_paths:
            if os.path.exists(path) and os.path.isdir(path):
                # Check for key files that would indicate this is our data directory
                if any(
                    os.path.exists(os.path.join(path, f))
                    for f in [
                        "tested_models.yaml",
                        "progress.yaml",
                        "tuning_status.txt",
                    ]
                ):
                    return os.path.abspath(path)

        # If we can't find it, default to a reasonable location
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(default_path, exist_ok=True)
        logger.warning(
            f"Could not locate data directory, defaulting to: {default_path}"
        )
        return default_path

    def log_error(self, error_type, message, details=None, model_type=None, trial_number=None):
        """
        Enhanced error logging with categorization and tracking, considering learning phases.

        Args:
            error_type: Type/category of error
            message: Main error message
            details: Additional error details or traceback
            model_type: Type of model reporting the error (optional)
            trial_number: Trial number associated with the error (optional)
        """
        # Check if this is a model-specific error in early learning phase
        if model_type and trial_number is not None:
            # Get trial count for this model type
            model_trials = self.model_trial_counts.get(model_type, 0)
            
            # Update trial count
            if trial_number > model_trials:
                self.model_trial_counts[model_type] = trial_number
                
            # Check if we're still in early learning phase for this model
            if trial_number < self.min_trials_before_error_flagging:
                # Don't log certain errors during early learning phase
                if "high_rmse" in error_type.lower() or "invalid_metrics" in error_type.lower():
                    logger.debug(f"Ignoring '{error_type}' in learning phase (trial {trial_number}) for {model_type}")
                    return
                
                # Downgrade error to learning warning
                if "error" in error_type.lower():
                    error_type = f"Learning Phase: {error_type}"
                    message = f"[Early Learning - Trial {trial_number}] {message}"

        # Log to standard logger
        logger.error(f"{error_type}: {message}")

        # Track error counts
        self.error_count_by_category[error_type] += 1

        # Only send notifications for serious errors
        notification_levels = {"INFO": 0, "WARNING": 1, "ERROR": 2}

        error_level = 2  # Default to ERROR
        for level_name, level_value in notification_levels.items():
            if level_name.lower() in error_type.lower():
                error_level = level_value
                break

        # Check if this error warrants a notification
        should_notify = error_level >= notification_levels.get(
            self.notification_level, 2
        )

        try:
            # Log to error log file
            with open(self.error_log_file, "a") as f:
                f.write(f"\n[{datetime.now().isoformat()}] {error_type}\n")
                if model_type:
                    f.write(f"Model: {model_type}\n")
                if trial_number is not None:
                    f.write(f"Trial: {trial_number}\n")
                f.write(f"{message}\n")
                if details:
                    f.write("Details:\n")
                    f.write(f"{details}\n")
                f.write("-" * 80 + "\n")

            # Add to dashboard data
            timestamp = datetime.now().isoformat()
            dashboard_entry = {
                "type": "error" if "error" in error_type.lower() else "warning",
                "category": error_type,
                "message": message,
                "timestamp": timestamp,
            }
            
            if model_type:
                dashboard_entry["model_type"] = model_type
            if trial_number is not None:
                dashboard_entry["trial"] = trial_number
                
            self.dashboard_data["recent_activity"].insert(0, dashboard_entry)

            # Keep dashboard activity limited to 20 entries
            if len(self.dashboard_data["recent_activity"]) > 20:
                self.dashboard_data["recent_activity"] = self.dashboard_data[
                    "recent_activity"
                ][:20]

            # Send notification if configured
            if should_notify:
                self._send_notification(error_type, message)

        except Exception as e:
            logger.error(f"Failed to write to error log: {e}")

    def _send_notification(self, error_type, message):
        """
        Send a notification about a serious error.

        Args:
            error_type: Type of error
            message: Error message
        """
        try:
            # This is a stub that could be expanded to send emails, SMS, etc.
            logger.warning(f"Would send notification: {error_type} - {message}")

            # You could add code here to send notifications via:
            # - Email (using smtplib)
            # - SMS (using a service like Twilio)
            # - Slack/Discord webhooks
            # - Desktop notifications

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def backup_critical_files(self):
        """
        Create backups of critical files before attempting fixes.

        Returns:
            bool: True if backups were created successfully, False otherwise
        """
        if not self.create_backups:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Files to backup
            critical_files = [
                self.tested_models_file,
                self.progress_file,
                self.status_file,
            ]

            # Add SQLite database files if they exist
            db_dir = os.path.join(self.data_dir, "DB")
            if os.path.exists(db_dir):
                for file in os.listdir(db_dir):
                    if file.endswith(".db"):
                        critical_files.append(os.path.join(db_dir, file))

            # Create backups
            for file_path in critical_files:
                if os.path.exists(file_path):
                    backup_path = os.path.join(
                        self.backup_dir, f"{os.path.basename(file_path)}.{timestamp}"
                    )
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Backed up {file_path} to {backup_path}")

            return True
        except Exception as e:
            logger.error(f"Failed to create backups: {e}")
            return False

    def monitor_system_resources(self):
        """
        Monitor system resources (CPU, RAM, disk, GPU).

        Returns:
            dict: Resource utilization data or empty dict if monitoring failed
        """
        if not self.monitor_resources or psutil is None:
            return {}

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage(self.data_dir)
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = disk.percent

            # GPU usage (if available)
            gpu_info = {}
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_info[f"gpu_{i}"] = {
                            "name": gpu.name,
                            "memory_percent": gpu.memoryUtil * 100,
                            "gpu_percent": gpu.load * 100,
                            "temperature": gpu.temperature,
                        }
                except Exception as e:
                    logger.warning(f"Error getting GPU info: {e}")

            # Process information
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)

            # Python processes
            python_processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cmdline", "memory_percent"]
            ):
                try:
                    if "python" in proc.info["name"].lower():
                        cmd = (
                            " ".join(proc.info["cmdline"])
                            if proc.info["cmdline"]
                            else "Unknown"
                        )
                        if (
                            "tuning" in cmd.lower()
                            or "train" in cmd.lower()
                            or "stream" in cmd.lower()
                        ):
                            python_processes.append(
                                {
                                    "pid": proc.info["pid"],
                                    "cmd": cmd[:100]
                                    + ("..." if len(cmd) > 100 else ""),
                                    "memory_percent": proc.info["memory_percent"],
                                }
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Compile results
            resource_data = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory": {
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb,
                    "percent": memory_percent,
                },
                "disk": {
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb,
                    "percent": disk_percent,
                },
                "process": {"memory_mb": process_memory_mb},
                "gpu": gpu_info,
                "python_processes": python_processes,
            }

            # Log resource data periodically to the resource log
            self.resource_history.append(resource_data)
            if len(self.resource_history) > 100:  # Keep last 100 entries
                self.resource_history = self.resource_history[-100:]

            # Log to file every 5 minutes
            current_time = time.time()
            if (
                not hasattr(self, "last_resource_log_time")
                or current_time - getattr(self, "last_resource_log_time", 0) > 300
            ):
                try:
                    with open(self.resource_log_file, "a") as f:
                        f.write(f"{json.dumps(resource_data)}\n")
                    self.last_resource_log_time = current_time
                except Exception as e:
                    logger.warning(f"Failed to write resource log: {e}")

            return resource_data
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
            return {}

    def check_tested_models(self):
        """
        Check tested_models.yaml for new trial completions and model-specific issues.
        
        This revised version is more tolerant of high loss values and NaN values during 
        early learning phases.

        Returns:
            dict: Status information about the tested models
        """
        result = {
            "is_updating": False,
            "model_count": 0,
            "last_update_time": 0,
            "models_by_type": {},
            "errors_detected": False,
            "error_details": [],
            "learning_phase_models": set(),  # Track models still in learning phase
        }

        try:
            if not os.path.exists(self.tested_models_file):
                return result

            # Check file modification time
            result["last_update_time"] = os.path.getmtime(self.tested_models_file)

            # Determine if the file has been updated since our last check
            result["is_updating"] = (
                result["last_update_time"] > self.last_models_update_time
            )

            # Read and analyze the tested models file
            with open(self.tested_models_file, "r") as f:
                models_data = yaml.safe_load(f) or []

            if not isinstance(models_data, list):
                self.log_error(
                    "Format Error",
                    f"tested_models.yaml is not a list: {type(models_data)}",
                )
                return result

            result["model_count"] = len(models_data)

            # Only process in detail if we have new models
            if result["is_updating"] or result["model_count"] > self.last_models_count:
                # Process model data to detect patterns or issues
                models_by_type = {}

                # Look at the most recent models (up to 20)
                recent_models = (
                    models_data[-20:] if len(models_data) > 20 else models_data
                )

                for model in recent_models:
                    if not isinstance(model, dict):
                        continue

                    model_type = model.get(
                        "model_type",
                        model.get("params", {}).get("model_type", "unknown"),
                    )

                    if model_type not in models_by_type:
                        models_by_type[model_type] = []

                    models_by_type[model_type].append(model)

                    # Check for error patterns, with trial-based thresholds
                    self._check_model_for_errors(
                        model, 
                        result["error_details"],
                        learning_phase_models=result["learning_phase_models"]
                    )

                result["models_by_type"] = {
                    model_type: len(models)
                    for model_type, models in models_by_type.items()
                }

                # Update our state
                self.last_models_update_time = result["last_update_time"]
                self.last_models_count = result["model_count"]

            result["errors_detected"] = len(result["error_details"]) > 0

            return result

        except Exception as e:
            error_msg = f"Error checking tested models: {e}"
            self.log_error("Monitoring Error", error_msg, traceback.format_exc())
            result["error"] = str(e)
            return result

    def _check_model_for_errors(self, model, error_list, learning_phase_models=None):
        """
        Check a single model entry for error patterns, with significantly expanded learning phase tolerance.
        
        This revised implementation:
        1. Uses a dynamic learning phase based on performance metrics
        2. Only starts flagging errors after a model shows signs of convergence 
        3. Uses much higher thresholds (100,000 trials or RMSE below 50)
        4. Tracks model-specific metrics to determine learning progress
        
        Args:
            model: The model data dictionary
            error_list: List to append errors to
            learning_phase_models: Set for tracking models still in learning phase
        """
        # Extract relevant information
        model_type = model.get("model_type", "unknown")
        trial_number = model.get("trial_number", 0)
        rmse = model.get("rmse", float("inf"))
        mape = model.get("mape", float("inf"))
        metrics = model.get("metrics", {})

        # Initialize tracking for this model type if not already done
        if model_type not in self.model_performance_history:
            self.model_performance_history[model_type] = []
            self.model_error_count[model_type] = 0
            self.model_trial_counts[model_type] = 0
            self.convergence_by_model_type[model_type] = False
            self.best_metric_by_model[model_type] = float("inf")

        # Update trial count
        self.model_trial_counts[model_type] = max(
            self.model_trial_counts[model_type], trial_number
        )

        # Check if this model is still in the learning phase
        if (
            trial_number < self.min_trials_before_error_flagging
            or rmse > self.rmse_threshold_for_convergence
        ):
            learning_phase_models.add(model_type)
            return

        # Check for convergence
        if rmse < self.best_metric_by_model[model_type]:
            self.best_metric_by_model[model_type] = rmse
            self.convergence_by_model_type[model_type] = True

        # Check for errors
        if not self.error_logging_enabled or self.rmse_threshold_start is None:
            return

        if rmse > self.rmse_threshold_end:
            error_list.append(
                {
                    "model_type": model_type,
                    "trial_number": trial_number,
                    "error_type": "High RMSE",
                    "message": f"RMSE {rmse} exceeds threshold {self.rmse_threshold_end}",
                }
            )
            self.model_error_count[model_type] += 1

        if math.isnan(rmse) or math.isnan(mape):
            error_list.append(
                {
                    "model_type": model_type,
                    "trial_number": trial_number,
                    "error_type": "Invalid Metrics",
                    "message": f"RMSE or MAPE is NaN (RMSE: {rmse}, MAPE: {mape})",
                }
            )
            self.model_error_count[model_type] += 1

        # Track performance history
        self.model_performance_history[model_type].append(rmse)

    def create_streamlit_component(self):
        """
        Create a comprehensive Streamlit component to display the watchdog status and controls.
        
        This method provides detailed monitoring capabilities including:
        - Logs and activity tracking
        - Model performance visualization
        - System resource monitoring
        - Tested models analysis
        """
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import time
        from datetime import datetime
        
        # Create main container for watchdog information
        st.subheader("Tuning Watchdog Monitor")
        
        # Create status indicator with more details
        status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
        
        with status_col1:
            if self.running:
                st.success("● ACTIVE")
            else:
                st.warning("○ INACTIVE")
        
        with status_col2:
            st.write(f"**Monitoring**: {self.ticker}/{self.timeframe}")
            st.write(f"**Check interval**: {self.check_interval} seconds")
            
        with status_col3:
            # Add quick stats
            if hasattr(self, 'model_error_count'):
                total_errors = sum(self.model_error_count.values()) if self.model_error_count else 0
                st.write(f"**Total errors**: {total_errors}")
        
        # Add controls with more options
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if not self.running:
                if st.button("▶️ Start Watchdog", key="start_watchdog", use_container_width=True):
                    self.start()
                    st.success("Watchdog started")
            else:
                if st.button("⏹️ Stop Watchdog", key="stop_watchdog", use_container_width=True):
                    self.stop()
                    st.info("Watchdog stopped")
        
        with control_col2:
            if st.button("🔍 Run Diagnostics", key="run_diagnostics", use_container_width=True):
                with st.spinner("Running diagnostics..."):
                    # Run diagnostics and show results
                    try:
                        # Check tested models file
                        test_models_result = self.check_tested_models()
                        
                        # Create success message with details
                        st.success(f"Found {test_models_result.get('model_count', 0)} tested models")
                        
                        # Display model types breakdown
                        if 'models_by_type' in test_models_result and test_models_result['models_by_type']:
                            model_breakdown = pd.DataFrame({
                                'Model Type': list(test_models_result['models_by_type'].keys()),
                                'Count': list(test_models_result['models_by_type'].values())
                            })
                            st.dataframe(model_breakdown, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Diagnostics error checking models: {str(e)}")
        
        with control_col3:
            if st.button("🧹 Clean Stale Locks", key="clean_locks", use_container_width=True):
                try:
                    from src.utils.threadsafe import cleanup_stale_locks
                    count = cleanup_stale_locks(force=True)
                    st.success(f"Cleaned {count} stale lock files")
                except Exception as e:
                    st.error(f"Error cleaning locks: {str(e)}")
        
        # In controls section, add a new column for error logging toggle and threshold configuration
        control_col4 = st.columns(1)
        with control_col4:
            error_logging = st.checkbox("Enable Error Logging", value=self.error_logging_enabled, key="error_logging_toggle")
            self.error_logging_enabled = error_logging
            if error_logging:
                new_threshold = st.slider("RMSE Threshold Start", min_value=1000, max_value=100000, value=self.rmse_threshold_start if self.rmse_threshold_start is not None else 100000, step=1000, key="rmse_threshold_slider")
                self.rmse_threshold_start = new_threshold
            else:
                self.rmse_threshold_start = None

        # Create tabs for different types of information
        tabs = st.tabs(["Activity Log", "Model Testing", "Resources", "Error Log"])
        
        # Tab 1: Activity Log
        with tabs[0]:
            st.subheader("Recent Activity")
            
            # Add filter options
            log_filter_col1, log_filter_col2 = st.columns(2)
            with log_filter_col1:
                log_types = ["All", "Error", "Warning", "Info"]
                selected_log_type = st.selectbox("Filter by type", log_types, key="log_type_filter")
            
            with log_filter_col2:
                max_entries = st.slider("Max entries", 5, 100, 20, key="log_max_entries")
            
            # Display activity log with more details and formatting
            if hasattr(self, 'dashboard_data') and 'recent_activity' in self.dashboard_data:
                activities = self.dashboard_data.get('recent_activity', [])
                
                # Apply filters
                if selected_log_type != "All":
                    activities = [a for a in activities if a.get('type', '').lower() == selected_log_type.lower()]
                
                activities = activities[:max_entries]  # Limit entries
                
                if activities:
                    for activity in activities:
                        timestamp = activity.get('timestamp', '')
                        if timestamp and 'T' in timestamp:
                            activity_time = timestamp.split('T')[1].split('.')[0]
                        else:
                            activity_time = timestamp
                            
                        message = activity.get('message', 'Unknown activity')
                        category = activity.get('category', '')
                        model_type = activity.get('model_type', '')
                        
                        # Format message with additional details if available
                        full_message = f"{activity_time}: {message}"
                        if model_type:
                            full_message += f" (Model: {model_type})"
                        if category and category not in full_message:
                            full_message += f" - {category}"
                        
                        # Display with appropriate styling
                        if activity.get('type') == 'error':
                            st.error(full_message)
                        elif activity.get('type') == 'warning':
                            st.warning(full_message)
                        else:
                            st.info(full_message)
                else:
                    st.info("No activity logs available")
            else:
                st.info("Watchdog has not recorded any activity yet")
        
        # Tab 2: Model Testing
        with tabs[1]:
            st.subheader("Model Performance")
            
            # Get performance history for models
            if hasattr(self, 'model_performance_history') and self.model_performance_history:
                # Let user select which models to view
                model_types = list(self.model_performance_history.keys())
                selected_models = st.multiselect(
                    "Select models to display", 
                    model_types,
                    default=model_types[:min(3, len(model_types))],  # Default to first 3
                    key="model_selection"
                )
                
                if selected_models:
                    # Create performance chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for model in selected_models:
                        history = self.model_performance_history.get(model, [])
                        if history:
                            # Plot the RMSE trend
                            ax.plot(history, label=f"{model}")
                            
                            # Add best RMSE as text
                            best_rmse = min(history) if history else float('inf')
                            if best_rmse != float('inf'):
                                ax.text(len(history)-1, best_rmse, f"{best_rmse:.4f}", 
                                        fontsize=9, va='bottom')
                    
                    ax.set_title('RMSE by Model Type and Trial')
                    ax.set_xlabel('Trial Sequence')
                    ax.set_ylabel('RMSE')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Show summary metrics table
                    metrics_data = []
                    for model in selected_models:
                        history = self.model_performance_history.get(model, [])
                        if history:
                            metrics_data.append({
                                'Model': model,
                                'Best RMSE': f"{min(history):.4f}",
                                'Last RMSE': f"{history[-1]:.4f}" if history else "N/A",
                                'Trials': len(history),
                                'Errors': self.model_error_count.get(model, 0)
                            })
                    
                    if metrics_data:
                        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                else:
                    st.info("Select models to display performance data")
                    
                # Display model trial counts
                if hasattr(self, 'model_trial_counts') and self.model_trial_counts:
                    st.subheader("Trial Distribution")
                    
                    trial_data = []
                    for model, count in self.model_trial_counts.items():
                        trial_data.append({'Model': model, 'Trials': count})
                        
                    trial_df = pd.DataFrame(trial_data)
                    
                    # Create a bar chart of trial counts
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(trial_df['Model'], trial_df['Trials'])
                    ax.set_title('Number of Trials by Model Type')
                    ax.set_xlabel('Model Type')
                    ax.set_ylabel('Number of Trials')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:,}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
            else:
                st.info("No model performance data available yet")
        
        # Tab 3: Resource Monitoring
        with tabs[2]:
            st.subheader("System Resources")
            
            # Display current resource stats
            resources = self.monitor_system_resources()
            if resources:
                # Create 3 columns for different resource types
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("CPU Usage", f"{resources.get('cpu_percent', 0):.1f}%")
                    
                with res_col2:
                    mem = resources.get('memory', {})
                    st.metric("Memory Usage", 
                             f"{mem.get('percent', 0):.1f}% ({mem.get('used_gb', 0):.1f}/{mem.get('total_gb', 0):.1f} GB)")
                    
                with res_col3:
                    disk = resources.get('disk', {})
                    st.metric("Disk Usage", 
                             f"{disk.get('percent', 0):.1f}% ({disk.get('used_gb', 0):.1f}/{disk.get('total_gb', 0):.1f} GB)")
                
                # Show GPU info if available
                gpu_info = resources.get('gpu', {})
                if gpu_info:
                    st.subheader("GPU Information")
                    for gpu_id, gpu_data in gpu_info.items():
                        gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
                        with gpu_col1:
                            st.metric(f"GPU {gpu_id.split('_')[1]}", gpu_data.get('name', 'Unknown'))
                        with gpu_col2:
                            st.metric("Memory Usage", f"{gpu_data.get('memory_percent', 0):.1f}%")
                        with gpu_col3:
                            st.metric("Temperature", f"{gpu_data.get('temperature', 0):.1f}°C")
                
                # Show historical resource usage if available
                if hasattr(self, 'resource_history') and len(self.resource_history) > 1:
                    st.subheader("Resource History")
                    
                    # Extract time series data
                    history = pd.DataFrame({
                        'timestamp': [r.get('timestamp', 0) for r in self.resource_history],
                        'cpu': [r.get('cpu_percent', 0) for r in self.resource_history],
                        'memory': [r.get('memory', {}).get('percent', 0) for r in self.resource_history],
                        'disk': [r.get('disk', {}).get('percent', 0) for r in self.resource_history]
                    })
                    
                    # Convert timestamps to datetime for better display
                    history['time'] = [datetime.fromtimestamp(ts) for ts in history['timestamp']]
                    
                    # Create resource usage chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history['time'], history['cpu'], label='CPU %')
                    ax.plot(history['time'], history['memory'], label='Memory %')
                    ax.plot(history['time'], history['disk'], label='Disk %')
                    
                    ax.set_title('System Resource Usage Over Time')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Utilization %')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Set y-axis limits for percentage
                    ax.set_ylim(0, max(100, history['cpu'].max() * 1.1))
                    
                    # Format x-axis with time
                    fig.autofmt_xdate()
                    
                    st.pyplot(fig)
                    
                # Show Python processes
                python_processes = resources.get('python_processes', [])
                if python_processes:
                    st.subheader("Python Processes")
                    
                    process_data = []
                    for proc in python_processes:
                        process_data.append({
                            'PID': proc.get('pid', 'N/A'),
                            'Memory %': f"{proc.get('memory_percent', 0):.1f}%",
                            'Command': proc.get('cmd', 'Unknown')
                        })
                    
                    if process_data:
                        st.dataframe(pd.DataFrame(process_data), use_container_width=True)
            else:
                st.info("Resource monitoring is disabled or no data available")
        
        # Tab 4: Error Log
        with tabs[3]:
            st.subheader("Error Log")
            
            # Check if error log file exists and display its contents
            if os.path.exists(self.error_log_file):
                try:
                    with open(self.error_log_file, 'r') as f:
                        log_contents = f.read()
                    
                    # Parse log into structured format
                    entries = log_contents.split("-" * 80)
                    parsed_entries = []
                    
                    for entry in entries:
                        if not entry.strip():
                            continue
                            
                        lines = entry.strip().split('\n')
                        if not lines:
                            continue
                            
                        # Parse timestamp and error type from first line
                        header = lines[0]
                        timestamp = header[1:header.find("]")] if header.startswith("[") else ""
                        error_type = header[header.find("]")+1:].strip() if "]" in header else header
                        
                        # Get other fields
                        model = ""
                        trial = ""
                        message = ""
                        details = ""
                        
                        in_details = False
                        for i, line in enumerate(lines[1:], 1):
                            if line.startswith("Model:"):
                                model = line[6:].strip()
                            elif line.startswith("Trial:"):
                                trial = line[6:].strip()
                            elif line.startswith("Details:"):
                                in_details = True
                            elif in_details:
                                details += line + "\n"
                            elif i <= 3 and line.strip():
                                message = line.strip()
                        
                        parsed_entries.append({
                            'timestamp': timestamp,
                            'error_type': error_type,
                            'model': model,
                            'trial': trial,
                            'message': message,
                            'details': details.strip()
                        })
                    
                    # Add filters for error log
                    error_filter_col1, error_filter_col2 = st.columns(2)
                    
                    with error_filter_col1:
                        error_types = ["All"] + sorted(list({e['error_type'] for e in parsed_entries if e['error_type']}))
                        selected_error_type = st.selectbox("Filter by error type", error_types, key="error_type_filter")
                    
                    with error_filter_col2:
                        model_types = ["All"] + sorted(list({e['model'] for e in parsed_entries if e['model']}))
                        selected_model_type = st.selectbox("Filter by model", model_types, key="error_model_filter")
                    
                    # Apply filters
                    filtered_entries = parsed_entries
                    if selected_error_type != "All":
                        filtered_entries = [e for e in filtered_entries if selected_error_type in e['error_type']]
                    if selected_model_type != "All":
                        filtered_entries = [e for e in filtered_entries if e['model'] == selected_model_type]
                    
                    # Limit to last 50 entries
                    filtered_entries = filtered_entries[-50:]
                    
                    # Display filtered errors
                    if filtered_entries:
                        for entry in filtered_entries:
                            # Create expandable error entry
                            with st.expander(f"{entry['timestamp']} - {entry['error_type']}"):
                                if entry['model']:
                                    st.write(f"**Model**: {entry['model']}")
                                if entry['trial']:
                                    st.write(f"**Trial**: {entry['trial']}")
                                if entry['message']:
                                    st.write(f"**Message**: {entry['message']}")
                                if entry['details']:
                                    st.code(entry['details'])
                    else:
                        st.info("No matching error log entries")
                except Exception as e:
                    st.error(f"Error reading log file: {str(e)}")
            else:
                st.info("No error log file found")

    def start(self):
        """Start the watchdog thread."""
        if self.running:
            logger.warning("Watchdog is already running")
            return False

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.running = True
        logger.info(f"Started watchdog for {self.ticker}/{self.timeframe}")
        return True

    def stop(self):
        """Stop the watchdog thread."""
        if not self.running:
            logger.warning("Watchdog is not running")
            return False

        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)  # Wait for thread to terminate
        self.running = False
        logger.info("Stopped watchdog")
        return True
        
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        logger.info("Starting monitoring loop")
        
        # Wait for initial startup grace period