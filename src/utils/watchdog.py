"""
ML Tuning Watchdog

This module provides automatic monitoring and recovery for the ML tuning process.
It specifically monitors tested_models.yaml as the primary indicator of progress,
and provides detailed error monitoring and logging for ensemble models.

Features:
- Primary monitoring of tested_models.yaml for trial progress
- Secondary monitoring of progress.yaml for overall state
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

    This class provides automated monitoring and recovery for machine learning model
    tuning processes. It tracks file modifications, process statuses, and resource usage
    to detect stalls or failures in the tuning pipeline. When issues are detected,
    it can automatically restart processes, create backups, and notify users.

    The watchdog integrates with the model tuning pipeline by monitoring key files:
    - tested_models.yaml: Contains completed model trials
    - progress.yaml: Tracks overall tuning progress
    - tuning_status.txt: Indicates if tuning is currently running

    This module is not typically user-controlled directly but runs as a background
    service during tuning operations.
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
        max_stale_time=3600,  # 60 minutes
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
    ):
        """
        Initialize the tuning watchdog with enhanced features.

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
                </div>
                
                <div class="status-card">
                    <h2>System Status</h2>
                    <p>Dashboard is being initialized. Please refresh in a few moments.</p>
                </div>
                
                <div class="footer">
                    <p>Last updated: {datetime.now().isoformat()}</p>
                    <p>Tuning Watchdog v1.0</p>
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

    def log_error(self, error_type, message, details=None):
        """
        Enhanced error logging with categorization and tracking.

        Args:
            error_type: Type/category of error
            message: Main error message
            details: Additional error details or traceback
        """
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
                f.write(f"{message}\n")
                if details:
                    f.write("Details:\n")
                    f.write(f"{details}\n")
                f.write("-" * 80 + "\n")

            # Add to dashboard data
            timestamp = datetime.now().isoformat()
            self.dashboard_data["recent_activity"].insert(
                0,
                {
                    "type": "error",
                    "category": error_type,
                    "message": message,
                    "timestamp": timestamp,
                },
            )

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

                    # Check for error patterns
                    self._check_model_for_errors(model, result["error_details"])

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

    def _check_model_for_errors(self, model, error_list):
        """
        Check a single model entry for error patterns.

        Args:
            model: The model data dictionary
            error_list: List to append errors to
        """
        try:
            model_type = model.get(
                "model_type", model.get("params", {}).get("model_type", "unknown")
            )

            # Initialize tracking for this model type if needed
            if model_type not in self.model_error_count:
                self.model_error_count[model_type] = 0

            if model_type not in self.model_performance_history:
                self.model_performance_history[model_type] = []

            # Get performance metrics
            rmse = model.get("rmse", model.get("value"))
            mape = model.get("mape")
            trial = model.get("trial_number", model.get("number"))

            # Track performance
            if rmse is not None:
                self.model_performance_history[model_type].append(
                    {
                        "rmse": rmse,
                        "mape": mape,
                        "trial": trial,
                        "timestamp": model.get("timestamp", time.time()),
                    }
                )

                # Keep only recent history
                if len(self.model_performance_history[model_type]) > 10:
                    self.model_performance_history[model_type] = (
                        self.model_performance_history[model_type][-10:]
                    )

            # Pattern 1: Extremely high RMSE (potential model failure)
            if rmse is not None and rmse > 1000:
                error_list.append(
                    {
                        "model_type": model_type,
                        "error_type": "high_rmse",
                        "value": rmse,
                        "trial": trial,
                    }
                )
                self.model_error_count[model_type] += 1

            # Pattern 2: NaN or None values in metrics
            if (
                rmse is None
                or (isinstance(rmse, float) and (math.isnan(rmse) or math.isinf(rmse)))
            ) or (
                mape is not None
                and isinstance(mape, float)
                and (math.isnan(mape) or math.isinf(mape))
            ):
                error_list.append(
                    {
                        "model_type": model_type,
                        "error_type": "invalid_metrics",
                        "rmse": rmse,
                        "mape": mape,
                        "trial": trial,
                    }
                )
                self.model_error_count[model_type] += 1

            # Pattern 3: State is not COMPLETE
            state = model.get("state")
            if state and state != "COMPLETE" and state != "PRUNED":
                error_list.append(
                    {
                        "model_type": model_type,
                        "error_type": "incomplete_state",
                        "state": state,
                        "trial": trial,
                    }
                )
                self.model_error_count[model_type] += 1

        except Exception as e:
            # Don't let a single model error checking fail the whole process
            logger.warning(
                f"Error checking model {model.get('trial_number', 'unknown')}: {e}"
            )

    def check_progress_file(self):
        """
        Check progress.yaml for overall tuning status.

        Returns:
            dict: Status information about overall tuning progress
        """
        result = {
            "is_updating": False,
            "last_update_time": 0,
            "current_trial": None,
            "total_trials": None,
            "current_cycle": None,
            "errors_detected": False,
        }

        try:
            if not os.path.exists(self.progress_file):
                return result

            # Check file modification time
            result["last_update_time"] = os.path.getmtime(self.progress_file)

            # Determine if the file has been updated since our last check
            result["is_updating"] = (
                result["last_update_time"] > self.last_progress_update_time
            )

            # Read and analyze the progress file
            with open(self.progress_file, "r") as f:
                progress_data = yaml.safe_load(f) or {}

            if not isinstance(progress_data, dict):
                self.log_error(
                    "Format Error",
                    f"progress.yaml is not a dictionary: {type(progress_data)}",
                )
                return result

            # Extract key information
            result["current_trial"] = progress_data.get("current_trial")
            result["total_trials"] = progress_data.get("total_trials")
            result["current_cycle"] = progress_data.get("cycle")
            result["ticker"] = progress_data.get("ticker")
            result["timeframe"] = progress_data.get("timeframe")

            # Check for valid values
            if (
                result["current_trial"] is not None
                and result["total_trials"] is not None
            ):
                # Percent complete
                if result["total_trials"] > 0:
                    result["percent_complete"] = (
                        result["current_trial"] / result["total_trials"]
                    ) * 100

                # Check for errors or inconsistencies
                if result["current_trial"] > result["total_trials"]:
                    result["errors_detected"] = True
                    result["error_details"] = "current_trial exceeds total_trials"

            # Update our state
            if result["is_updating"]:
                self.last_progress_update_time = result["last_update_time"]

            return result

        except Exception as e:
            error_msg = f"Error checking progress file: {e}"
            self.log_error("Monitoring Error", error_msg, traceback.format_exc())
            result["error"] = str(e)
            return result

    def is_tuning_running(self):
        """
        Check if tuning is currently running based on status file.

        Returns:
            bool: True if tuning is running, False otherwise
        """
        try:
            if not os.path.exists(self.status_file):
                return False

            # Parse status file
            status_info = {}
            with open(self.status_file, "r") as f:
                for line in f.readlines():
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        status_info[key.strip()] = value.strip()

            # Check if status shows running
            is_running = str(status_info.get("is_running", "False")).lower() == "true"

            # If it claims to be running, check how old the file is
            if is_running:
                mtime = os.path.getmtime(self.status_file)
                if time.time() - mtime > self.max_stale_time * 2:
                    logger.warning(
                        f"Status file claims tuning is running but file is {time.time() - mtime:.1f}s old"
                    )
                    return False

            return is_running
        except Exception as e:
            self.log_error("Status Check Error", f"Error checking tuning status: {e}")
            return False

    def is_making_progress(self):
        """
        Check if tuning is making progress by analyzing both tested_models.yaml and progress.yaml.

        Returns:
            tuple: (is_making_progress, status_dict)
        """
        try:
            # Get detailed status from both files
            models_status = self.check_tested_models()
            progress_status = self.check_progress_file()

            # Calculate time since last update for each file
            time_since_models_update = (
                time.time() - models_status["last_update_time"]
                if models_status["last_update_time"] > 0
                else float("inf")
            )
            time_since_progress_update = (
                time.time() - progress_status["last_update_time"]
                if progress_status["last_update_time"] > 0
                else float("inf")
            )

            # Log model-specific errors if detected
            if models_status["errors_detected"]:
                for error in models_status["error_details"]:
                    self.log_error(
                        f"Model Error ({error.get('model_type', 'unknown')})",
                        f"Error type: {error.get('error_type')} in trial {error.get('trial')}",
                        str(error),
                    )

            # Primary check: Are we seeing new entries in tested_models.yaml?
            models_active = (
                models_status["is_updating"]
                or time_since_models_update < self.max_stale_time
            )

            # Secondary check: Is progress.yaml being updated?
            progress_active = (
                progress_status["is_updating"]
                or time_since_progress_update < self.max_stale_time * 2
            )  # More lenient threshold

            # Combined status
            is_progressing = models_active  # Primary indicator is tested_models.yaml

            # Create status report
            status = {
                "is_progressing": is_progressing,
                "models_active": models_active,
                "progress_active": progress_active,
                "time_since_models_update": time_since_models_update,
                "time_since_progress_update": time_since_progress_update,
                "models_status": models_status,
                "progress_status": progress_status,
                "model_type_counts": models_status.get("models_by_type", {}),
                "model_errors": self.model_error_count,
            }

            # Log potential issues
            if not is_progressing:
                self.log_error(
                    "Progress Stalled",
                    f"No updates in tested_models.yaml for {time_since_models_update:.1f}s and "
                    + f"no updates in progress.yaml for {time_since_progress_update:.1f}s",
                )

            return is_progressing, status

        except Exception as e:
            self.log_error(
                "Progress Check Error",
                f"Error checking progress: {e}",
                traceback.format_exc(),
            )
            return False, {"error": str(e)}

    def cleanup_stale_locks(self):
        """
        Clean up stale lock files in the data directory.

        Identifies and removes lock files that have been abandoned by crashed processes
        or incomplete operations. This helps prevent resource contention and deadlocks.

        Returns:
            int: Number of lock files removed
        """
        try:
            logger.info("Cleaning up stale lock files...")
            lock_files_removed = 0

            # Find all lock files
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".lock"):
                        lock_path = os.path.join(root, file)
                        try:
                            lock_age = time.time() - os.path.getmtime(lock_path)
                            # Consider locks stale after 5 minutes
                            if lock_age > 300:
                                logger.debug(f"Removing stale lock file: {lock_path}")
                                os.remove(lock_path)
                                lock_files_removed += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to check or remove lock file {lock_path}: {e}"
                            )

            logger.info(f"Removed {lock_files_removed} stale lock files")
            return lock_files_removed
        except Exception as e:
            self.log_error(
                "Lock Cleanup Error",
                f"Error during lock file cleanup: {e}",
                traceback.format_exc(),
            )
            return 0

    def reset_tuning_status(self, force=False):
        """
        Reset the tuning status file to indicate tuning is not running.

        Updates the tuning_status.txt file to indicate that tuning is no longer running.
        This is useful for recovering from crashed processes or for manual intervention.

        Args:
            force: If True, force reset even if tuning appears to be actively running
        """
        try:
            if os.path.exists(self.status_file):
                # Check if we should force reset
                if not force:
                    # Only reset if not actively running or if file is too old
                    file_age = time.time() - os.path.getmtime(self.status_file)
                    status_info = self.read_tuning_status()

                    is_stale = file_age > self.max_stale_time
                    is_running = status_info.get("is_running", False)

                    if is_running and not is_stale:
                        logger.warning(
                            "Not resetting tuning status because process appears to be actively running"
                        )
                        return

                # Create backup of existing file
                if self.create_backups:
                    backup_path = f"{self.status_file}.{int(time.time())}.bak"
                    try:
                        shutil.copy2(self.status_file, backup_path)
                        logger.info(f"Created backup of status file at {backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create backup of status file: {e}")

                # Properly update status file to show not running
                with open(self.status_file, "w") as f:
                    f.write("is_running: False\n")
                    f.write("reset_by: watchdog\n")
                    f.write(f"reset_time: {time.time()}\n")
                    f.write(f"timestamp: {datetime.now().isoformat()}\n")

                logger.info("Reset tuning status to 'not running'")
        except Exception as e:
            self.log_error("Status Reset Error", f"Error resetting tuning status: {e}")

    def read_tuning_status(self):
        """
        Read and parse the tuning status file.

        Returns:
            dict: Status information parsed from the status file, or empty dict if file doesn't exist
        """
        try:
            if not os.path.exists(self.status_file):
                return {}

            status_info = {}
            with open(self.status_file, "r") as f:
                for line in f.readlines():
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        status_info[key.strip()] = value.strip()

            # Convert is_running to boolean
            if "is_running" in status_info:
                status_info["is_running"] = status_info["is_running"].lower() == "true"

            return status_info
        except Exception as e:
            self.log_error("Status Read Error", f"Error reading tuning status: {e}")
            return {}

    def analyze_trial_history(self):
        """
        Analyze the trial history to detect patterns in model performance and failures.

        Returns:
            dict: Analysis of trial history data
        """
        try:
            # Get the complete tested_models data
            if not os.path.exists(self.tested_models_file):
                return {}

            with open(self.tested_models_file, "r") as f:
                models_data = yaml.safe_load(f) or []

            if not isinstance(models_data, list):
                return {}

            # Count trials by model type
            trial_counts = Counter()
            recent_trials = Counter()  # Last 20 trials

            # Performance metrics by model type
            performance = defaultdict(list)
            recent_performance = defaultdict(list)  # Last 5 trials per model

            # Hyperparameter tracking
            hyperparams = defaultdict(lambda: defaultdict(list))

            # Process all models, focusing on the most recent ones
            recent_slice = models_data[-100:] if len(models_data) > 100 else models_data
            very_recent_slice = (
                models_data[-20:] if len(models_data) > 20 else models_data
            )

            for model in recent_slice:
                if not isinstance(model, dict):
                    continue

                # Extract key information
                model_type = model.get(
                    "model_type", model.get("params", {}).get("model_type", "unknown")
                )
                trial_num = model.get("trial_number", model.get("number"))
                rmse = model.get("rmse", model.get("value"))
                mape = model.get("mape")
                state = model.get("state", "UNKNOWN")

                # Count by model type
                trial_counts[model_type] += 1

                # Very recent trials
                if model in very_recent_slice:
                    recent_trials[model_type] += 1

                # Track performance metrics
                if rmse is not None and not (
                    isinstance(rmse, float) and (math.isnan(rmse) or math.isinf(rmse))
                ):
                    performance[model_type].append(
                        {"trial": trial_num, "rmse": rmse, "mape": mape, "state": state}
                    )

                    # Keep only the most recent performance data
                    performance[model_type] = sorted(
                        performance[model_type], key=lambda x: x.get("trial", 0)
                    )[-10:]

                    # Track hyperparameters for successful trials
                    if state == "COMPLETE":
                        params = model.get("params", {})
                        for param_name, param_value in params.items():
                            if isinstance(param_value, (int, float, str, bool)):
                                hyperparams[model_type][param_name].append(
                                    {"value": param_value, "rmse": rmse}
                                )

            # Calculate performance statistics
            performance_stats = {}
            for model_type, trials in performance.items():
                if not trials:
                    continue

                rmse_values = [
                    t.get("rmse") for t in trials if t.get("rmse") is not None
                ]
                mape_values = [
                    t.get("mape") for t in trials if t.get("mape") is not None
                ]

                if rmse_values:
                    performance_stats[model_type] = {
                        "trial_count": len(trials),
                        "mean_rmse": sum(rmse_values) / len(rmse_values),
                        "min_rmse": min(rmse_values),
                        "max_rmse": max(rmse_values),
                        "mean_mape": (
                            sum(mape_values) / len(mape_values) if mape_values else None
                        ),
                        "trend": (
                            "improving"
                            if len(rmse_values) > 1 and rmse_values[-1] < rmse_values[0]
                            else "steady"
                        ),
                    }

            # Find best hyperparameters for each model type
            best_hyperparams = {}
            for model_type, params in hyperparams.items():
                best_hyperparams[model_type] = {}

                for param_name, values in params.items():
                    # Sort values by RMSE
                    sorted_values = sorted(
                        values, key=lambda x: x.get("rmse", float("inf"))
                    )

                    if sorted_values:
                        # Get the value from the best trial
                        best_hyperparams[model_type][param_name] = sorted_values[0][
                            "value"
                        ]

            # Model type balance analysis
            total_trials = sum(trial_counts.values())
            balance = {
                model_type: count / total_trials if total_trials > 0 else 0
                for model_type, count in trial_counts.items()
            }

            # Detect underrepresented models (less than 5% of trials)
            underrepresented = [
                model_type
                for model_type, ratio in balance.items()
                if ratio < 0.05 and trial_counts[model_type] < 5
            ]

            # Recent activity analysis
            recent_activity = []
            for i, model in enumerate(reversed(very_recent_slice)):
                if i >= 10:  # Only show most recent 10
                    break

                if not isinstance(model, dict):
                    continue

                model_type = model.get(
                    "model_type", model.get("params", {}).get("model_type", "unknown")
                )
                trial = model.get("trial_number", model.get("number"))
                rmse = model.get("rmse", model.get("value"))
                timestamp = model.get("timestamp")

                if timestamp and isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        friendly_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        friendly_time = "Unknown"
                else:
                    friendly_time = "Unknown"

                recent_activity.append(
                    {
                        "model_type": model_type,
                        "trial": trial,
                        "rmse": rmse,
                        "time": friendly_time,
                    }
                )

            # Compile results
            analysis = {
                "trial_counts": dict(trial_counts),
                "recent_trials": dict(recent_trials),
                "performance_stats": performance_stats,
                "best_hyperparams": best_hyperparams,
                "balance": balance,
                "underrepresented": underrepresented,
                "recent_activity": recent_activity,
            }

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing trial history: {e}")
            return {}

    def check_per_model_health(self, models_status):
        """
        Analyze the health of individual models in the ensemble.

        Args:
            models_status: Status information from check_tested_models

        Returns:
            dict: Health analysis per model type
        """
        model_health = {}

        try:
            model_types = models_status.get("models_by_type", {})

            for model_type, count in model_types.items():
                # Skip unknown
                if model_type == "unknown":
                    continue

                # Get error count for this model
                error_count = self.model_error_count.get(model_type, 0)

                # Get performance history
                history = self.model_performance_history.get(model_type, [])

                model_health[model_type] = {
                    "count": count,
                    "error_count": error_count,
                    "error_rate": error_count / max(1, count),
                    "performance_history_count": len(history),
                    "status": "healthy",
                }

                # Analyze performance trend if we have enough data
                if len(history) >= 3:
                    recent_rmse = [
                        entry.get("rmse", float("inf")) for entry in history[-3:]
                    ]
                    if all(rmse > 1000 for rmse in recent_rmse):
                        model_health[model_type]["status"] = "failing"

                # Flag high error rate
                if model_health[model_type]["error_rate"] > 0.3:  # More than 30% errors
                    model_health[model_type]["status"] = "unstable"

            return model_health

        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {"error": str(e)}

    def update_dashboard(self):
        """Update the status dashboard HTML file with the latest information."""
        try:
            # Check if it's time to update the dashboard
            current_time = time.time()
            if (
                current_time - self.last_dashboard_update
                < self.dashboard_update_interval
            ):
                return

            self.last_dashboard_update = current_time

            # Get required status information
            is_running = self.is_tuning_running()
            making_progress, progress_details = self.is_making_progress()
            resource_data = self.monitor_system_resources()
            trial_analysis = self.analyze_trial_history()

            # Determine overall system status
            if is_running and making_progress:
                system_status = "healthy"
                status_text = "Healthy"
            elif is_running and not making_progress:
                system_status = "warning"
                status_text = "Stalled"
            elif not is_running and self.auto_restart:
                system_status = "warning"
                status_text = "Not Running (Auto-Restart Enabled)"
            else:
                system_status = "error"
                status_text = "Not Running"

            # Time-since calculations
            time_since_models_update = "N/A"
            if self.last_models_update_time > 0:
                seconds = int(current_time - self.last_models_update_time)
                if seconds < 60:
                    time_since_models_update = f"{seconds} seconds ago"
                elif seconds < 3600:
                    time_since_models_update = f"{seconds // 60} minutes ago"
                else:
                    time_since_models_update = (
                        f"{seconds // 3600} hours, {(seconds % 3600) // 60} minutes ago"
                    )

            # Create the dashboard HTML
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
                    .progress-bar {{
                        height: 20px;
                        background-color: #ecf0f1;
                        border-radius: 10px;
                        overflow: hidden;
                    }}
                    .progress-bar-fill {{
                        height: 100%;
                        background-color: #3498db;
                        border-radius: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Tuning Watchdog Dashboard</h1>
                    <p>Monitoring {self.ticker}/{self.timeframe}</p>
                </div>
                
                <div class="status-card">
                    <h2>System Status</h2>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="status-{system_status}">{status_text}</span>
                            <p>Last model update: {time_since_models_update}</p>
                            <p>Auto-restart: {'Enabled' if self.auto_restart else 'Disabled'}</p>
                        </div>
                        <div>
                            <button onclick="window.location.reload()">Refresh Now</button>
                        </div>
                    </div>
                </div>
                
                <div class="grid-container">
                    <div class="status-card">
                        <h2>Model Status</h2>
                        <table>
                            <tr>
                                <th>Model Type</th>
                                <th>Trial Count</th>
                                <th>Recent Trials</th>
                                <th>Status</th>
                            </tr>
            """

            # Add model status rows
            model_types = sorted(trial_analysis.get("trial_counts", {}).keys())
            for model_type in model_types:
                count = trial_analysis.get("trial_counts", {}).get(model_type, 0)
                recent = trial_analysis.get("recent_trials", {}).get(model_type, 0)

                # Determine model status
                model_health = "unknown"
                if model_type in trial_analysis.get("underrepresented", []):
                    model_health = "warning"
                elif (
                    model_type in self.model_error_count
                    and self.model_error_count[model_type] > 5
                ):
                    model_health = "error"
                else:
                    model_health = "healthy"

                dashboard_html += f"""
                            <tr>
                                <td>{model_type}</td>
                                <td>{count}</td>
                                <td>{recent}</td>
                                <td><span class="status-{model_health}">{model_health.capitalize()}</span></td>
                            </tr>
                """

            dashboard_html += """
                        </table>
                    </div>
                    
                    <div class="status-card">
                        <h2>System Resources</h2>
            """

            # Add resource information if available
            if resource_data:
                cpu = resource_data.get("cpu_percent", 0)
                memory = resource_data.get("memory", {}).get("percent", 0)
                disk = resource_data.get("disk", {}).get("percent", 0)

                dashboard_html += f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>CPU Usage</span>
                                <span>{cpu:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill" style="width: {cpu}%;"></div>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Memory Usage</span>
                                <span>{memory:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill" style="width: {memory}%;"></div>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Disk Usage</span>
                                <span>{disk:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill" style="width: {disk}%;"></div>
                            </div>
                        </div>
                """

                # Add GPU information if available
                gpu_info = resource_data.get("gpu", {})
                for gpu_id, gpu in gpu_info.items():
                    gpu_usage = gpu.get("gpu_percent", 0)
                    gpu_memory = gpu.get("memory_percent", 0)
                    gpu_name = gpu.get("name", "Unknown GPU")

                    dashboard_html += f"""
                        <div style="margin-bottom: 15px;">
                            <h4>{gpu_name}</h4>
                            <div style="display: flex; justify-content: space-between;">
                                <span>GPU Usage</span>
                                <span>{gpu_usage:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill" style="width: {gpu_usage}%;"></div>
                            </div>
                            
                            <div style="display: flex; justify-content: space-between;">
                                <span>GPU Memory</span>
                                <span>{gpu_memory:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill" style="width: {gpu_memory}%;"></div>
                            </div>
                        </div>
                    """
            else:
                dashboard_html += "<p>Resource monitoring not available</p>"

            dashboard_html += """
                    </div>
                </div>
                
                <div class="status-card">
                    <h2>Recent Trial Activity</h2>
                    <table>
                        <tr>
                            <th>Trial</th>
                            <th>Model Type</th>
                            <th>RMSE</th>
                            <th>Time</th>
                        </tr>
            """

            # Add recent activity
            for activity in trial_analysis.get("recent_activity", []):
                rmse = activity.get("rmse", "N/A")
                if isinstance(rmse, (int, float)):
                    rmse = f"{rmse:.4f}"

                dashboard_html += f"""
                        <tr>
                            <td>{activity.get('trial', 'N/A')}</td>
                            <td>{activity.get('model_type', 'unknown')}</td>
                            <td>{rmse}</td>
                            <td>{activity.get('time', 'Unknown')}</td>
                        </tr>
                """

            dashboard_html += """
                    </table>
                </div>
                
                <div class="status-card">
                    <h2>Error Summary</h2>
            """

            # Add error summary
            if self.error_count_by_category:
                dashboard_html += """
                    <table>
                        <tr>
                            <th>Error Category</th>
                            <th>Count</th>
                            <th>Last Occurrence</th>
                        </tr>
                """

                for category, count in self.error_count_by_category.most_common():
                    dashboard_html += f"""
                        <tr>
                            <td>{category}</td>
                            <td>{count}</td>
                            <td>Recently</td>
                        </tr>
                    """

                dashboard_html += "</table>"
            else:
                dashboard_html += "<p>No errors detected</p>"

            dashboard_html += (
                """
                </div>
                
                <div class="footer">
                    <p>Last updated: """
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
                    <p>Tuning Watchdog v1.0</p>
                </div>
            </body>
            </html>
            """
            )

            # Write to file
            with open(self.dashboard_file, "w") as f:
                f.write(dashboard_html)

            logger.debug(f"Updated dashboard at {self.dashboard_file}")

        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")

    def _check_database_integrity(self):
        """
        Check SQLite database integrity.

        Returns:
            list: List of detected database problems
        """
        problems = []

        try:
            import sqlite3

            # Find all SQLite databases
            db_dir = os.path.join(self.data_dir, "DB")
            if not os.path.exists(db_dir):
                return []

            for file in os.listdir(db_dir):
                if file.endswith(".db"):
                    db_path = os.path.join(db_dir, file)

                    try:
                        # Try to connect and run integrity check
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check")
                        result = cursor.fetchone()

                        if result[0] != "ok":
                            problems.append(
                                f"Database integrity check failed for {file}: {result[0]}"
                            )

                        conn.close()
                    except sqlite3.Error as e:
                        problems.append(f"Database error when checking {file}: {e}")

            return problems
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return [f"Could not check database integrity: {e}"]

    def _repair_databases(self):
        """Try to repair corrupt databases."""
        try:
            import sqlite3

            # Find all SQLite databases
            db_dir = os.path.join(self.data_dir, "DB")
            if not os.path.exists(db_dir):
                return

            for file in os.listdir(db_dir):
                if file.endswith(".db"):
                    db_path = os.path.join(db_dir, file)

                    try:
                        # Try to connect and run integrity check
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check")
                        result = cursor.fetchone()

                        if result[0] != "ok":
                            logger.warning(
                                f"Attempting to repair corrupt database: {file}"
                            )

                            # Create backup
                            if self.create_backups:
                                backup_path = os.path.join(
                                    self.backup_dir,
                                    f"{file}.corrupt.{int(time.time())}",
                                )
                                shutil.copy2(db_path, backup_path)
                                logger.info(
                                    f"Created backup of corrupt database at {backup_path}"
                                )

                            # Try to repair with recovery mode
                            try:
                                conn.close()
                                time.sleep(1)  # Wait for connection to close completely
                                conn = sqlite3.connect(
                                    f"file:{db_path}?mode=rw", uri=True
                                )
                                conn.execute("VACUUM;")
                                conn.close()
                                logger.info(f"Attempted repair of {file}")
                            except Exception as e:
                                logger.error(f"Failed to repair {file}: {e}")

                                # If repair failed, try to create a new database file
                                new_db_path = f"{db_path}.new"
                                try:
                                    with sqlite3.connect(new_db_path) as new_conn:
                                        new_conn.execute(
                                            "CREATE TABLE IF NOT EXISTS studies (study_id INTEGER PRIMARY KEY)"
                                        )
                                    logger.info(
                                        f"Created replacement database: {new_db_path}"
                                    )
                                except Exception as e2:
                                    logger.error(
                                        f"Failed to create replacement database: {e2}"
                                    )
                        else:
                            conn.close()
                    except sqlite3.Error as e:
                        logger.error(f"Database error when checking {file}: {e}")

        except Exception as e:
            logger.error(f"Error repairing databases: {e}")

    def _try_restart_methods(self):
        """
        Try different methods to restart tuning.

        Returns:
            bool: True if any restart method succeeded, False otherwise
        """
        # Method 1: Import and call directly if we're within the same Python environment
        try:
            logger.info("Attempting direct module import for restart...")
            sys.path.append(os.path.abspath(os.path.dirname(self.data_dir)))

            # Try to import the meta_tuning module
            from src.tuning.meta_tuning import start_tuning_process

            # Get latest ticker and timeframe from progress file if available
            ticker = self.ticker
            timeframe = self.timeframe

            try:
                with open(self.progress_file, "r") as f:
                    progress_data = yaml.safe_load(f) or {}

                if progress_data.get("ticker"):
                    ticker = progress_data["ticker"]
                if progress_data.get("timeframe"):
                    timeframe = progress_data["timeframe"]

                logger.info(
                    f"Using ticker {ticker} and timeframe {timeframe} from progress file"
                )
            except Exception:
                logger.info(f"Using default ticker {ticker} and timeframe {timeframe}")

            # Call the start function directly
            start_tuning_process(ticker, timeframe)

            logger.info("Successfully restarted tuning via direct import")
            return True
        except ImportError:
            logger.warning("Could not import start_tuning_process directly")
        except Exception as e:
            self.log_error(
                "Direct Restart Error",
                f"Error during direct tuning restart: {e}",
                traceback.format_exc(),
            )

        # Method 2: Look for a script that can start tuning
        try:
            # Look for the main script in standard locations
            potential_scripts = [
                "run_tuning.py",
                "start_tuning.py",
                "scripts/run_tuning.py",
                "src/tuning/run_tuning.py",
                "../scripts/run_tuning.py",
            ]

            script_path = None
            for path in potential_scripts:
                if os.path.exists(path):
                    script_path = path
                    break

            if script_path:
                logger.info(f"Found script to restart tuning: {script_path}")

                # Run the script as a subprocess
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        script_path,
                        "--ticker",
                        self.ticker,
                        "--timeframe",
                        self.timeframe,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Wait a bit to see if it starts successfully
                time.sleep(self.startup_grace_period)

                # Check if process is still running
                if proc.poll() is None:
                    logger.info("Tuning restart via script appears successful")
                    return True
                else:
                    stdout, stderr = proc.communicate()
                    self.log_error(
                        "Script Restart Error",
                        f"Script exited with code {proc.returncode}",
                        f"STDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}",
                    )
        except Exception as e:
            self.log_error(
                "Script Restart Error",
                f"Error restarting via script: {e}",
                traceback.format_exc(),
            )

        # Method 3: Last resort - try to trigger via Streamlit API call
        try:
            import requests

            # Try to find the Streamlit server if it's running
            streamlit_port = self._find_streamlit_port()

            if streamlit_port:
                logger.info(f"Found Streamlit running on port {streamlit_port}")
                # This would need custom implementation in your Streamlit app
                # to accept API calls for tuning
        except ImportError:
            logger.warning(
                "Requests module not available, cannot try Streamlit API restart"
            )
        except Exception as e:
            self.log_error(
                "Streamlit Restart Error",
                f"Error during Streamlit API restart attempt: {e}",
            )

        return False

    def _find_streamlit_port(self):
        """
        Try to find a running Streamlit server's port.

        Returns:
            int: Port number if found, None otherwise
        """
        if psutil is None:
            return None

        try:
            # Look for Python processes
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if "python" in proc.info["name"].lower():
                    cmdline = proc.info["cmdline"]
                    if cmdline and "streamlit" in " ".join(cmdline):
                        # Look for the port in the command line
                        for i, arg in enumerate(cmdline):
                            if arg == "--server.port" and i + 1 < len(cmdline):
                                try:
                                    return int(cmdline[i + 1])
                                except ValueError:
                                    pass
                        # Default port if not specified
                        return 8501
        except Exception:
            pass
        return None

    def restart_tuning(self):
        """
        Enhanced restart_tuning with additional safety checks.

        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            # Check if we've hit the restart limit
            if (
                time.time() - self.last_restart_time < self.restart_cooldown
                and self.consecutive_restarts >= self.max_consecutive_restarts
            ):
                logger.warning(
                    f"Hit maximum consecutive restarts ({self.max_consecutive_restarts}). Cooling down."
                )
                return False

            # Increment restart counter
            self.consecutive_restarts += 1
            self.last_restart_time = time.time()

            logger.info(
                f"Attempting to restart tuning (attempt {self.consecutive_restarts})"
            )

            # Step 1: Analyze what might be wrong
            problems = []

            # Check for stale lock files
            lock_file_count = 0
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".lock"):
                        lock_file_count += 1

            if lock_file_count > 0:
                problems.append(f"Found {lock_file_count} stale lock files")

            # Check for corrupted databases
            db_problems = self._check_database_integrity()
            if db_problems:
                problems.extend(db_problems)

            # Log detected problems
            if problems:
                self.log_error(
                    "Restart Analysis",
                    "Detected potential issues before restart:",
                    "\n".join(problems),
                )

            # Step 2: Create backups
            if self.create_backups:
                self.backup_critical_files()

            # Step 3: Reset status to ensure clean state
            self.reset_tuning_status()

            # Step 4: Clean up any lock files
            self.cleanup_stale_locks()

            # Step 5: Try to repair databases if needed
            if any("database" in p.lower() for p in problems):
                self._repair_databases()

            # Step 6: Restart tuning
            restart_success = self._try_restart_methods()

            # Reset consecutive restarts counter if successful
            if restart_success:
                self.consecutive_restarts = 0
                return True

            self.log_error("Restart Failure", "All restart methods failed")
            return False

        except Exception as e:
            self.log_error(
                "Restart Error",
                f"Error during tuning restart: {e}",
                traceback.format_exc(),
            )
            return False

    def _check_for_future_timestamp(self, status_info):
        """Check if status file contains timestamps from the future, indicating clock issues."""
        current_time = time.time()
        future_threshold = current_time + 3600  # 1 hour into the future

        future_timestamps = []

        # Check timestamp fields
        for key in ["timestamp", "start_time", "stop_time"]:
            if key in status_info:
                try:
                    # Handle both string ISO timestamps and float timestamps
                    if isinstance(status_info[key], str):
                        if status_info[key].replace(".", "").isdigit():
                            # Numeric string
                            ts_value = float(status_info[key])
                        else:
                            # Try ISO format parsing
                            try:
                                ts_value = datetime.fromisoformat(
                                    status_info[key].replace("Z", "+00:00")
                                ).timestamp()
                            except ValueError:
                                # Couldn't parse timestamp
                                continue
                    else:
                        ts_value = float(status_info[key])

                    if ts_value > future_threshold:
                        future_timestamps.append(
                            (
                                key,
                                ts_value,
                                datetime.fromtimestamp(ts_value).isoformat(),
                            )
                        )
                except Exception as e:
                    self.log_error(
                        "Timestamp Error", f"Error parsing timestamp '{key}': {e}"
                    )

        if future_timestamps:
            error_msg = "Detected future timestamps in tuning status file:"
            for field, ts, iso in future_timestamps:
                error_msg += f"\n - {field}: {iso} (from now: {ts - current_time:.1f}s)"

            self.log_error(
                "Future Timestamp Error",
                error_msg,
                "This may indicate a system clock issue or corrupted status file.",
            )
            return True

        return False

    def read_tuning_status(self):
        """
        Read and validate the tuning status file.

        Returns:
            Dict with status information
        """
        status = {"is_running": False}

        if not os.path.exists(self.status_file):
            return status

        try:
            with open(self.status_file, "r") as f:
                lines = f.readlines()

            # Parse simple key:value format
            for line in lines:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert "true"/"false" strings to boolean
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

                    status[key] = value

            # Check for future timestamps which indicate a problem
            self._check_for_future_timestamp(status)

            print(f"Read tuning status: {status}")
        except Exception as e:
            self.log_error("Status Read Error", f"Error reading tuning status: {e}")

        return status

    def diagnose_sqlite_initialization(self):
        """
        Perform in-depth diagnosis of SQLite initialization issues.

        Returns:
            Dict containing diagnosis results
        """
        results = {
            "can_import_sqlite": False,
            "can_create_test_db": False,
            "can_create_optuna_db": False,
            "sqlite_version": None,
            "detailed_errors": [],
            "recommendations": [],
        }

        # Test 1: Can import sqlite3
        try:
            import sqlite3

            results["can_import_sqlite"] = True
            results["sqlite_version"] = sqlite3.sqlite_version
        except ImportError as e:
            results["detailed_errors"].append(f"Cannot import sqlite3: {e}")
            results["recommendations"].append(
                "Install sqlite3 with pip install pysqlite3"
            )
            return results

        # Test 2: Can create and use SQLite database in DB_DIR
        import sqlite3

        try:
            db_dir = os.path.join(self.data_dir, "DB")
            os.makedirs(db_dir, exist_ok=True)

            test_db_path = os.path.join(db_dir, "watchdog_test.db")
            conn = sqlite3.connect(test_db_path)

            # Test creating a table and inserting data
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS watchdog_test (id INTEGER PRIMARY KEY, test_value TEXT)"
            )
            cursor.execute(
                "INSERT INTO watchdog_test (test_value) VALUES (?)", ("test_record",)
            )
            conn.commit()

            # Verify data was written
            cursor.execute("SELECT test_value FROM watchdog_test LIMIT 1")
            result = cursor.fetchone()

            if result and result[0] == "test_record":
                results["can_create_test_db"] = True
            else:
                results["detailed_errors"].append(
                    "Data verification failed in test database"
                )

            # Clean up
            conn.close()
            try:
                os.remove(test_db_path)
            except:
                pass

        except Exception as e:
            results["detailed_errors"].append(f"Error creating test database: {e}")

        # Test 3: Can create an Optuna study database
        try:
            import optuna

            study_name = f"watchdog_test_study_{int(time.time())}"
            db_path = os.path.join(db_dir, f"{study_name}.db")
            storage_url = f"sqlite:///{db_path}"

            # Try to create a study
            study = optuna.create_study(
                study_name=study_name, storage=storage_url, direction="minimize"
            )

            # Add a simple trial
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x**2

            study.optimize(objective, n_trials=1)

            # Check if study was created
            if os.path.exists(db_path) and study.trials:
                results["can_create_optuna_db"] = True

            # Clean up
            try:
                os.remove(db_path)
            except:
                pass

        except Exception as e:
            results["detailed_errors"].append(f"Error creating Optuna study: {e}")

        # Generate recommendations
        if not results["can_create_test_db"]:
            results["recommendations"].append("Check permissions on the DB directory")

        if not results["can_create_optuna_db"]:
            results["recommendations"].append(
                "Check Optuna installation: pip install optuna"
            )
            if "OperationalError" in str(results["detailed_errors"]):
                results["recommendations"].append(
                    "SQLite database may be locked or corrupted"
                )

        return results

    def analyze_and_fix_stuck_tuning(self):
        """
        Analyze and attempt to fix stuck tuning process that doesn't create database files.

        Returns:
            bool: True if fix was attempted, False otherwise
        """
        # First, get tuning status
        status_info = self.read_tuning_status()

        # Check if tuning is marked as running
        if not status_info.get("is_running", False):
            return False

        # Check for DB files
        db_dir = os.path.join(self.data_dir, "DB")
        db_files_exist = False

        if os.path.exists(db_dir):
            db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
            db_files_exist = len(db_files) > 0

        # Check if tested_models.yaml exists and has content
        tested_models_exists = os.path.exists(self.tested_models_file)
        tested_models_has_content = False

        if tested_models_exists:
            try:
                with open(self.tested_models_file, "r") as f:
                    content = yaml.safe_load(f)
                    tested_models_has_content = content and len(content) > 0
            except:
                pass

        # Diagnose the issue if tuning is running but no DB files exist
        if (
            status_info.get("is_running", False)
            and not db_files_exist
            and not tested_models_has_content
        ):
            # This is likely a stuck initialization
            self.log_error(
                "Stuck Initialization",
                "Tuning process is marked as running but no database files or trials exist",
                "This indicates the process may be stuck in initialization",
            )

            # Run SQLite diagnostics
            sqlite_diagnosis = self.diagnose_sqlite_initialization()

            if not sqlite_diagnosis["can_create_optuna_db"]:
                self.log_error(
                    "SQLite Initialization Error",
                    f"Cannot create Optuna database: {sqlite_diagnosis['detailed_errors']}",
                )

                # Log recommendations
                for recommendation in sqlite_diagnosis["recommendations"]:
                    logger.warning(f"Recommendation: {recommendation}")

            # Check if the DB directory actually exists and is writable
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    self.log_error(
                        "DB Directory Created",
                        f"Created missing DB directory: {db_dir}",
                    )
                except Exception as e:
                    self.log_error(
                        "DB Directory Error", f"Failed to create DB directory: {e}"
                    )

            # Check DB directory permissions
            try:
                test_file = os.path.join(db_dir, "permission_test.txt")
                with open(test_file, "w") as f:
                    f.write("Test")
                os.remove(test_file)
            except Exception as e:
                self.log_error("Permission Error", f"DB directory is not writable: {e}")

            # Force reset status file and remove any stale locks
            self.reset_tuning_status(force=True)
            self.cleanup_stale_locks()

            return True

        return False

    def monitoring_loop(self):
        """Enhanced monitoring loop with additional checks for stuck initialization."""
        logger.info("Starting enhanced monitoring loop...")
        initialization_check_interval = (
            60  # Check for stuck initialization more frequently
        )
        last_initialization_check = 0

        while not self.stop_event.is_set():
            try:
                # Check running status
                is_running = self.is_tuning_running()

                # More frequent check for stuck initialization
                current_time = time.time()
                if is_running and (
                    current_time - last_initialization_check
                    > initialization_check_interval
                ):
                    # Check for stuck initialization where status is running but no DB files
                    self.analyze_and_fix_stuck_tuning()
                    last_initialization_check = current_time

                # Check progress
                making_progress, progress_details = self.is_making_progress()

                # Monitor system resources
                resource_data = self.monitor_system_resources()

                # Check individual model health
                model_health = self.check_per_model_health(
                    progress_details.get("models_status", {})
                )

                # Analyze trial history
                trial_analysis = self.analyze_trial_history()

                # Update dashboard
                self.update_dashboard()

                # Log health of ensemble
                failing_models = [
                    model
                    for model, details in model_health.items()
                    if details.get("status") == "failing"
                ]
                unstable_models = [
                    model
                    for model, details in model_health.items()
                    if details.get("status") == "unstable"
                ]

                if failing_models:
                    logger.warning(
                        f"Models potentially failing: {', '.join(failing_models)}"
                    )

                if unstable_models:
                    logger.warning(
                        f"Models potentially unstable: {', '.join(unstable_models)}"
                    )

                # Check for resource constraints
                if resource_data.get("memory", {}).get("percent", 0) > 95:
                    self.log_error(
                        "Resource Warning",
                        "Memory usage critically high (>95%)",
                        f"Memory: {resource_data['memory']['used_gb']:.1f}GB/{resource_data['memory']['total_gb']:.1f}GB",
                    )

                if resource_data.get("disk", {}).get("percent", 0) > 95:
                    self.log_error(
                        "Resource Warning",
                        "Disk space critically low (>95%)",
                        f"Disk: {resource_data['disk']['used_gb']:.1f}GB/{resource_data['disk']['total_gb']:.1f}GB",
                    )

                # Decision logic for restart
                needs_restart = False
                restart_reason = ""

                # Case 1: Status says running but no progress
                if is_running and not making_progress:
                    needs_restart = True
                    restart_reason = "tuning marked as running but not making progress"

                # Case 2: Status says not running but should be
                elif not is_running and self.auto_restart:
                    # Only restart if we had previous activity
                    models_status = progress_details.get("models_status", {})
                    if models_status.get("model_count", 0) > 0:
                        needs_restart = True
                        restart_reason = "tuning is not running but should be"

                # Perform restart if needed
                if needs_restart:
                    logger.error(f"Tuning needs restart: {restart_reason}")

                    if self.auto_restart:
                        # Create backups before restart
                        if self.create_backups:
                            self.backup_critical_files()

                        logger.info("Attempting to restart tuning")
                        self.restart_tuning()
                    else:
                        logger.info("Auto-restart is disabled - skipping restart")

                # Sleep before next check
                self.stop_event.wait(self.check_interval)

            except Exception as e:
                self.log_error(
                    "Monitoring Loop Error",
                    f"Error in monitoring loop: {e}",
                    traceback.format_exc(),
                )
                # Sleep a bit before retrying after error
                self.stop_event.wait(30)

        logger.info("Monitoring loop stopped")

    def create_streamlit_component(self):
        """
        Create a Streamlit component that displays watchdog information.
        Call this from your Streamlit app to render watchdog status.
        """
        try:
            import pandas as pd
            import streamlit as st

            st.subheader("Tuning Watchdog Status")

            # Get current status
            is_running = self.is_tuning_running()
            making_progress, progress_details = self.is_making_progress()

            # Create columns for status indicators
            col1, col2, col3 = st.columns(3)

            # System Status
            with col1:
                if is_running and making_progress:
                    st.success("⚡ Tuning Active")
                elif is_running and not making_progress:
                    st.warning("⚠️ Tuning Stalled")
                else:
                    st.error("⛔ Tuning Inactive")

            # Progress
            with col2:
                if "models_status" in progress_details:
                    models = progress_details["models_status"]
                    st.metric("Completed Trials", models.get("model_count", 0))

            # Last Update
            with col3:
                if self.last_models_update_time > 0:
                    time_diff = time.time() - self.last_models_update_time
                    if time_diff < 60:
                        time_str = f"{int(time_diff)} seconds ago"
                    elif time_diff < 3600:
                        time_str = f"{int(time_diff / 60)} minutes ago"
                    else:
                        time_str = f"{int(time_diff / 3600)} hours ago"
                    st.metric("Last Update", time_str)

            # Model Type Distribution
            trial_analysis = self.analyze_trial_history()
            st.subheader("Model Distribution")

            # Only display if we have data
            if trial_analysis and "trial_counts" in trial_analysis:
                # Create a bar chart of model distribution
                counts = trial_analysis["trial_counts"]
                if counts:
                    import plotly.express as px

                    df = pd.DataFrame(
                        {
                            "Model Type": list(counts.keys()),
                            "Trial Count": list(counts.values()),
                        }
                    )

                    fig = px.bar(
                        df,
                        x="Model Type",
                        y="Trial Count",
                        title="Trials by Model Type",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Recent Activity Table
            st.subheader("Recent Trial Activity")
            if trial_analysis and "recent_activity" in trial_analysis:
                activity = trial_analysis["recent_activity"]
                if activity:
                    df = pd.DataFrame(activity)
                    st.dataframe(df, use_container_width=True)

            # Resource Usage
            if self.monitor_resources and psutil is not None:
                st.subheader("Resource Usage")
                resource_data = self.monitor_system_resources()

                if resource_data:
                    # Create 2 columns for CPU/Memory and Disk/GPU
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        cpu = resource_data.get("cpu_percent", 0)
                        memory = resource_data.get("memory", {}).get("percent", 0)

                        st.metric("CPU Usage", f"{cpu:.1f}%")
                        st.progress(cpu / 100)

                        st.metric("Memory Usage", f"{memory:.1f}%")
                        st.progress(memory / 100)

                    with res_col2:
                        disk = resource_data.get("disk", {}).get("percent", 0)
                        st.metric("Disk Usage", f"{disk:.1f}%")
                        st.progress(disk / 100)

                        # Show GPU if available
                        gpu_info = resource_data.get("gpu", {})
                        if gpu_info:
                            for gpu_id, gpu in gpu_info.items():
                                gpu_name = gpu.get("name", "GPU")
                                gpu_usage = gpu.get("gpu_percent", 0)
                                st.metric(f"{gpu_name} Usage", f"{gpu_usage:.1f}%")
                                st.progress(gpu_usage / 100)

            # Error Analysis
            if len(self.error_count_by_category) > 0:
                st.subheader("Error Summary")

                for category, count in self.error_count_by_category.most_common(5):
                    st.error(f"{category}: {count} occurrences")

                if st.button("View Full Error Log"):
                    if os.path.exists(self.error_log_file):
                        with open(self.error_log_file, "r") as f:
                            errors = f.read()
                        st.code(errors)

            # Control Section
            st.subheader("Watchdog Controls")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Force Restart Tuning"):
                    with st.spinner("Restarting tuning process..."):
                        success = self.restart_tuning()
                        if success:
                            st.success("Tuning restarted successfully!")
                        else:
                            st.error(
                                "Failed to restart tuning. Check logs for details."
                            )

            with col2:
                if st.button("Clean Lock Files"):
                    with st.spinner("Cleaning lock files..."):
                        self.cleanup_stale_locks()
                        st.success("Lock files cleaned")
        except ImportError as e:
            logger.error(f"Error creating Streamlit component (missing imports): {e}")
            import streamlit as st

            st.error("Error: Required packages for watchdog visualization are missing.")
            st.info("Please install: streamlit, pandas, plotly")
        except Exception as e:
            logger.error(f"Error creating Streamlit component: {e}")
            import streamlit as st

            st.error(f"Error displaying watchdog data: {e}")

    def get_status_report(self):
        """
        Generate a comprehensive status report.

        Returns:
            dict: Status report with all monitored information
        """
        try:
            is_running = self.is_tuning_running()
            making_progress, progress_details = self.is_making_progress()
            model_health = self.check_per_model_health(
                progress_details.get("models_status", {})
            )

            # Calculate elapsed times
            time_since_models_update = (
                time.time() - self.last_models_update_time
                if self.last_models_update_time > 0
                else None
            )
            time_since_restart = (
                time.time() - self.last_restart_time
                if self.last_restart_time > 0
                else None
            )

            return {
                "timestamp": time.time(),
                "is_running": is_running,
                "making_progress": making_progress,
                "model_health": model_health,
                "progress_details": progress_details,
                "watchdog_state": {
                    "running": self.running,
                    "consecutive_restarts": self.consecutive_restarts,
                    "last_models_update_time": self.last_models_update_time,
                    "time_since_models_update": time_since_models_update,
                    "time_since_models_update_minutes": (
                        time_since_models_update / 60
                        if time_since_models_update
                        else None
                    ),
                    "last_restart_time": self.last_restart_time,
                    "time_since_restart": time_since_restart,
                    "time_since_restart_minutes": (
                        time_since_restart / 60 if time_since_restart else None
                    ),
                },
                "config": {
                    "ticker": self.ticker,
                    "timeframe": self.timeframe,
                    "check_interval": self.check_interval,
                    "max_stale_time": self.max_stale_time,
                    "auto_restart": self.auto_restart,
                },
            }
        except Exception as e:
            logger.error(f"Error generating status report: {e}")
            return {"error": str(e)}

    def start(self):
        """Start the watchdog monitoring."""
        if self.running:
            logger.warning("Watchdog is already running")
            return

        logger.info("Starting tuning watchdog...")
        self.running = True
        self.stop_event.clear()

        # Start monitoring in a background thread
        self.thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.thread.start()

        # Force a check for stale lock files on startup
        self.cleanup_stale_locks()

        logger.info("Watchdog started")

    def stop(self):
        """Stop the watchdog monitoring."""
        if not self.running:
            logger.warning("Watchdog is not running")
            return

        logger.info("Stopping tuning watchdog...")
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=self.shutdown_grace_period)
            if self.thread.is_alive():
                logger.warning("Watchdog thread did not terminate cleanly")

        self.running = False
        logger.info("Watchdog stopped")


def setup_signal_handlers(watchdog):
    """Setup signal handlers for clean shutdown."""

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        watchdog.stop()
        sys.exit(0)

    # Register signal handlers (for non-Windows platforms)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point when run as a script."""
    import argparse

    parser = argparse.ArgumentParser(description="ML Tuning Watchdog")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument(
        "--ticker",
        default="BTC-USD",
        help="Default ticker to use when restarting tuning",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Default timeframe to use when restarting tuning",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="How often to check for stalls (seconds)",
    )
    parser.add_argument(
        "--max-stale-time",
        type=int,
        default=1800,
        help="Maximum time without progress before restarting (seconds)",
    )
    parser.add_argument(
        "--no-auto-restart", action="store_true", help="Disable automatic restarts"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Just show status and exit without starting watchdog",
    )
    parser.add_argument(
        "--no-resources", action="store_true", help="Disable resource monitoring"
    )

    args = parser.parse_args()

    # Create watchdog
    watchdog = TuningWatchdog(
        data_dir=args.data_dir,
        ticker=args.ticker,
        timeframe=args.timeframe,
        check_interval=args.check_interval,
        max_stale_time=args.max_stale_time,
        auto_restart=not args.no_auto_restart,
        monitor_resources=not args.no_resources,
    )

    if args.status_only:
        # Just show status and exit
        is_running = watchdog.is_tuning_running()
        making_progress, progress_details = watchdog.is_making_progress()

        print("\n=== Tuning Status ===")
        print(f"Running: {is_running}")
        print(f"Making Progress: {making_progress}")

        if "models_status" in progress_details:
            models = progress_details["models_status"]
            print(f"\nTested Models: {models.get('model_count', 0)}")
            print(
                f"Last Update: {datetime.fromtimestamp(models.get('last_update_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if "models_by_type" in models:
                print("\nModel Types:")
                for model_type, count in models.get("models_by_type", {}).items():
                    print(f"  {model_type}: {count}")

        if "progress_status" in progress_details:
            progress = progress_details["progress_status"]
            print(
                f"\nCurrent Trial: {progress.get('current_trial')}/{progress.get('total_trials')}"
            )
            print(f"Current Cycle: {progress.get('current_cycle')}")

        print(
            "\nWatchdog was not started. Use without --status-only to start monitoring."
        )
        sys.exit(0)

    # Setup signal handlers
    setup_signal_handlers(watchdog)

    # Start monitoring
    watchdog.start()

    logger.info("Watchdog is running. Press Ctrl+C to stop.")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        watchdog.stop()


if __name__ == "__main__":
    main()
