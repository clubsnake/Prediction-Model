"""
Main launcher for the Prediction Model project.
This script serves as the primary entry point and integrates all components.
"""

import argparse
import os
import subprocess
import sys

# Add project root to system path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(project_root, "Data", "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set environment variables BEFORE any ML library imports
from src.utils.env_setup import setup_tf_environment

env_vars = setup_tf_environment()

# Now the rest of the imports
import logging
import threading
import time
import webbrowser
from datetime import datetime
from typing import Any, Dict

# Configure logging to use the logs folder
log_file = os.path.join(
    logs_dir, f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("launcher")
logger.info(f"Logs will be saved to: {log_file}")

# Import project modules
from src.utils.robust_handler import (
    GracefulShutdownHandler,
    initiate_shutdown,
    register_shutdown_callback,
    register_thread,
)


class PredictionModelLauncher:
    """
    Main launcher class for the Prediction Model project.
    Handles starting different components and ensures graceful shutdown.
    """

    def __init__(self):
        """Initialize the launcher."""
        self.processes: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.shutdown_handler = GracefulShutdownHandler()
        self.stop_event = threading.Event()

        # Register own shutdown callback
        register_shutdown_callback(self.cleanup)

        # Status tracking
        self.component_status: Dict[str, bool] = {
            "dashboard": False,
            "model_training": False,
            "data_processor": False,
            "api_server": False,
        }

        logger.info("Launcher initialized")

    def cleanup(self):
        """Clean up all resources during shutdown."""
        logger.info("Cleaning up launcher resources...")

        # Signal all threads to stop
        self.stop_event.set()

        # Terminate processes
        for name, process in list(self.processes.items()):
            try:
                if (
                    process and process.poll() is None
                ):  # If process exists and is running
                    logger.info(f"Terminating process: {name}")
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating process {name}: {str(e)}")

        # Wait for threads
        for name, thread in list(self.threads.items()):
            try:
                if thread and thread.is_alive():
                    logger.info(f"Waiting for thread to finish: {name}")
                    thread.join(timeout=2)
            except Exception as e:
                logger.error(f"Error joining thread {name}: {str(e)}")

        logger.info("Launcher cleanup completed")

    def start_dashboard(self, port=8501, open_browser=True) -> bool:
        """
        Start the Streamlit dashboard.

        Args:
            port: Port for the dashboard to run on
            open_browser: Whether to automatically open a browser window

        Returns:
            True if started successfully
        """
        try:
            logger.info(f"Starting dashboard on port {port}...")

            # Check if dashboard is already running
            if self.component_status["dashboard"]:
                logger.warning("Dashboard is already running")
                return False

            # Try multiple possible dashboard locations
            possible_paths = [
                os.path.join(project_root, "src", "dashboard", "dashboard", "dashboard_core.py"),
                os.path.join(project_root, "src", "dashboard", "dashboard_core.py"),
                os.path.join(project_root, "src", "dashboard", "main.py"),
            ]
            
            dashboard_path = None
            for path in possible_paths:
                logger.info(f"Checking dashboard path: {path}")
                if os.path.exists(path):
                    dashboard_path = path
                    logger.info(f"Found dashboard at: {dashboard_path}")
                    break
            
            if dashboard_path is None:
                logger.error("Dashboard file not found in any expected location")
                logger.error(f"Checked paths: {possible_paths}")
                return False
            
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                dashboard_path,
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--browser.serverAddress",
                "localhost",
                "--server.enableCORS",
                "false",
            ]

            # Start process
            logger.info(f"Starting streamlit with command: {' '.join(cmd)}")
            env_vars = os.environ.copy()
            env_vars["STREAMLIT_SERVER_PORT"] = str(port)
            process = subprocess.Popen(
                cmd,
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes["dashboard"] = process

            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self._monitor_process_logs,
                args=("dashboard", process),
                daemon=True,
            )
            register_thread(log_thread)
            log_thread.start()
            self.threads["dashboard_logs"] = log_thread
            self.component_status["dashboard"] = True

            # Wait a bit for startup
            time.sleep(3)

            # Open browser if requested
            if open_browser:
                webbrowser.open(f"http://localhost:{port}")

            logger.info(f"Dashboard started on http://localhost:{port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            return False

    def start_model_training(self, config_path=None) -> bool:
        """
        Start model training process.

        Args:
            config_path: Path to configuration file

        Returns:
            True if started successfully
        """
        try:
            logger.info("Starting model training...")

            # Check if training is already running
            if self.component_status["model_training"]:
                logger.warning("Model training is already running")
                return False

            # Default config path if not provided
            if not config_path:
                config_path = "config.json"

            # Prepare command
            cmd = [
                sys.executable,
                "scripts\optimization_handler.py",
                "--config",
                config_path,
            ]

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes["model_training"] = process

            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self._monitor_process_logs,
                args=("model_training", process),
                daemon=True,
            )
            register_thread(log_thread)
            log_thread.start()
            self.threads["training_logs"] = log_thread
            self.component_status["model_training"] = True

            logger.info("Model training started")
            return True
        except Exception as e:
            logger.error(f"Failed to start model training: {str(e)}")
            return False

    def start_data_processor(
        self, tickers=None, interval="1d", days_back=365, refresh_rate=3600
    ) -> bool:
        """
        Start data processing component.

        Args:
            tickers: List of stock symbols to process (default uses config)
            interval: Data interval ('1d', '1h', etc.)
            days_back: Number of days of historical data to fetch
            refresh_rate: Seconds between data refresh cycles

        Returns:
            True if started successfully
        """
        try:
            logger.info("Starting data processor...")

            # Check if processor is already running
            if self.component_status["data_processor"]:
                logger.warning("Data processor is already running")
                return False

            # Import data processor
            from src.data.data_manager import process_data_job, stop_processing

            # Register the stop function as shutdown callback
            register_shutdown_callback(stop_processing)

            # Use config or default tickers
            if tickers is None:
                try:
                    from config import TICKERS

                    tickers = TICKERS
                except (ImportError, AttributeError):
                    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

            # Start processing thread
            processor_thread = threading.Thread(
                target=process_data_job,
                args=(tickers, interval, days_back, refresh_rate),
                daemon=True,
                name="DataProcessor",
            )
            register_thread(processor_thread)
            processor_thread.start()

            self.threads["data_processor"] = processor_thread
            self.component_status["data_processor"] = True

            logger.info(f"Data processor started for tickers: {tickers}")
            return True

        except Exception as e:
            logger.error(f"Failed to start data processor: {str(e)}")
            return False

    def _monitor_process_logs(self, name: str, process: subprocess.Popen):
        """Monitor and log output from a subprocess."""
        try:
            logger.info(f"Starting log monitor for {name}")

            # Monitor stdout
            for line in iter(process.stdout.readline, ""):
                if self.stop_event.is_set():
                    break
                if line:
                    logger.info(f"{name} | {line.strip()}")

            # Check if process is still running
            if process.poll() is None:
                # Also read stderr
                for line in iter(process.stderr.readline, ""):
                    if self.stop_event.is_set():
                        break
                    if line:
                        logger.error(f"{name} stderr | {line.strip()}")
        except Exception as e:
            logger.error(f"Error monitoring logs for {name}: {str(e)}")
        finally:
            logger.info(f"Log monitor for {name} stopped")

            # Update component status when process exits
            if name == "dashboard":
                self.component_status["dashboard"] = False
            elif name == "model_training":
                self.component_status["model_training"] = False

    def start_interactive_mode(self):
        """Start interactive command mode."""
        try:
            print("\n" + "=" * 50)
            print("Welcome to the Prediction Model Launcher")
            print("=" * 50)

            print("Press Enter to start the dashboard with default settings")
            print("\nAvailable commands:")
            print("  dashboard  - Start the dashboard")
            print("  train      - Start model training")
            print("  process    - Start data processor")
            print("  status     - Check component status")
            print("  stop NAME  - Stop a specific component")
            print("  exit       - Exit the launcher")
            print("=" * 50)

            while True:
                try:
                    cmd = input("\nEnter command: ").strip().lower()

                    if cmd == "dashboard":
                        port = input("Enter port (default 8501): ").strip()
                        port = int(port) if port.isdigit() else 8501

                        open_browser = (
                            input("Open browser? (y/n, default y): ").strip().lower()
                        )
                        open_browser = False if open_browser == "n" else True
                        self.start_dashboard(port, open_browser)

                    elif cmd == "train":
                        config = input(
                            "Enter config path (or press Enter for default): "
                        ).strip()
                        self.start_model_training(config if config else None)

                    elif cmd == "process":
                        self.start_data_processor()

                    elif cmd == "status":
                        self.show_status()

                    elif cmd.startswith("stop "):
                        component = cmd[5:].strip()
                        self.stop_component(component)

                    elif cmd == "exit":
                        print("Exiting launcher...")
                        break

                    elif not cmd:  # User just hit enter
                        print("Launching dashboard with default settings...")
                        self.start_dashboard(port=8501, open_browser=True)

                    else:
                        print(f"Unknown command: {cmd}")
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
            # Initiate shutdown when exiting interactive mode
            initiate_shutdown(source="Interactive mode exit")
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")

    def show_status(self):
        """Display current status of all components."""
        print("\n--- Component Status ---")
        for component, status in self.component_status.items():
            status_text = "RUNNING" if status else "STOPPED"
            print(f"{component.ljust(15)}: {status_text}")

        # Also show memory usage
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.5)
            print(f"\nMemory Usage: {memory_mb:.1f} MB")
            print(f"CPU Usage: {cpu_percent:.1f}%")
        except:
            pass

    def stop_component(self, component: str):
        """Stop a specific component."""
        if component not in self.component_status:
            print(f"Unknown component: {component}")
            return

        if not self.component_status[component]:
            print(f"{component} is not running")
            return

        print(f"Stopping {component}...")

        try:
            # Stop process if it exists
            if component in self.processes:
                process = self.processes[component]
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

                    self.processes[component] = None

            # Join thread if it exists
            thread_name = (
                f"{component}" if component in self.threads else f"{component}_logs"
            )
            if thread_name in self.threads:
                thread = self.threads[thread_name]
                if thread and thread.is_alive():
                    thread.join(timeout=2)
                self.threads[thread_name] = None

            # Update status
            self.component_status[component] = False
            print(f"{component} stopped")

        except Exception as e:
            logger.error(f"Error stopping {component}: {str(e)}")
            print(f"Error stopping {component}: {str(e)}")


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(description="Prediction Model Launcher")
    parser.add_argument("--dashboard", action="store_true", help="Start the dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--train", action="store_true", help="Start model training")
    parser.add_argument("--config", type=str, help="Path to config file for training")
    parser.add_argument(
        "--process-data", action="store_true", help="Start data processor"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )
    args = parser.parse_args()

    # Create launcher
    launcher = PredictionModelLauncher()

    # Handle command line options
    components_started = 0

    if args.dashboard:
        if launcher.start_dashboard(port=args.port):
            components_started += 1

    if args.train:
        if launcher.start_model_training(config_path=args.config):
            components_started += 1

    if args.process_data:
        if launcher.start_data_processor():
            components_started += 1

    # If interactive mode requested or no specific components requested, start interactive mode
    if args.interactive or components_started == 0:
        launcher.start_interactive_mode()
    else:
        # If components were started non-interactively, wait for them
        try:
            print("Press Ctrl+C to exit")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInitiating shutdown...")
            initiate_shutdown(source="Keyboard interrupt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down gracefully.")
