"""
Provides thread-safe file operations with cross-platform locking for YAML and other files.
"""

import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict

import yaml

# Configure logging
logger = logging.getLogger("FileLock")


class FileLockError(Exception):
    """Exception raised for file locking errors."""


class FileLock:
    """
    Cross-platform file locking implementation.

    Supports:
    - Windows (using a separate lock file)
    - Unix/Linux/macOS (using fcntl)
    """

    def __init__(self, filepath: str, timeout: float = 10.0, retry_delay: float = 0.1):
        """
        Initialize a FileLock for the given file.

        Args:
            filepath: Path to the file to lock
            timeout: Maximum time to wait for the lock (seconds)
            retry_delay: Delay between lock attempts (seconds)
        """
        self.filepath = filepath
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.lockfile = f"{filepath}.lock"
        self._lock_handle = None
        self._is_unix = os.name == "posix"
        self._thread_lock = threading.RLock()  # Reentrant lock for thread safety

    def __enter__(self):
        """Acquire the lock when entering a 'with' block."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock when exiting a 'with' block."""
        self.release()

    def acquire(self):
        """
        Acquire the lock with timeout and retry.

        Raises:
            FileLockError: If the lock cannot be acquired
        """
        with self._thread_lock:
            start_time = time.time()

            while True:
                try:
                    if self._acquire_lock():
                        return True
                except Exception as e:
                    # If timeout exceeded, raise exception
                    if time.time() - start_time >= self.timeout:
                        raise FileLockError(
                            f"Could not acquire lock for {self.filepath}: {str(e)}"
                        )

                # Random jitter to prevent lock-step retries from multiple processes
                jitter = random.uniform(0, self.retry_delay * 0.5)
                time.sleep(self.retry_delay + jitter)

                # Check for timeout
                if time.time() - start_time >= self.timeout:
                    raise FileLockError(
                        f"Timeout while waiting for lock on {self.filepath}"
                    )

    def release(self):
        """
        Release the lock.

        Raises:
            FileLockError: If the lock cannot be released
        """
        with self._thread_lock:
            if self._lock_handle is None:
                return

            try:
                self._release_lock()
            except Exception as e:
                raise FileLockError(
                    f"Could not release lock for {self.filepath}: {str(e)}"
                )

    def _acquire_lock(self):
        """
        Platform-specific lock acquisition.

        Returns:
            True if the lock was acquired

        Raises:
            Exception: If the lock cannot be acquired
        """
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl

            try:
                # Open or create the file in write mode
                self._lock_handle = open(self.filepath, "a+")
                fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                # Lock is held by another process
                if self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
                raise
        else:
            # Use a separate lock file on Windows
            if os.path.exists(self.lockfile):
                # Lock file exists, check if it's stale
                try:
                    lock_age = time.time() - os.path.getmtime(self.lockfile)
                    if lock_age > 60:  # Consider lock stale after 60 seconds
                        logger.warning(f"Removing stale lock file: {self.lockfile}")
                        os.remove(self.lockfile)
                    else:
                        raise IOError("Lock file exists and is not stale")
                except:
                    raise IOError("Lock file exists")

            # Create lock file
            try:
                with open(self.lockfile, "w") as f:
                    f.write(f"Locked by process {os.getpid()} at {time.time()}")
                return True
            except:
                raise

    def _release_lock(self):
        """
        Platform-specific lock release.

        Raises:
            Exception: If the lock cannot be released
        """
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl

            if self._lock_handle:
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
        else:
            # Remove lock file on Windows
            if os.path.exists(self.lockfile):
                try:
                    os.remove(self.lockfile)
                except:
                    pass
            self._lock_handle = None


def safe_read_yaml(filepath: str, default: Any = None, max_retries: int = 5) -> Dict:
    """
    Thread-safe YAML file reading with retries and locking.

    Args:
        filepath: Path to the YAML file
        default: Default value if file doesn't exist or is invalid
        max_retries: Maximum number of retries on failure

    Returns:
        Parsed YAML data or default value
    """
    if not os.path.exists(filepath):
        return {} if default is None else default

    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data
        except (yaml.YAMLError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to read YAML after {max_retries} attempts: {e}")
                return {} if default is None else default

            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return {} if default is None else default


def safe_write_yaml(filepath: str, data: Any, max_retries: int = 5) -> bool:
    """
    Thread-safe YAML file writing with retries and locking.

    Args:
        filepath: Path to the YAML file
        data: Data to write
        max_retries: Maximum number of retries on failure

    Returns:
        True if successful, False otherwise
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Convert numpy types to Python native types
    def convert_to_native_types(obj):
        """Convert numpy types to Python native types recursively."""
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(i) for i in obj]
        elif hasattr(obj, "item") and callable(getattr(obj, "item")):
            # Handle numpy scalars that have .item() method
            return obj.item()
        else:
            return obj

    # Process the data
    processed_data = convert_to_native_types(data)

    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(filepath):
                # First write to a temporary file
                temp_path = f"{filepath}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(processed_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                # Then rename to the target file (atomic operation)
                os.replace(temp_path, filepath)
                return True

        except (yaml.YAMLError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to write YAML after {max_retries} attempts: {e}")
                return False

            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error writing {filepath}: {e}")
            return False


def safe_read_json(filepath: str, default: Any = None, max_retries: int = 5) -> Dict:
    """
    Thread-safe JSON file reading with retries and locking.

    Args:
        filepath: Path to the JSON file
        default: Default value if file doesn't exist or is invalid
        max_retries: Maximum number of retries on failure

    Returns:
        Parsed JSON data or default value
    """
    if not os.path.exists(filepath):
        return {} if default is None else default

    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                    return data
        except (json.JSONDecodeError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to read JSON after {max_retries} attempts: {e}")
                return {} if default is None else default

            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return {} if default is None else default


def safe_write_json(filepath: str, data: Any, max_retries: int = 5) -> bool:
    """
    Thread-safe JSON file writing with retries and locking.

    Args:
        filepath: Path to the JSON file
        data: Data to write
        max_retries: Maximum number of retries on failure

    Returns:
        True if successful, False otherwise
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Convert numpy types to Python native types
    processed_data = convert_to_native_types(data)

    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(filepath):
                # First write to a temporary file
                temp_path = f"{filepath}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                # Then rename to the target file (atomic operation)
                os.replace(temp_path, filepath)
                return True

        except (json.JSONDecodeError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to write JSON after {max_retries} attempts: {e}")
                return False

            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error writing {filepath}: {e}")
            return False


def convert_to_native_types(obj):
    """Convert numpy types to Python native types recursively."""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        # Handle numpy scalars that have .item() method
        return obj.item()
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        # Handle numpy arrays
        return obj.tolist()
    else:
        return obj


class AtomicFileWriter:
    """
    Thread-safe atomic file writer with locking.
    """

    def __init__(self, filepath: str, mode: str = "w", encoding: str = "utf-8"):
        """
        Initialize an atomic file writer.

        Args:
            filepath: Path to the file
            mode: File opening mode
            encoding: File encoding
        """
        self.filepath = filepath
        self.mode = mode
        self.encoding = encoding
        self.temp_path = f"{filepath}.tmp"
        self.lock = FileLock(filepath)
        self.file = None

    def __enter__(self):
        """Acquire lock and open temporary file when entering a 'with' block."""
        self.lock.acquire()
        self.file = open(self.temp_path, self.mode, encoding=self.encoding)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file, commit changes, and release lock when exiting a 'with' block."""
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()

        if exc_type is None:
            # No exception, perform the rename
            os.replace(self.temp_path, self.filepath)
        else:
            # Exception occurred, remove the temporary file
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)

        self.lock.release()


def remove_stale_lock(lock_path, max_age=300):
    """Remove lock file if it is older than max_age (seconds)."""
    if os.path.exists(lock_path):
        lock_age = time.time() - os.path.getmtime(lock_path)
        if lock_age > max_age:
            try:
                os.remove(lock_path)
                print(f"Removed stale lock file: {lock_path}")
            except Exception as e:
                print(f"Could not remove stale lock file {lock_path}: {e}")
