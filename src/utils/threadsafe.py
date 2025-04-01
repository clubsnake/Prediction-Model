"""
Provides thread-safe file operations to prevent data corruption and ensure consistency across multiple threads.
"""

import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import yaml

# Configure logging
logger = logging.getLogger("threadsafe")

# Import paths from config_loader
from config.config_loader import (
    PROGRESS_FILE,
    TUNING_STATUS_FILE,
    TESTED_MODELS_FILE
)

# Define critical files by extracting filenames from full paths
CRITICAL_FILES = [
    os.path.basename(PROGRESS_FILE),
    os.path.basename(TUNING_STATUS_FILE),
    os.path.basename(TESTED_MODELS_FILE)
]

# Global monitoring
_active_locks = {}
_locks_lock = threading.Lock()
_stop_cleanup_event = threading.Event()
_cleanup_thread = None

# Critical files that need special handling
MAX_RETRY_COUNT = 5
STALE_LOCK_THRESHOLD = 60  # seconds


def cleanup_stale_temp_files(directory=None, max_age=3600, pattern=None):
    """
    Clean up stale temporary files created during atomic write operations.
    
    Args:
        directory: Directory to search (defaults to project data directory)
        max_age: Maximum age in seconds before a temp file is considered stale
        pattern: Optional filename pattern to match (e.g., 'tested_models.yaml')
        
    Returns:
        int: Number of temporary files cleaned up
    """
    try:
        # Determine directory to clean
        if directory is None:
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(current_file))
            project_root = os.path.dirname(src_dir)
            directory = os.path.join(project_root, "data")

        logger.info(f"Cleaning up stale temporary files in {directory}")
        cleaned_count = 0
        
        # Find all temporary files recursively
        for root, _, files in os.walk(directory):
            for file in files:
                # Match different patterns of temp files created during atomic writes
                is_temp = (file.endswith('.tmp') or '.tmp.' in file or 
                          file.endswith('.temp') or '.temp.' in file or
                          '.emergency.' in file)
                
                # If pattern is provided, only clean temp files related to that pattern
                if is_temp and (pattern is None or pattern in file):
                    temp_path = os.path.join(root, file)
                    try:
                        # Check age of temp file
                        try:
                            file_age = time.time() - os.path.getmtime(temp_path)
                        except OSError:
                            # File might be inaccessible - consider it stale
                            file_age = max_age + 1
                            logger.warning(f"Could not check age of temp file: {temp_path}")

                        if file_age > max_age:
                            try:
                                os.remove(temp_path)
                                cleaned_count += 1
                                logger.info(f"Removed stale temp file: {temp_path} (Age: {file_age:.1f}s)")
                            except PermissionError:
                                logger.warning(f"Permission error removing temp file {temp_path} - may be in use")
                            except Exception as e:
                                logger.warning(f"Could not remove temp file {temp_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error checking temp file {temp_path}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale temporary files")
        return cleaned_count

    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")
        return 0


def cleanup_stale_locks(directory=None, max_age=300, force=False):
    """Clean up stale lock files in the directory with more aggressive handling for critical files."""
    try:
        # Determine directory to clean
        if directory is None:
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(current_file))
            project_root = os.path.dirname(src_dir)
            directory = os.path.join(project_root, "data")

        logger.info(f"Cleaning up stale lock files in {directory}")
        cleaned_count = 0
        
        # Find all lock files recursively
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(".lock"):
                    continue
                    
                lock_path = os.path.join(root, file)
                try:
                    # Check if the lock file is stale or force cleanup
                    if not os.path.exists(lock_path):
                        continue
                        
                    # Check age of lock file
                    try:
                        lock_age = time.time() - os.path.getmtime(lock_path)
                    except OSError:
                        # File might be inaccessible - consider it stale
                        lock_age = max_age + 1
                        logger.warning(f"Could not check age of lock file: {lock_path}")

                    # Determine appropriate max age based on file type
                    # More aggressive timeouts for critical files
                    is_critical = any(critical_file in lock_path for critical_file in CRITICAL_FILES)
                    effective_max_age = 30 if is_critical else max_age  # Shorter timeout for critical files
                    
                    if lock_age > effective_max_age or force:
                        # Try to log lock file content before removing
                        try:
                            with open(lock_path, "r") as lock_file:
                                lock_content = lock_file.read().strip()
                            logger.info(f"Removing stale lock file: {lock_path} (Age: {lock_age:.1f}s, Content: {lock_content})")
                        except Exception:
                            logger.info(f"Removing stale lock file: {lock_path} (Age: {lock_age:.1f}s)")
                            
                        # Remove the lock file
                        try:
                            os.remove(lock_path)
                            cleaned_count += 1
                        except PermissionError:
                            logger.warning(f"Permission error removing lock file {lock_path} - may be in use")
                            
                            # For critical files, try more aggressive approach
                            if is_critical:
                                try:
                                    # On Windows, try different approach to unlock
                                    if sys.platform == 'win32':
                                        import win32file
                                        try:
                                            # Try to open with share delete flag
                                            handle = win32file.CreateFile(
                                                lock_path,
                                                win32file.GENERIC_READ,
                                                win32file.FILE_SHARE_DELETE | win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                                                None,
                                                win32file.OPEN_EXISTING,
                                                0,
                                                None
                                            )
                                            win32file.CloseHandle(handle)
                                            # Now try to delete it again
                                            os.remove(lock_path)
                                            cleaned_count += 1
                                            logger.info(f"Successfully removed critical lock file using win32file: {lock_path}")
                                        except Exception as win_err:
                                            logger.warning(f"Could not remove critical lock file using win32file: {win_err}")
                                    else:
                                        # On Unix, try chmod to ensure we have permissions
                                        os.chmod(lock_path, 0o666)
                                        os.remove(lock_path)
                                        cleaned_count += 1
                                        logger.info(f"Successfully removed critical lock file after chmod: {lock_path}")
                                except Exception as e2:
                                    logger.warning(f"Could not remove critical lock file {lock_path} with enhanced methods: {e2}")
                        except Exception as e:
                            logger.warning(f"Could not remove lock file {lock_path}: {e}")
                except Exception as e:
                    logger.error(f"Error checking lock file {lock_path}: {e}")

        # Also clean up temporary files while we're at it
        temp_files_cleaned = cleanup_stale_temp_files(directory, max_age=max_age*2)  # Use longer timeout for temp files
        cleaned_count += temp_files_cleaned

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale files total")
        return cleaned_count

    except Exception as e:
        logger.error(f"Error cleaning up lock files: {e}")
        return 0


def _background_cleanup_task():
    """Background task that periodically cleans up stale lock files."""
    logger.info("Starting background lock file cleanup task")
    while not _stop_cleanup_event.is_set():
        try:
            # Sleep first to allow startup to complete
            for _ in range(60):  # Check stop event every second
                if _stop_cleanup_event.is_set():
                    break
                time.sleep(1)

            if _stop_cleanup_event.is_set():
                break

            # Run cleanup with shorter max_age for background task
            # Also cleans temporary files now
            count = cleanup_stale_locks(max_age=180)  # 3 minutes

            # If there are still many temp files, do targeted cleanup for tested_models
            if count > 0:
                # Special focus on tested_models.yaml temp files since those accumulate
                temp_count = cleanup_stale_temp_files(pattern='tested_models.yaml', max_age=120)  # More aggressive 2 minute timeout
                if temp_count > 0:
                    logger.info(f"Cleaned {temp_count} stale tested_models temporary files")

            # Log only if we cleaned something
            if count > 0:
                logger.info(f"Background task cleaned {count} stale files")

        except Exception as e:
            logger.error(f"Error in background cleanup task: {e}")


def start_background_cleanup():
    """Start the background cleanup thread if it's not already running."""
    global _cleanup_thread
    
    if _cleanup_thread is None or not _cleanup_thread.is_alive():
        _stop_cleanup_event.clear()
        _cleanup_thread = threading.Thread(target=_background_cleanup_task, daemon=True)
        _cleanup_thread.start()
        logger.info("Started background lock cleanup thread")


def stop_background_cleanup():
    """Stop the background cleanup thread."""
    if _cleanup_thread and _cleanup_thread.is_alive():
        _stop_cleanup_event.set()
        _cleanup_thread.join(timeout=2.0)
        logger.info("Stopped background lock cleanup thread")


class FileLockError(Exception):
    """Exception raised for file locking errors."""
    pass


class FileLock:
    """Cross-platform file locking with improved handling for contention."""

    def __init__(self, filepath: str, timeout: float = 10.0, retry_delay: float = 0.1):
        # Store path information
        self.filepath = filepath
        self.lockfile = f"{filepath}.lock"
        
        # Check if this is a critical file and adjust timeout accordingly
        is_critical = any(critical_file in filepath for critical_file in CRITICAL_FILES)
        self.timeout = min(5.0, timeout) if is_critical else timeout
        self.retry_delay = retry_delay
        
        # Lock state tracking
        self._lock_handle = None
        self._is_unix = os.name == "posix"
        self._thread_lock = threading.RLock()
        self._locked = False
        self._pid = os.getpid()
        self._lock_time = None
        
        # Register with active locks
        self._register_lock()

    def _register_lock(self):
        """Register this lock in the global active locks dictionary.
        Only called after successful lock acquisition."""
        with _locks_lock:
            if self._pid not in _active_locks:
                _active_locks[self._pid] = []
            if self.filepath not in [lock.filepath for lock in _active_locks[self._pid]]:
                _active_locks[self._pid].append(self)

    def _unregister_lock(self):
        """Unregister this lock from the global active locks dictionary."""
        with _locks_lock:
            if self._pid in _active_locks and self in _active_locks[self._pid]:
                _active_locks[self._pid].remove(self)
                if not _active_locks[self._pid]:  # Remove empty list
                    del _active_locks[self._pid]
                    
    def __enter__(self):
        logger.debug(f"Attempting to acquire lock: {self.lockfile}")
        if not self.acquire():
            raise FileLockError(f"Failed to acquire lock for {self.filepath}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock when exiting a 'with' block."""
        self.release()

    def acquire(self):
        """
        Acquire the lock with timeout and retry.

        Raises:
            FileLockError: If the lock cannot be acquired
        
        Returns:
            bool: True if lock was acquired successfully
        """
        with self._thread_lock:
            # Already locked by this instance
            if self._locked:
                return True
                
            # Check for and clean stale lock before attempting to acquire
            self._check_and_clean_stale_lock()
            
            start_time = time.time()
            acquired_handle = False
            
            # Add retry loop with exponential backoff
            retries = 0
            max_retries = MAX_RETRY_COUNT
            
            while True:
                try:
                    # Try to acquire the lock
                    if self._acquire_lock_impl():
                        acquired_handle = True
                        self._locked = True
                        self._lock_time = time.time()
                        # Only register AFTER successful acquisition
                        self._register_lock()
                        logger.debug(f"Lock acquired for {self.filepath} (PID: {self._pid})")
                        return True
                except Exception as e:
                    # Check if we've exceeded our retry limit
                    retries += 1
                    if retries >= max_retries:
                        # For critical files, try emergency cleanup before giving up
                        if any(critical_file in self.filepath for critical_file in CRITICAL_FILES):
                            try:
                                if os.path.exists(self.lockfile):
                                    logger.warning(f"Emergency cleanup for {self.lockfile}")
                                    os.remove(self.lockfile)
                                    # Try one last time
                                    if self._acquire_lock_impl():
                                        acquired_handle = True
                                        self._locked = True
                                        self._lock_time = time.time()
                                        # Only register AFTER successful acquisition
                                        self._register_lock()
                                        return True
                            except Exception:
                                pass
                        
                        # If we still couldn't acquire, raise an error
                        if acquired_handle and not self._locked:
                            try:
                                self._release_lock_impl()
                            except Exception:
                                pass
                        
                        raise FileLockError(f"Could not acquire lock after {max_retries} attempts: {self.filepath}")
                    
                    # Calculate exponential backoff with jitter
                    delay = min(self.retry_delay * (2 ** retries), 1.0) + random.random() * 0.1
                    time.sleep(delay)
                    
                    # Clean stale locks on each retry
                    if os.path.exists(self.lockfile):
                        self._check_and_clean_stale_lock(force=retries >= max_retries//2)
                
                # Check for timeout
                if time.time() - start_time >= self.timeout:
                    if acquired_handle and not self._locked:
                        try:
                            self._release_lock_impl()
                        except Exception:
                            pass
                    
                    # Try emergency cleanup on timeout for critical files
                    if any(critical_file in self.filepath for critical_file in CRITICAL_FILES):
                        try:
                            if os.path.exists(self.lockfile):
                                os.remove(self.lockfile)
                                logger.warning(f"Emergency removal of lock due to timeout: {self.lockfile}")
                        except Exception:
                            pass
                    
                    raise FileLockError(f"Timeout waiting for lock on {self.filepath}")
            
            # We should never reach here, but return False if we somehow do
            return False

    def _check_and_clean_stale_lock(self, force=False):
        """Check if the lock file is stale and clean it if needed."""
        if not os.path.exists(self.lockfile):
            return
            
        try:
            try:
                lock_age = time.time() - os.path.getmtime(self.lockfile)
            except OSError:
                # File might be inaccessible - consider it stale if force=True
                lock_age = float('inf') if force else 0
                logger.warning(f"Could not check age of lock file: {self.lockfile}")

            # Use shorter timeout for critical files
            is_critical = any(critical_file in self.filepath for critical_file in CRITICAL_FILES)
            max_age = STALE_LOCK_THRESHOLD if is_critical else 300  # 30 seconds for critical files

            if force or lock_age > max_age:
                logger.warning(f"Found stale lock file: {self.lockfile} (Age: {lock_age:.1f}s)")
                try:
                    os.remove(self.lockfile)
                    logger.info(f"Removed stale lock file: {self.lockfile}")
                except Exception as e:
                    logger.warning(f"Error removing stale lock: {e}")
                    
                    # Try more aggressive approach for critical files
                    if is_critical:
                        try:
                            # Try to change permissions and remove
                            os.chmod(self.lockfile, 0o666)
                            os.remove(self.lockfile)
                            logger.info(f"Removed stale lock file after chmod: {self.lockfile}")
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Error checking lock staleness: {e}")

    def _acquire_lock_impl(self):
        """Platform-specific lock acquisition implementation."""
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl
            
            try:
                # Open or create the file in write mode
                self._lock_handle = open(self.lockfile, "a+")
                fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Write PID and timestamp to lock file for better debugging
                self._lock_handle.seek(0)
                self._lock_handle.truncate()
                self._lock_handle.write(f"PID: {os.getpid()}, Thread: {threading.current_thread().name}, Time: {time.time()}")
                self._lock_handle.flush()
                
                return True
            except IOError:
                # Lock is held by another process
                if self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
                raise
        else:
            # Windows implementation - use file creation exclusivity
            try:
                # Check if lock file exists
                if os.path.exists(self.lockfile):
                    # Need to handle existing lock
                    raise IOError(f"Lock file exists: {self.lockfile}")
                
                # Create lock file with detailed info
                with open(self.lockfile, "w") as f:
                    f.write(f"PID: {os.getpid()}, Thread: {threading.current_thread().name}, Time: {time.time()}")
                    f.flush()
                    os.fsync(f.fileno())  # Ensure file is written to disk
                
                return True
            except Exception:
                raise

    def release(self):
        """Release the lock."""
        with self._thread_lock:
            if not self._locked:
                return

            try:
                self._release_lock_impl()
                # Unregister before setting _locked to false for better tracking
                self._unregister_lock()
                self._locked = False
                logger.debug(f"Released lock for {self.filepath} (PID: {self._pid})")
            except Exception as e:
                logger.error(f"Could not release lock for {self.filepath}: {str(e)}")
                # Try emergency cleanup if release fails
                try:
                    if os.path.exists(self.lockfile):
                        os.remove(self.lockfile)
                        logger.warning(f"Emergency removal during failed release: {self.lockfile}")
                except Exception:
                    pass

    def _release_lock_impl(self):
        """Platform-specific lock release implementation."""
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl
            
            if self._lock_handle:
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
                
                # Try to remove the lock file but don't fail if we can't
                try:
                    if os.path.exists(self.lockfile):
                        os.remove(self.lockfile)
                except Exception:
                    pass
        else:
            # Remove lock file on Windows
            try:
                if os.path.exists(self.lockfile):
                    os.remove(self.lockfile)
            except Exception as e:
                logger.error(f"Failed to remove lock file {self.lockfile}: {e}")
            self._lock_handle = None


def emergency_cleanup_for_file(filepath: str) -> bool:
    """
    Perform emergency cleanup for a specific file, more aggressive than regular cleanup.
    This method should only be used when a critical file operation must proceed.
    
    Args:
        filepath: The file to clean lock files for
    
    Returns:
        bool: True if cleanup was successful
    """
    try:
        lock_file = f"{filepath}.lock"
        
        if os.path.exists(lock_file):
            logger.warning(f"Emergency cleanup requested for {lock_file}")
            
            try:
                # Check for multiple processes using the file (Windows specific)
                if sys.platform == 'win32':
                    import subprocess
                    # Try to check if file is locked
                    try:
                        # Attempt to open with exclusive access to check if locked
                        with open(lock_file, 'r+'):
                            pass
                        # If we get here, file isn't locked by another process
                    except PermissionError:
                        logger.warning(f"Lock file {lock_file} is in use by another process")
                        
                        try:
                            # Try to use win32file for emergency cleanup on Windows
                            import win32file
                            try:
                                handle = win32file.CreateFile(
                                    lock_file,
                                    win32file.GENERIC_READ,
                                    win32file.FILE_SHARE_DELETE | win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                                    None,
                                    win32file.OPEN_EXISTING,
                                    0,
                                    None
                                )
                                win32file.CloseHandle(handle)
                                # Now try to delete it again
                                os.remove(lock_file)
                                logger.info(f"Successfully removed lock file in emergency cleanup with win32file: {lock_file}")
                                return True
                            except Exception as win_err:
                                logger.warning(f"Could not use win32file emergency cleanup: {win_err}")
                        except ImportError:
                            pass
                            
                        return False
                
                # Try to remove the lock file directly
                os.remove(lock_file)
                logger.info(f"Successfully removed lock file in emergency cleanup: {lock_file}")
                return True
                
            except PermissionError:
                logger.warning(f"Permission error removing lock file {lock_file} - still in use")
                return False
            except Exception as e:
                logger.error(f"Error in emergency cleanup for {lock_file}: {e}")
                return False
                
        return True  # No lock file to clean up
        
    except Exception as e:
        logger.error(f"Error in emergency cleanup: {e}")
        return False


def safe_read_yaml(filepath: str, default: Any = None, max_retries: int = 5) -> Dict:
    """
    Thread-safe YAML file reading with retries and locking.
    Enhanced with more aggressive handling of critical files.

    Args:
        filepath: Path to the YAML file
        default: Default value if file doesn't exist or is invalid
        max_retries: Maximum number of retries on failure

    Returns:
        Parsed YAML data or default value
    """
    # Special case for tuning_status.txt file
    if filepath.endswith("tuning_status.txt"):
        try:
            # Read as text file with key: value format instead of YAML
            result = {}
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for line in f:
                        if ":" in line:
                            parts = line.strip().split(":", 1)
                            if len(parts) == 2:
                                key, value = parts
                                result[key.strip()] = value.strip()
            return result or default
        except Exception as e:
            logger.error(f"Error reading tuning status file: {e}")
            return {} if default is None else default
    
    # Special handling for critical files - use shorter timeout
    file_path = filepath  # Avoid variable name 'os'
    is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
    timeout = 5.0 if is_critical else 20.0
    
    if not os.path.exists(file_path):
        return {} if default is None else default
        
    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(file_path, timeout=timeout):
                # Check again after acquiring lock
                if not os.path.exists(file_path):
                    return {} if default is None else default
                    
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data
        except (yaml.YAMLError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to read YAML after {max_retries} attempts: {e}")
                # For critical files, try emergency read without lock as last resort
                if is_critical:
                    try:
                        logger.warning(f"Attempting emergency read of critical file without lock: {file_path}")
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f) or {}
                            return data
                    except Exception as last_e:
                        logger.error(f"Emergency read failed: {last_e}")
                return {} if default is None else default
                
            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            return {} if default is None else default


def safe_write_yaml(filepath: str, data: Any, max_retries: int = 5) -> bool:
    """Thread-safe YAML writing with temporary file approach and better error handling."""
    temp_file = f"{filepath}.tmp.{int(time.time() * 1000)}"
    
    try:
        # Write to temp file first
        with open(temp_file, 'w') as f:
            yaml.dump(data, f)
            
        # Try to replace with original file
        for retry in range(max_retries):
            try:
                os.replace(temp_file, filepath)
                return True
            except PermissionError as e:
                # If file is locked, wait and retry
                if retry < max_retries - 1:
                    logger.debug(f"File locked, retrying in {(retry+1)*0.5}s: {filepath}")
                    time.sleep((retry + 1) * 0.5)
                else:
                    raise e
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write YAML to {filepath}: {e}")
        return False
    
    finally:
        # Always try to clean up the temp file
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            # Log but don't halt execution
            logger.warning(f"Failed to remove temp file {temp_file}: {e}")


def append_to_yaml_list(filepath: str, item: Any, max_attempts: int = 5) -> bool:
    """
    Append an item to a YAML file containing a list.
    Enhanced with better stale lock handling.

    Args:
        filepath: Path to the YAML file
        item: Item to append to the list
        max_attempts: Maximum number of retry attempts

    Returns:
        bool: True if successful, False otherwise
    """
    file_path = filepath  # Avoid variable name 'os'
    
    # Remove any stale locks before attempting to acquire a new one
    lock_file = f"{file_path}.lock"
    if os.path.exists(lock_file):
        try:
            lock_age = time.time() - os.path.getmtime(lock_file)
            if lock_age > 30:  # Consider lock stale after 30 seconds (more aggressive)
                os.remove(lock_file)
                logger.info(f"Removed stale lock file before appending: {lock_file}")
        except Exception as e:
            logger.warning(f"Could not remove stale lock file: {e}")
    
    # Read existing data
    try:
        existing_data = safe_read_yaml(file_path, default=[])
        if not isinstance(existing_data, list):
            logger.warning(f"File {file_path} doesn't contain a list, will use empty list")
            existing_data = []
    except Exception as e:
        logger.warning(f"Error reading existing list, will create new file: {e}")
        existing_data = []
    
    # Add timestamp if not present
    if isinstance(item, dict) and "timestamp" not in item:
        item["timestamp"] = datetime.now().isoformat()
    
    # Append the new item
    item_processed = convert_to_native_types(item)
    existing_data.append(item_processed)
    
    # Write back to file
    return safe_write_yaml(file_path, existing_data, max_retries=max_attempts)


# Utility function to convert NumPy types to native Python types
def convert_to_native_types(data):
    """Convert numpy types to Python native types recursively."""
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(i) for i in data]
    elif hasattr(data, "item") and callable(getattr(data, "item")):
        # Handle numpy scalars that have .item() method
        return data.item()
    elif hasattr(data, "tolist") and callable(getattr(data, "tolist")):
        # Handle numpy arrays
        return data.tolist()
    else:
        return data


# Add an atexit handler to clean up all locks on program exit
import atexit

def _cleanup_all_locks():
    """Clean up all locks registered by this process on exit."""
    process_id = os.getpid()  # Avoid variable name 'os'
    with _locks_lock:
        if process_id in _active_locks:
            locks = _active_locks[process_id].copy()  # Make a copy as we'll modify the original
            for lock in locks:
                try:
                    if lock._locked:
                        logger.warning(f"Force releasing lock on exit: {lock.filepath}")
                        lock.release()
                except Exception:
                    pass
            # Clean up any leftover lock files
            _active_locks.pop(process_id, None)

atexit.register(_cleanup_all_locks)

# Run cleanup at module import and start background task
cleaned_lock_count = cleanup_stale_locks()
cleaned_temp_count = cleanup_stale_temp_files(pattern='tested_models.yaml', max_age=7200)  # More aggressive cleanup for tested_models on startup
if cleaned_lock_count > 0 or cleaned_temp_count > 0:
    logger.warning(f"Found and cleaned up {cleaned_lock_count} stale lock files and {cleaned_temp_count} temporary files on startup")
start_background_cleanup()

class AtomicFileWriter:
    """
    Class for atomically writing to files to prevent data corruption.
    Uses a write-to-temp-then-rename pattern for atomic operations.
    """
    
    def __init__(self, filepath: str, mode: str = 'w', encoding: str = 'utf-8', 
                 use_lock: bool = True, timeout: float = 10.0):
        """
        Initialize the atomic file writer.
        
        Args:
            filepath: The target file path
            mode: File open mode ('w' for write, 'a' for append)
            encoding: File encoding
            use_lock: Whether to use FileLock for thread safety
            timeout: Lock acquisition timeout in seconds
        """
        self.filepath = filepath
        self.mode = mode
        self.encoding = encoding
        self.use_lock = use_lock
        self.timeout = timeout
        self.temp_path = None
        self.temp_file = None
        self.lock = None
        self._acquired_lock = False
        
    def __enter__(self):
        """Context manager entry - open temp file and acquire lock if needed."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(self.filepath))
        os.makedirs(directory, exist_ok=True)
        
        # Acquire lock if requested
        if self.use_lock:
            self.lock = FileLock(self.filepath, timeout=self.timeout)
            self.lock.acquire()
            self._acquired_lock = True
            
        # Create a unique temporary filename
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_suffix = random.randint(1000, 9999)
        filename = os.path.basename(self.filepath)
        self.temp_path = f"{self.filepath}.{timestamp}.{random_suffix}.tmp"
        
        # Open the temporary file
        self.temp_file = open(self.temp_path, self.mode, encoding=self.encoding)
        
        return self.temp_file
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close temp file, rename to target, release lock.
        If an exception occurred, delete the temp file without renaming.
        """
        # Close the temp file
        if self.temp_file:
            self.temp_file.flush()
            os.fsync(self.temp_file.fileno())  # Force write to disk
            self.temp_file.close()
            self.temp_file = None
            
        try:
            # If no exception occurred, perform the rename
            if exc_type is None:
                try:
                    # Use atomic rename operation
                    os.replace(self.temp_path, self.filepath)
                except Exception as e:
                    logger.error(f"Error renaming temp file {self.temp_path} to {self.filepath}: {e}")
                    # Try to delete the temp file on failure
                    if os.path.exists(self.temp_path):
                        try:
                            os.remove(self.temp_path)
                        except Exception:
                            pass
                    raise
            else:
                # An exception occurred, clean up the temp file
                if os.path.exists(self.temp_path):
                    try:
                        os.remove(self.temp_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {self.temp_path}: {e}")
        finally:
            # Always release the lock if we acquired one
            if self._acquired_lock and self.lock:
                self.lock.release()
                self._acquired_lock = False
                
    def write_string(self, content: str) -> bool:
        """
        Atomically write a string to the file.
        
        Args:
            content: String content to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing string to {self.filepath}: {e}")
            return False
            
    def write_bytes(self, content: bytes) -> bool:
        """
        Atomically write bytes to the file.
        
        Args:
            content: Bytes content to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use binary mode for bytes
            temp_mode = 'wb' if 'b' not in self.mode else self.mode
            orig_mode = self.mode
            self.mode = temp_mode
            
            with self as f:
                f.write(content)
            
            # Restore original mode
            self.mode = orig_mode
            return True
        except Exception as e:
            logger.error(f"Error writing bytes to {self.filepath}: {e}")
            self.mode = orig_mode  # Restore original mode
            return False
            
    def write_json(self, data: Any) -> bool:
        """
        Atomically write JSON data to the file.
        
        Args:
            data: Python object to serialize as JSON
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            processed_data = convert_to_native_types(data)  # Use existing function
            with self as f:
                json.dump(processed_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON to {self.filepath}: {e}")
            return False
            
    def write_yaml(self, data: Any) -> bool:
        """
        Atomically write YAML data to the file.
        
        Args:
            data: Python object to serialize as YAML
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            processed_data = convert_to_native_types(data)  # Use existing function
            with self as f:
                yaml.safe_dump(processed_data, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Error writing YAML to {self.filepath}: {e}")
            return False


class AtomicMultiFileUpdate:
    """Ensures multiple files are updated in a coordinated, atomic way."""
    
    def __init__(self, file_paths, timeout=10.0):
        self.file_paths = file_paths
        self.timeout = timeout
        self.locks = []
        self.temp_files = {}
        
    def __enter__(self):
        # Sort file paths to prevent deadlocks
        sorted_paths = sorted(self.file_paths)
        
        # Acquire locks in sorted order
        for path in sorted_paths:
            lock = FileLock(path, timeout=self.timeout)
            lock.acquire()
            self.locks.append(lock)
            
            # Create temp file for each path
            temp_path = f"{path}.tmp.{int(time.time() * 1000)}"
            self.temp_files[path] = temp_path
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # If no exception, commit all changes
            for path, temp_path in self.temp_files.items():
                if os.path.exists(temp_path):
                    os.replace(temp_path, path)
        else:
            # If exception, clean up temp files
            for temp_path in self.temp_files.values():
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Release all locks in reverse order
        for lock in reversed(self.locks):
            lock.release()
    
    def write_yaml(self, path, data):
        """Write YAML data to the temp file for a path"""
        if path not in self.temp_files:
            raise ValueError(f"Path {path} not in managed files")
            
        temp_path = self.temp_files[path]
        with open(temp_path, 'w') as f:
            yaml.safe_dump(data, f)
        
        return True
