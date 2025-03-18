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

# Global monitoring
_active_locks = {}
_locks_lock = threading.Lock()
_stop_cleanup_event = threading.Event()
_cleanup_thread = None

# Critical files that need special handling
CRITICAL_FILES = ["progress.yaml", "tuning_status.txt", "tested_models.yaml"]


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

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale lock files")
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
            count = cleanup_stale_locks(max_age=180)  # 3 minutes

            # Log only if we cleaned something
            if count > 0:
                logger.info(f"Background task cleaned {count} stale lock files")

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
    """Cross-platform file locking implementation with improved handling of stale locks."""

    def __init__(self, filepath: str, timeout: float = 10.0, retry_delay: float = 0.1):
        """
        Initialize a FileLock for the given file.

        Args:
            filepath: Path to the file to lock
            timeout: Maximum time to wait for the lock (seconds)
            retry_delay: Delay between lock attempts (seconds)
        """
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
        """Register this lock in the global active locks dictionary."""
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
        """Acquire the lock when entering a 'with' block."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock when exiting a 'with' block."""
        self.release()

    def __del__(self):
        """Ensure lock is released when object is garbage collected."""
        if self._locked:
            try:
                logger.warning(f"Lock on {self.filepath} was not explicitly released, cleaning up")
                self.release()
            except Exception:
                pass
            finally:
                self._unregister_lock()

    def acquire(self):
        """
        Acquire the lock with timeout and retry.

        Raises:
            FileLockError: If the lock cannot be acquired
        """
        with self._thread_lock:
            start_time = time.time()
            
            # Already locked by this instance
            if self._locked:
                return True
                
            # Check for and clean stale lock before attempting to acquire
            self._check_and_clean_stale_lock()
            
            while True:
                try:
                    if self._acquire_lock_impl():
                        self._locked = True
                        self._lock_time = time.time()
                        logger.debug(f"Acquired lock for {self.filepath} (PID: {self._pid})")
                        return True
                except Exception as e:
                    # If timeout exceeded, raise exception
                    if time.time() - start_time >= self.timeout:
                        # For critical files, try cleaning one more time before failing
                        file_path = self.filepath  # Avoid using 'os' variable name here
                        is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
                        lock_file_path = self.lockfile  # Avoid using 'os' variable name here
                        
                        if is_critical and os.path.exists(lock_file_path):
                            logger.warning(f"Timeout exceeded for critical file, attempting emergency cleanup: {file_path}")
                            try:
                                lock_age = time.time() - os.path.getmtime(lock_file_path)
                                if lock_age > 30:  # More aggressive 30 second threshold for emergency cleanup
                                    os.remove(lock_file_path)
                                    logger.warning(f"Emergency removal of stale lock: {lock_file_path}")
                                    # Try one more time
                                    if self._acquire_lock_impl():
                                        self._locked = True
                                        self._lock_time = time.time()
                                        logger.debug(f"Acquired lock after emergency cleanup: {file_path}")
                                        return True
                            except Exception as cleanup_e:
                                logger.error(f"Emergency cleanup failed: {cleanup_e}")
                                
                        raise FileLockError(f"Could not acquire lock for {file_path}: {str(e)}")
                
                # Random jitter to prevent lock-step retries from multiple processes
                jitter = random.uniform(0, self.retry_delay * 0.5)
                time.sleep(self.retry_delay + jitter)
                
                # Check for timeout
                if time.time() - start_time >= self.timeout:
                    # For critical files, one last attempt to clean stale lock
                    file_path = self.filepath  # Avoid using 'os' variable name here
                    is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
                    if is_critical:
                        self._check_and_clean_stale_lock(force=True)
                    
                    raise FileLockError(f"Timeout while waiting for lock on {file_path}")

    def _check_and_clean_stale_lock(self, force=False):
        """
        Check if the lock file is stale and clean it if needed.
        Enhanced for more aggressive handling of critical files.
        """
        if not self._is_unix:
            lock_file_path = self.lockfile  # Avoid variable name 'os'
            if os.path.exists(lock_file_path):
                try:
                    try:
                        lock_age = time.time() - os.path.getmtime(lock_file_path)
                    except OSError:
                        # File might be inaccessible - consider it stale if force is True
                        lock_age = float('inf') if force else 0
                        logger.warning(f"Could not check age of lock file: {lock_file_path}")

                    # Use a shorter timeout for critical files
                    file_path = self.filepath  # Avoid variable name 'os'
                    is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
                    max_age = 30 if is_critical else 300  # 30 seconds for critical, 5 minutes otherwise

                    if force or lock_age > max_age:
                        logger.warning(f"Found stale lock file: {lock_file_path} (Age: {lock_age:.1f}s)")
                        try:
                            with open(lock_file_path, "r") as f:
                                lock_content = f.read().strip()
                            logger.debug(f"Stale lock content: {lock_content}")
                        except Exception:
                            pass
                            
                        try:
                            os.remove(lock_file_path)
                            logger.info(f"Removed stale lock file: {lock_file_path}")
                        except PermissionError:
                            logger.warning(f"Permission error removing stale lock: {lock_file_path}")
                            
                            # For critical files, try more aggressive cleanup
                            if is_critical:
                                try:
                                    # On Windows, try win32file approach
                                    if sys.platform == 'win32':
                                        try:
                                            import win32file
                                            handle = win32file.CreateFile(
                                                lock_file_path,
                                                win32file.GENERIC_READ,
                                                win32file.FILE_SHARE_DELETE | win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                                                None,
                                                win32file.OPEN_EXISTING,
                                                0,
                                                None
                                            )
                                            win32file.CloseHandle(handle)
                                            # Now try to delete it again
                                            os.remove(lock_file_path)
                                            logger.info(f"Successfully removed stale lock using win32file: {lock_file_path}")
                                        except Exception as win_err:
                                            logger.warning(f"Could not remove stale lock using win32file: {win_err}")
                                    else:
                                        # On Unix, try chmod
                                        os.chmod(lock_file_path, 0o666)
                                        os.remove(lock_file_path)
                                        logger.info(f"Successfully removed stale lock with chmod: {lock_file_path}")
                                except Exception as e2:
                                    logger.warning(f"All cleanup methods failed for {lock_file_path}: {e2}")
                        except Exception as e:
                            logger.error(f"Failed to remove stale lock file: {e}")
                except Exception as e:
                    logger.error(f"Error checking lock staleness: {e}")

    def _acquire_lock_impl(self):
        """Implementation of platform-specific lock acquisition."""
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl
            
            try:
                # Open or create the file in write mode
                self._lock_handle = open(self.filepath, "a+")
                fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError as e:
                # Lock is held by another process - log details
                logger.debug(f"Lock acquisition failed for {self.filepath}: {e}")
                if self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
                raise
        else:
            # Windows implementation
            try:
                lock_file_path = self.lockfile  # Avoid variable name 'os'
                if os.path.exists(lock_file_path):
                    # Check lock file age
                    lock_age = time.time() - os.path.getmtime(lock_file_path)
                    
                    # Different staleness thresholds for different file types
                    file_path = self.filepath  # Avoid variable name 'os'
                    is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
                    is_db = file_path.endswith(".db") or ".db." in file_path
                    
                    # Configure thresholds based on file type
                    stale_threshold = 15 if is_critical else (60 if is_db else 30)  # More aggressive timeouts
                    
                    if lock_age > stale_threshold:
                        logger.warning(f"Removing stale lock file: {lock_file_path} (Age: {lock_age:.1f}s)")
                        os.remove(lock_file_path)
                    else:
                        # Try to read the lock file to get info about lock holder
                        try:
                            with open(lock_file_path, "r") as f:
                                lock_info = f.read()
                            logger.debug(f"Lock file exists: {lock_file_path}\nInfo: {lock_info}")
                        except Exception:
                            logger.debug(f"Lock file exists but couldn't read it: {lock_file_path}")
                        raise IOError(f"Lock file exists and is not stale: {lock_file_path}")
                
                # Create lock file with detailed info
                process_id = self._pid  # Avoid variable name confusion
                current_time = time.time()
                timestamp = datetime.now().isoformat()
                filename = os.path.basename(self.filepath)
                thread_name = threading.current_thread().name
                
                # Create parent directory if needed
                lock_dir = os.path.dirname(lock_file_path)
                if lock_dir and not os.path.exists(lock_dir):
                    os.makedirs(lock_dir, exist_ok=True)
                    
                with open(lock_file_path, "w") as f:
                    f.write(f"Locked by process {process_id} at {current_time} on {timestamp}\n")
                    f.write(f"File: {filename}\n")
                    f.write(f"Thread: {thread_name}")
                return True
            except Exception as e:
                logger.debug(f"Lock acquisition error for {self.lockfile}: {e}")
                raise

    def release(self):
        """
        Release the lock.

        Raises:
            FileLockError: If the lock cannot be released
        """
        with self._thread_lock:
            if not self._locked:
                return

            try:
                self._release_lock_impl()
                self._locked = False
                self._unregister_lock()
                logger.debug(f"Released lock for {self.filepath} (PID: {self._pid})")
            except Exception as e:
                raise FileLockError(f"Could not release lock for {self.filepath}: {str(e)}")

    def _release_lock_impl(self):
        """Platform-specific lock release."""
        if self._is_unix:
            # Use fcntl on Unix systems
            import fcntl
            
            if self._lock_handle:
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
        else:
            # Remove lock file on Windows
            lock_file_path = self.lockfile
            if os.path.exists(lock_file_path):
                try:
                    os.remove(lock_file_path)
                except PermissionError as e:
                    logger.warning(f"Permission error removing lock file {lock_file_path}: {e}")
                except Exception as e:
                    logger.error(f"Failed to remove lock file {lock_file_path}: {e}")
            self._lock_handle = None

    def is_locked(self):
        """Check if the file is locked by this instance."""
        return self._locked

    def break_lock(self):
        """
        Forcibly break an existing lock.
        Use with caution as it might cause data corruption if the lock is actively used.
        """
        logger.warning(f"Forcibly breaking lock on {self.filepath}")
        try:
            if self._is_unix and self._lock_handle:
                import fcntl
                fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
            else:
                lock_file_path = self.lockfile
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
            self._locked = False
            return True
        except Exception as e:
            logger.error(f"Failed to break lock on {self.filepath}: {e}")
            return False


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
    """
    Thread-safe YAML file writing with retries and locking.
    Enhanced with better error handling and emergency fallbacks.
    
    Args:
        filepath: Path to the YAML file
        data: Data to write
        max_retries: Maximum number of retries on failure
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    file_path = filepath  # Avoid variable name 'os'
    try:
        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory for {file_path}: {e}")
        
    # Convert NumPy types to native Python types
    processed_data = convert_to_native_types(data)
    
    # Configure timeout based on file type
    is_critical = any(critical_file in file_path for critical_file in CRITICAL_FILES)
    timeout = 5.0 if is_critical else 30.0
    
    # Try to write with retries
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Use atomic write pattern
            with FileLock(file_path, timeout=timeout):
                # Write to temp file with unique timestamp
                temp_path = f"{file_path}.tmp.{int(time.time() * 1000)}"
                with open(temp_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(processed_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    
                # Rename temp file to target file (atomic operation)
                os.replace(temp_path, file_path)
                return True
                
        except FileLockError as e:
            retry_count += 1
            logger.warning(f"Lock error writing YAML (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count >= max_retries - 1:
                # On the last retry, attempt emergency cleanup first
                logger.warning(f"Attempting emergency cleanup for {file_path} before final attempt")
                emergency_cleanup_for_file(file_path)
                
            if retry_count >= max_retries:
                # If all retries failed, try emergency direct write as last resort
                try:
                    logger.warning(f"Attempting emergency direct write for {file_path}")
                    # Write to unique temp file
                    temp_path = f"{file_path}.emergency.{int(time.time() * 1000)}"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(processed_data, f)
                        f.flush()
                        os.fsync(f.fileno())
                        
                    # Atomic rename
                    os.replace(temp_path, file_path)
                    logger.warning(f"Emergency direct write succeeded for {file_path}")
                    return True
                except Exception as last_e:
                    logger.error(f"Emergency write failed: {last_e}")
                    return False
                    
            # Exponential backoff with jitter
            delay = 0.2 * (2**retry_count) + random.uniform(0, 0.2)
            time.sleep(delay)
                
        except Exception as e:
            retry_count += 1
            logger.error(f"Error writing YAML (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count >= max_retries:
                return False
                
            # Linear backoff for other errors
            time.sleep(0.2 * retry_count)
            
    return False


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
cleaned_count = cleanup_stale_locks()
if cleaned_count > 0:
    logger.warning(f"Found and cleaned up {cleaned_count} stale lock files on startup")
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