"""
Provides thread-safe file operations with cross-platform locking for YAML and other files.
"""

import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from typing import Any, Dict

import yaml

# Configure logging
logger = logging.getLogger("FileLock")

# Background cleanup task
_cleanup_thread = None
_stop_cleanup_event = threading.Event()

# Keep track of active locks by process ID
_active_locks = {}
_locks_lock = threading.Lock()

# Critical files that need special handling
CRITICAL_FILES = ["progress.yaml", "tuning_status.txt", "tested_models.yaml"]


# Clean up stale lock files at startup - Make this more comprehensive
def cleanup_stale_locks(directory=None, max_age=300, force=False):
    """Clean up stale lock files in the directory."""
    import os
    import time

    try:
        if directory is None:
            # Default to looking in data directory at project root
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(current_file))
            project_root = os.path.dirname(src_dir)
            directory = os.path.join(project_root, "data")

        logger.info(f"Cleaning up stale lock files in {directory}")

        # Find all lock files
        all_locks = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".lock"):
                    all_locks.append(os.path.join(root, file))

        cleaned_count = 0
        for lock_file in all_locks:
            try:
                # Check if the lock file is stale or force cleanup
                if os.path.exists(lock_file):
                    lock_age = time.time() - os.path.getmtime(lock_file)

                    # Use shorter max_age for critical files
                    is_critical = any(
                        critical_file in lock_file for critical_file in CRITICAL_FILES
                    )
                    critical_max_age = 60  # 1 minute for critical files

                    if (
                        lock_age > (critical_max_age if is_critical else max_age)
                        or force
                    ):
                        # Read lock file content for debugging before removing
                        try:
                            with open(lock_file, "r") as f:
                                lock_content = f.read().strip()
                            logger.info(
                                f"Removing stale lock file: {lock_file} (Age: {lock_age:.1f}s, Content: {lock_content})"
                            )
                        except:
                            logger.info(
                                f"Removing stale lock file: {lock_file} (Age: {lock_age:.1f}s)"
                            )

                        try:
                            os.remove(lock_file)
                            cleaned_count += 1
                        except PermissionError as pe:
                            logger.warning(
                                f"Permission error removing lock, may be in use: {lock_file}: {pe}"
                            )
                            # Try more aggressive approach for critical files
                            if is_critical and os.name == "nt":  # Windows
                                try:
                                    import ctypes
                                    from ctypes import wintypes

                                    kernel32 = ctypes.WinDLL(
                                        "kernel32", use_last_error=True
                                    )
                                    MoveFileEx = kernel32.MoveFileExW
                                    MoveFileEx.argtypes = [
                                        wintypes.LPCWSTR,
                                        wintypes.LPCWSTR,
                                        wintypes.DWORD,
                                    ]
                                    MoveFileEx.restype = wintypes.BOOL
                                    MOVEFILE_DELAY_UNTIL_REBOOT = 4
                                    if MoveFileEx(
                                        lock_file, None, MOVEFILE_DELAY_UNTIL_REBOOT
                                    ):
                                        logger.info(
                                            f"Scheduled removal of lock file at next reboot: {lock_file}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to schedule lock removal: {e}"
                                    )
            except Exception as e:
                logger.error(f"Error checking lock file {lock_file}: {e}")

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
    global _stop_cleanup_event

    if _cleanup_thread is None or not _cleanup_thread.is_alive():
        _stop_cleanup_event.clear()
        _cleanup_thread = threading.Thread(target=_background_cleanup_task, daemon=True)
        _cleanup_thread.start()
        logger.info("Started background lock cleanup thread")


def stop_background_cleanup():
    """Stop the background cleanup thread."""
    global _cleanup_thread

    if _cleanup_thread and _cleanup_thread.is_alive():
        _stop_cleanup_event.set()
        _cleanup_thread.join(timeout=2.0)
        logger.info("Stopped background lock cleanup thread")


# New diagnostic functions for lock monitoring
def scan_lock_files(directory=None, return_details=False):
    """
    Scan for lock files in a directory and report statistics.

    Args:
        directory: Directory to scan (default: project data directory)
        return_details: If True, return detailed information about each lock file

    Returns:
        dict: Statistics about found lock files or detailed list if return_details=True
    """
    if directory is None:
        # Default to looking in data directory at project root
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))
        project_root = os.path.dirname(src_dir)
        directory = os.path.join(project_root, "data")

    lock_files = []
    current_time = time.time()

    try:
        # Find all lock files recursively
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".lock"):
                    lock_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(lock_path)
                        age = current_time - mtime

                        # Try to read lock content
                        content = ""
                        try:
                            with open(lock_path, "r") as f:
                                content = f.read().strip()
                        except Exception:
                            content = "<couldn't read content>"

                        lock_info = {
                            "path": lock_path,
                            "age": age,
                            "age_str": f"{age:.1f}s" if age < 60 else f"{age/60:.1f}m",
                            "is_stale": age > 300,  # Stale after 5 minutes
                            "content": content,
                            "locked_file": lock_path[:-5],  # Remove .lock
                        }
                        lock_files.append(lock_info)
                    except Exception as e:
                        lock_files.append(
                            {
                                "path": lock_path,
                                "error": str(e),
                                "is_stale": True,  # Consider it stale if we can't check
                            }
                        )
    except Exception as e:
        logger.error(f"Error scanning for lock files: {e}")

    # Prepare statistics
    stale_locks = [lock for lock in lock_files if lock.get("is_stale", True)]
    active_locks = [lock for lock in lock_files if not lock.get("is_stale", True)]

    stats = {
        "total": len(lock_files),
        "stale": len(stale_locks),
        "active": len(active_locks),
        "directory": directory,
        "timestamp": datetime.now().isoformat(),
    }

    if return_details:
        stats["lock_files"] = lock_files

    return stats


def diagnose_lock_issues(directory=None, fix_stale=False):
    """
    Diagnose lock file issues and optionally fix stale locks.

    Args:
        directory: Directory to scan (default: project data directory)
        fix_stale: If True, automatically remove stale lock files

    Returns:
        dict: Diagnostic results
    """
    # Get full scan with details
    scan_results = scan_lock_files(directory, return_details=True)

    # Process results to create diagnosis
    stale_locks = [
        lock
        for lock in scan_results.get("lock_files", [])
        if lock.get("is_stale", True)
    ]
    active_locks = [
        lock
        for lock in scan_results.get("lock_files", [])
        if not lock.get("is_stale", True)
    ]

    # Check for problems with the underlying files
    lock_file_issues = []
    for lock in scan_results.get("lock_files", []):
        if "locked_file" in lock:
            locked_file = lock["locked_file"]
            if not os.path.exists(locked_file):
                lock_file_issues.append(
                    {
                        "lock_path": lock["path"],
                        "issue": "Locked file does not exist",
                        "suggestion": "Remove the stale lock",
                    }
                )

    # Fix stale locks if requested
    cleaned = 0
    if fix_stale and stale_locks:
        for lock in stale_locks:
            try:
                os.remove(lock["path"])
                cleaned += 1
            except Exception as e:
                logger.error(f"Failed to remove stale lock {lock['path']}: {e}")

    # Prepare diagnosis report
    diagnosis = {
        "total_locks": scan_results["total"],
        "active_locks": len(active_locks),
        "stale_locks": len(stale_locks),
        "lock_file_issues": len(lock_file_issues),
        "directory": scan_results["directory"],
        "stale_locks_cleaned": cleaned,
        "timestamp": datetime.now().isoformat(),
        "issues": lock_file_issues,
        "detailed_active_locks": active_locks if active_locks else None,
        "detailed_stale_locks": stale_locks if stale_locks else None,
    }

    return diagnosis


def print_lock_diagnosis(diagnosis):
    """
    Print a human-readable report of lock file diagnosis.

    Args:
        diagnosis: Diagnosis dictionary from diagnose_lock_issues
    """
    print("\n===== Lock File Diagnosis =====")
    print(f"Timestamp: {diagnosis.get('timestamp')}")
    print(f"Directory: {diagnosis.get('directory')}")
    print(f"Total locks found: {diagnosis.get('total_locks')}")
    print(f"Active locks: {diagnosis.get('active_locks')}")
    print(f"Stale locks: {diagnosis.get('stale_locks')}")
    print(f"Locks with issues: {diagnosis.get('lock_file_issues')}")

    if diagnosis.get("stale_locks_cleaned", 0) > 0:
        print(f"Stale locks cleaned: {diagnosis.get('stale_locks_cleaned')}")

    if diagnosis.get("issues"):
        print("\nIssues found:")
        for issue in diagnosis.get("issues"):
            print(f"- {issue.get('lock_path')}: {issue.get('issue')}")
            print(f"  Suggestion: {issue.get('suggestion')}")

    if diagnosis.get("detailed_active_locks"):
        print("\nActive locks:")
        for lock in diagnosis.get("detailed_active_locks"):
            print(f"- {lock.get('path')} (Age: {lock.get('age_str')})")
            print(f"  Content: {lock.get('content')}")

    if diagnosis.get("detailed_stale_locks"):
        print("\nStale locks:")
        for lock in diagnosis.get("detailed_stale_locks"):
            print(f"- {lock.get('path')} (Age: {lock.get('age_str')})")
            print(f"  Content: {lock.get('content')}")

    print("\nRecommendations:")
    if diagnosis.get("stale_locks", 0) > 0:
        print("- Clean up stale lock files to prevent blocking file operations")
    if diagnosis.get("lock_file_issues", 0) > 0:
        print("- Address lock file issues by removing orphaned locks")
    if diagnosis.get("active_locks", 0) > 10:
        print("- High number of active locks may indicate process contention")


# Run cleanup at module import and start background task
cleaned_count = cleanup_stale_locks()
if cleaned_count > 0:
    logger.warning(f"Found and cleaned up {cleaned_count} stale lock files on startup")
start_background_cleanup()


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
        # Use shorter timeout for critical files to fail faster and prevent blocking
        is_critical = any(critical_file in filepath for critical_file in CRITICAL_FILES)
        self.timeout = (
            min(5.0, timeout) if is_critical else timeout
        )  # Max 5 seconds for critical files
        self.retry_delay = retry_delay
        self.lockfile = f"{filepath}.lock"
        self._lock_handle = None
        self._is_unix = os.name == "posix"
        self._thread_lock = threading.RLock()  # Reentrant lock for thread safety
        self._locked = False
        self._pid = os.getpid()
        self._lock_time = None

        # Register in active locks
        self._register_lock()

    def _register_lock(self):
        """Register this lock in the global active locks dictionary."""
        with _locks_lock:
            if self._pid not in _active_locks:
                _active_locks[self._pid] = []
            if self.filepath not in [
                lock.filepath for lock in _active_locks[self._pid]
            ]:
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
                logger.warning(
                    f"Lock on {self.filepath} was not explicitly released, cleaning up"
                )
                self.release()
            except:
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
                    if self._acquire_lock():
                        self._locked = True
                        self._lock_time = time.time()
                        logger.debug(
                            f"Acquired lock for {self.filepath} (PID: {self._pid})"
                        )
                        return True
                except Exception as e:
                    # If timeout exceeded, raise exception
                    if time.time() - start_time >= self.timeout:
                        # For critical files, try cleaning the lock one more time before failing
                        is_critical = any(
                            critical_file in self.filepath
                            for critical_file in CRITICAL_FILES
                        )
                        if is_critical and os.path.exists(self.lockfile):
                            logger.warning(
                                f"Timeout exceeded for critical file, attempting emergency cleanup: {self.filepath}"
                            )
                            try:
                                lock_age = time.time() - os.path.getmtime(self.lockfile)
                                if (
                                    lock_age > 60
                                ):  # 1 minute threshold for emergency cleanup
                                    os.remove(self.lockfile)
                                    logger.warning(
                                        f"Emergency removal of stale lock: {self.lockfile}"
                                    )
                                    # Try one more time
                                    if self._acquire_lock():
                                        self._locked = True
                                        self._lock_time = time.time()
                                        logger.debug(
                                            f"Acquired lock after emergency cleanup: {self.filepath}"
                                        )
                                        return True
                            except Exception as cleanup_e:
                                logger.error(f"Emergency cleanup failed: {cleanup_e}")

                        raise FileLockError(
                            f"Could not acquire lock for {self.filepath}: {str(e)}"
                        )

                # Random jitter to prevent lock-step retries from multiple processes
                jitter = random.uniform(0, self.retry_delay * 0.5)
                time.sleep(self.retry_delay + jitter)

                # Check for timeout
                if time.time() - start_time >= self.timeout:
                    # For critical files, one last attempt to clean stale lock
                    is_critical = any(
                        critical_file in self.filepath
                        for critical_file in CRITICAL_FILES
                    )
                    if is_critical:
                        self._check_and_clean_stale_lock(force=True)

                    raise FileLockError(
                        f"Timeout while waiting for lock on {self.filepath}"
                    )

    def _check_and_clean_stale_lock(self, force=False):
        """Check if the lock file is stale and clean it if needed."""
        if not self._is_unix and os.path.exists(self.lockfile):
            try:
                lock_age = time.time() - os.path.getmtime(self.lockfile)
                # Use a shorter timeout for critical files
                is_critical = any(
                    critical_file in self.filepath for critical_file in CRITICAL_FILES
                )
                max_age = (
                    60 if is_critical else 300
                )  # 1 minute for critical, 5 minutes otherwise

                if force or lock_age > max_age:
                    logger.warning(
                        f"Found stale lock file: {self.lockfile} (Age: {lock_age:.1f}s)"
                    )
                    try:
                        with open(self.lockfile, "r") as f:
                            lock_content = f.read().strip()
                        logger.debug(f"Stale lock content: {lock_content}")
                    except:
                        pass

                    try:
                        os.remove(self.lockfile)
                        logger.info(f"Removed stale lock file: {self.lockfile}")
                    except Exception as e:
                        logger.error(f"Failed to remove stale lock file: {e}")
            except Exception as e:
                logger.error(f"Error checking lock staleness: {e}")

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
                self._release_lock()
                self._locked = False
                self._unregister_lock()
                logger.debug(f"Released lock for {self.filepath} (PID: {self._pid})")
            except Exception as e:
                raise FileLockError(
                    f"Could not release lock for {self.filepath}: {str(e)}"
                )

    def _acquire_lock(self):
        """
        Platform-specific lock acquisition with improved error handling.

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
            except IOError as e:
                # Lock is held by another process - log details
                logger.debug(f"Lock acquisition failed for {self.filepath}: {e}")
                if self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
                raise
        else:
            # Windows implementation with improved error handling
            try:
                if os.path.exists(self.lockfile):
                    # Check lock file age
                    lock_age = time.time() - os.path.getmtime(self.lockfile)

                    # Different staleness thresholds for different file types
                    is_critical = any(
                        critical_file in self.filepath
                        for critical_file in CRITICAL_FILES
                    )
                    is_db = self.filepath.endswith(".db") or ".db." in self.filepath

                    # Configure thresholds based on file type
                    stale_threshold = (
                        30 if is_critical else (120 if is_db else 60)
                    )  # seconds

                    if lock_age > stale_threshold:
                        logger.warning(
                            f"Removing stale lock file: {self.lockfile} (Age: {lock_age:.1f}s)"
                        )
                        os.remove(self.lockfile)
                    else:
                        # Try to read the lock file to get info about lock holder
                        try:
                            with open(self.lockfile, "r") as f:
                                lock_info = f.read()
                            logger.debug(
                                f"Lock file exists: {self.lockfile}\nInfo: {lock_info}"
                            )
                        except:
                            logger.debug(
                                f"Lock file exists but couldn't read it: {self.lockfile}"
                            )
                        raise IOError(
                            f"Lock file exists and is not stale: {self.lockfile}"
                        )

                # Create lock file with detailed info
                lock_info = {
                    "pid": self._pid,
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "file": os.path.basename(self.filepath),
                }

                with open(self.lockfile, "w") as f:
                    f.write(
                        f"Locked by process {self._pid} at {time.time()} on {datetime.now().isoformat()}"
                    )
                    f.write(f"\nFile: {os.path.basename(self.filepath)}")
                    f.write(f"\nThread: {threading.current_thread().name}")
                return True
            except Exception as e:
                logger.debug(f"Lock acquisition error for {self.lockfile}: {e}")
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
                except Exception as e:
                    logger.error(f"Failed to remove lock file {self.lockfile}: {e}")
                    # Don't raise, as we want to continue even if lock removal fails
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
                if os.path.exists(self.lockfile):
                    os.remove(self.lockfile)
            self._locked = False
            return True
        except Exception as e:
            logger.error(f"Failed to break lock on {self.filepath}: {e}")
            return False


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
    # Special handling for critical files - use shorter timeout
    is_critical = any(critical_file in filepath for critical_file in CRITICAL_FILES)
    timeout = 5.0 if is_critical else 20.0

    if not os.path.exists(filepath):
        return {} if default is None else default

    retry_count = 0
    while retry_count < max_retries:
        try:
            with FileLock(filepath, timeout=timeout):
                if not os.path.exists(filepath):  # Check again after acquiring lock
                    return {} if default is None else default

                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data
        except (yaml.YAMLError, FileLockError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to read YAML after {max_retries} attempts: {e}")
                # For critical files, try emergency read without lock as last resort
                if is_critical:
                    try:
                        logger.warning(
                            f"Attempting emergency read of critical file without lock: {filepath}"
                        )
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f) or {}
                            return data
                    except Exception as last_e:
                        logger.error(f"Emergency read failed: {last_e}")
                return {} if default is None else default

            # Exponential backoff with jitter
            delay = 0.1 * (2**retry_count) + random.uniform(0, 0.1)
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return {} if default is None else default


def safe_write_yaml(filepath: str, data: Any, max_retries: int = 5) -> bool:
    """Thread-safe YAML file writing with retries and locking."""
    # Ensure the directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory for {filepath}: {e}")
        # Continue anyway, it might exist already

    # Convert numpy types to Python native types
    processed_data = convert_to_native_types(data)

    # Special handling for critical files
    is_critical = any(critical_file in filepath for critical_file in CRITICAL_FILES)
    timeout = 5.0 if is_critical else 30.0

    # More direct approach with better debugging
    retry_count = 0
    while retry_count < max_retries:
        try:
            # First check for and clean any stale lock
            if os.path.exists(f"{filepath}.lock"):
                lock_age = time.time() - os.path.getmtime(f"{filepath}.lock")
                if (
                    is_critical and lock_age > 30 or lock_age > 300
                ):  # 30s for critical, 5min otherwise
                    try:
                        os.remove(f"{filepath}.lock")
                        logger.warning(
                            f"Removed stale lock before write: {filepath}.lock"
                        )
                    except Exception as e:
                        logger.error(f"Failed to remove stale lock: {e}")

            # Use our FileLock to ensure thread safety
            with FileLock(filepath, timeout=timeout):
                logger.debug(
                    f"Writing YAML to {filepath} (attempt {retry_count+1}/{max_retries})"
                )

                # Write to temp file first with unique timestamp to avoid collisions
                temp_path = f"{filepath}.tmp.{int(time.time() * 1000)}"
                with open(temp_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(processed_data, f, default_flow_style=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                # Rename temp file to target file (atomic operation)
                os.replace(temp_path, filepath)
                logger.debug(f"Successfully wrote YAML to: {filepath}")
                return True

        except FileLockError as e:
            retry_count += 1
            logger.warning(
                f"Lock error writing YAML (attempt {retry_count}/{max_retries}): {str(e)}"
            )
            if retry_count >= max_retries:
                logger.error(
                    f"CRITICAL: Failed to acquire lock for {filepath} after {max_retries} attempts"
                )

                # Check if there's a stale lock file
                lock_file = f"{filepath}.lock"
                if os.path.exists(lock_file):
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > (
                        30 if is_critical else 60
                    ):  # Force cleanup of stale lock files
                        try:
                            logger.warning(
                                f"Removing stale lock file: {lock_file} (Age: {lock_age:.1f}s)"
                            )
                            os.remove(lock_file)
                            # Try one more time after removing lock
                            continue
                        except Exception as le:
                            logger.error(f"Failed to remove stale lock: {le}")

                # Fall back to direct write without locking
                try:
                    logger.warning(
                        f"Attempting direct write without locking for {filepath}"
                    )
                    # Write to temp file first
                    temp_path = f"{filepath}.emergency.{int(time.time() * 1000)}"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(processed_data, f)
                    # Then rename (atomic)
                    os.replace(temp_path, filepath)
                    logger.warning(f"Emergency direct write succeeded for {filepath}")
                    return True
                except Exception as last_e:
                    logger.error(f"Emergency write failed: {last_e}")
                    return False

            # Exponential backoff with jitter
            delay = 0.2 * (2**retry_count) + random.uniform(0, 0.2)
            time.sleep(delay)

        except Exception as e:
            retry_count += 1
            logger.error(
                f"Error writing YAML (attempt {retry_count}/{max_retries}): {str(e)}"
            )
            if retry_count >= max_retries:
                logger.error(
                    f"CRITICAL: Failed to write to {filepath} after {max_retries} attempts"
                )
                # Last resort - try direct write without temp file
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        yaml.safe_dump(processed_data, f)
                    logger.warning(f"Emergency direct write succeeded for {filepath}")
                    return True
                except Exception as last_e:
                    logger.error(f"Emergency write failed: {last_e}")
                    return False

            # Linear backoff for other errors
            time.sleep(0.2 * retry_count)

    return False


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


def update_nested_yaml(filepath: str, updates: Dict, max_attempts: int = 5) -> bool:
    """
    Update a YAML file with new values while preserving existing structure.

    Args:
        filepath: Path to the YAML file
        updates: Dictionary with updates to apply
        max_attempts: Maximum number of retry attempts

    Returns:
        bool: True if successful, False otherwise
    """
    # Read existing data
    existing_data = safe_read_yaml(filepath)

    # Deep merge updates into existing data
    merged_data = deep_merge(existing_data, updates)

    # Write back to file
    return safe_write_yaml(filepath, merged_data, max_attempts=max_attempts)


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    Values from dict2 override those in dict1 if there's a conflict.

    Args:
        dict1: Base dictionary
        dict2: Dictionary with updates

    Returns:
        Dict: Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def append_to_yaml_list(filepath: str, item: Any, max_attempts: int = 5) -> bool:
    """
    Append an item to a YAML file containing a list.

    Args:
        filepath: Path to the YAML file
        item: Item to append to the list
        max_attempts: Maximum number of retry attempts

    Returns:
        bool: True if successful, False otherwise
    """
    # Remove any stale lock files
    lock_file = f"{filepath}.lock"
    if os.path.exists(lock_file):
        try:
            lock_age = time.time() - os.path.getmtime(lock_file)
            if lock_age > 60:  # Consider lock stale after 60 seconds
                os.remove(lock_file)
                print(f"Removed stale lock file before appending: {lock_file}")
        except Exception as e:
            print(f"Could not remove stale lock file: {e}")

    print(f"Appending to YAML list in: {filepath}")

    # Attempt to read existing data
    existing_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    existing_data = data
                else:
                    print(f"File {filepath} doesn't contain a list, will overwrite")
        except Exception as e:
            print(f"Error reading existing data, will create new file: {e}")

    # Add the new item
    existing_data.append(convert_to_native_types(item))

    # Write all data back
    return safe_write_yaml(filepath, existing_data, max_retries=max_attempts)


# Add an atexit handler to clean up all locks on program exit
import atexit


def _cleanup_all_locks():
    """Clean up all locks registered by this process on exit."""
    pid = os.getpid()
    with _locks_lock:
        if pid in _active_locks:
            locks = _active_locks[
                pid
            ].copy()  # Make a copy as we'll modify the original
            for lock in locks:
                try:
                    if lock._locked:
                        logger.warning(f"Force releasing lock on exit: {lock.filepath}")
                        lock.release()
                except:
                    pass
            # Clean up any leftover lock files
            _active_locks.pop(pid, None)


atexit.register(_cleanup_all_locks)

# Run cleanup at module import and start background task
cleaned_count = cleanup_stale_locks()
if cleaned_count > 0:
    logger.warning(f"Found and cleaned up {cleaned_count} stale lock files on startup")
start_background_cleanup()
