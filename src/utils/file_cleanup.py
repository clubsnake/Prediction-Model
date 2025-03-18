import os
import glob
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def cleanup_temp_files(directory, pattern="*.yaml.tmp*", max_age_days=7):
    """
    Cleans up temporary files that are older than the specified age.
    
    Args:
        directory: Directory to search for temp files
        pattern: File pattern to match (default: *.yaml.tmp*)
        max_age_days: Maximum age in days before deletion (default: 7)
    
    Returns:
        int: Number of files deleted
    """
    files_removed = 0
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    
    try:
        temp_files = glob.glob(os.path.join(directory, pattern))
        
        for file_path in temp_files:
            try:
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_modified < cutoff_time:
                    os.remove(file_path)
                    files_removed += 1
                    logger.info(f"Deleted old temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")
                
        if files_removed > 0:
            logger.info(f"Cleanup complete: removed {files_removed} old temp files")
        
        return files_removed
    
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        return 0
