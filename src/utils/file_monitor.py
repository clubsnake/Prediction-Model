import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TuningStatusHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith("tuning_status.txt"):
            self.callback()

def monitor_tuning_status(file_path, callback):
    """Monitor tuning_status.txt and call callback when modified."""
    event_handler = TuningStatusHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=file_path.rsplit(os.sep, 1)[0], recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
