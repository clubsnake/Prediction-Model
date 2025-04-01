import subprocess
import os
import sys
import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create a log file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/launcher_log_{timestamp}.txt"

# Command to run
command = ["python", "launcher.py"]

# Add any arguments passed to this script
if len(sys.argv) > 1:
    command.extend(sys.argv[1:])

print(f"Running command: {' '.join(command)}")
print(f"Logging output to: {log_file}")

# Run the command and capture output
with open(log_file, "w") as f:
    f.write(f"Command: {' '.join(command)}\n")
    f.write(f"Started at: {datetime.datetime.now().isoformat()}\n\n")
    f.write("=== OUTPUT ===\n\n")
    
    # Run the process and redirect output to the log file
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read and write output line by line
    for line in process.stdout:
        f.write(line)
        f.flush()  # Ensure it's written immediately
        print(line, end='')  # Also print to console
        
    # Wait for process to complete
    return_code = process.wait()
    
    f.write(f"\n\nProcess completed with return code: {return_code}")
    f.write(f"\nFinished at: {datetime.datetime.now().isoformat()}")

print(f"\nProcess completed with return code: {return_code}")
print(f"Log file: {log_file}")
