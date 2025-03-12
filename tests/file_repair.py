"""
Utility to check and repair YAML data files.
"""

import os
import shutil
import time

import yaml


def check_and_repair_yaml(file_path, backup=True):
    """
    Check a YAML file for corruption and attempt to repair it.

    Args:
        file_path: Path to the YAML file
        backup: Whether to create a backup before repairing

    Returns:
        bool: True if file is now valid, False if repair failed
    """
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    try:
        # Try to load the file
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # File loaded successfully, no repair needed
        print(f"File {file_path} is valid YAML")
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

        # Create a backup if requested
        if backup:
            backup_path = f"{file_path}.{int(time.time())}.bak"
            try:
                shutil.copy2(file_path, backup_path)
                print(f"Created backup at {backup_path}")
            except Exception as be:
                print(f"Error creating backup: {be}")

        # Try to repair the file
        try:
            # Try to read as text and fix common YAML issues
            with open(file_path, "r") as f:
                content = f.read()

            # Remove any duplicate document markers
            if content.count("---") > 1:
                content = "---\n" + content.replace("---", "")

            # Try to parse line by line to find the error point
            lines = content.split("\n")
            valid_content = []

            for i, line in enumerate(lines):
                try:
                    yaml.safe_load("\n".join(valid_content + [line]))
                    valid_content.append(line)
                except Exception:
                    print(f"Found problematic line {i+1}: {line}")
                    # Skip this line

            # If we lost too much content, don't save the repair
            if len(valid_content) < len(lines) * 0.5:
                print("Too much content would be lost in repair, aborting")
                return False

            # Write repaired content
            with open(file_path, "w") as f:
                f.write("\n".join(valid_content))

            print(f"Repaired file {file_path}")

            # Verify the repair worked
            with open(file_path, "r") as f:
                yaml.safe_load(f)

            return True
        except Exception as re:
            print(f"Repair attempt failed: {re}")
            return False


def check_all_yaml_files(directory):
    """
    Check all YAML files in a directory and its subdirectories.

    Args:
        directory: Directory to scan

    Returns:
        tuple: (num_checked, num_valid, num_repaired, num_failed)
    """
    num_checked = 0
    num_valid = 0
    num_repaired = 0
    num_failed = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".yaml", ".yml")):
                file_path = os.path.join(root, file)
                num_checked += 1

                try:
                    # Try to load the file
                    with open(file_path, "r") as f:
                        data = yaml.safe_load(f)
                    num_valid += 1
                except Exception:
                    # Try to repair
                    if check_and_repair_yaml(file_path):
                        num_repaired += 1
                    else:
                        num_failed += 1

    return num_checked, num_valid, num_repaired, num_failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
        )

    print(f"Checking YAML files in {directory}")
    checked, valid, repaired, failed = check_all_yaml_files(directory)
    print(
        f"Results: {checked} checked, {valid} valid, {repaired} repaired, {failed} failed"
    )
