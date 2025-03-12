import os
import subprocess

# Set project root to be the parent of the current (tests) folder.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories to check: project_root/src and project_root/config
CHECK_DIRECTORIES = [
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "config"),
]

# Where to save reports: project_root/tests/errors
ERRORS_DIR = os.path.join(PROJECT_ROOT, "tests", "errors")
os.makedirs(ERRORS_DIR, exist_ok=True)

# Error level options and corresponding output filenames
ERROR_OPTIONS = {
    "full": {
        "pylint": "",
        "flake8": "",
        "mypy": "",
        "bandit": "",
        "filename": "full_errors.txt",
    },
    "critical_fatal": {
        "pylint": "--disable=all --enable=E,F",
        "flake8": "E,F",
        "mypy": "",
        "bandit": "",
        "filename": "critical_fatal_errors.txt",
    },
    "fatal": {
        "pylint": "--disable=all --enable=F",
        "flake8": "F",
        "mypy": "",
        "bandit": "",
        "filename": "fatal_errors.txt",
    },
}

# Tools and their command formats
TOOLS = {
    "pylint": "pylint {options} {paths}",
    "flake8": "flake8 --select={options} {paths}",
    "mypy": "mypy {paths} --ignore-missing-imports",
    "bandit": "bandit -r {paths} -ll",
}


def run_tool(tool_name: str, options: str) -> str:
    """Runs a static analysis tool with given options and returns its output.

    Args:
        tool_name (str): The name of the tool to run (e.g., "pylint", "flake8").
        options (str): The command-line options to pass to the tool.

    Returns:
        str: The combined stdout and stderr output of the tool, along with a header
             indicating the tool name, or an error message if the tool could not be run.
    """
    # Join the check directories, wrapping each path in quotes to handle spaces.
    paths = " ".join(f'"{path}"' for path in CHECK_DIRECTORIES)
    command = TOOLS[tool_name].format(options=options, paths=paths)
    print(f"Running command: {command}")  # Debugging information
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        return (
            f"\n--- {tool_name.upper()} RESULTS ---\n" + result.stdout + result.stderr
        )
    except subprocess.TimeoutExpired:
        return (
            f"\n--- {tool_name.upper()} ERROR ---\nError running {tool_name}: Command timed out"
        )
    except Exception as e:
        return (
            f"\n--- {tool_name.upper()} ERROR ---\nError running {tool_name}: {str(e)}"
        )


def main():
    """Runs all tools for each error level and writes combined outputs to files."""
    for error_level, settings in ERROR_OPTIONS.items():
        output_file = os.path.join(ERRORS_DIR, settings["filename"])
        print(f"Running checks for: {error_level.upper()} (Saving to {output_file})")
        results = []
        for tool_name, tool_options in settings.items():
            if tool_name == "filename":
                continue
            results.append(run_tool(tool_name, tool_options))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        print(f"Saved {error_level.upper()} results to {output_file}")


if __name__ == "__main__":
    main()
