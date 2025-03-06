# Configuration Files

## Setup Instructions

This directory contains configuration files for the prediction model.

### API Keys Setup

1. Copy the template file to create your private API keys file:
   ```
   cp api_keys.template.yaml api_keys.yaml
   ```

2. Edit `api_keys.yaml` and add your actual API keys.

3. Keep `api_keys.yaml` private - it's added to .gitignore to prevent accidental commits.

## Configuration Files

- `user_config.yaml` - Main configuration file with user-editable parameters
- `api_keys.yaml` - Private file containing your personal API keys (not committed to version control)
- `api_keys.template.yaml` - Template showing the structure of the API keys file

The application will look for `api_keys.yaml` at the path specified in the main configuration file.
If it's not found, default empty values will be used.
