@echo off
echo Setting up Conda environment...

:: Set environment name
set ml_env=ml_env

:: Unzip transfer files
powershell Expand-Archive -Path ml_env.zip -DestinationPath . -Force

:: Create Conda environment from exported YAML
conda env create -f env_transfer\environment.yml
echo Environment created.

:: Activate the environment
call conda activate %ENV_NAME%

:: Ensure exact package versions match
conda install --yes --file env_transfer\explicit_packages.txt
echo Dependencies installed.

:: Remove temporary files
rmdir /s /q env_transfer 2>nul
del env_transfer.zip 2>nul

echo Setup complete! Activate your environment with:
echo conda activate %ENV_NAME%
pause
