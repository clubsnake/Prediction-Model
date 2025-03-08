@echo off
echo Copying Conda environment...

:: Set environment name (change this if needed)
set ENV_NAME=ml_env

:: Get Conda's environment path
for /f "delims=" %%i in ('conda info --envs ^| findstr /C:"%ENV_NAME%"') do set ENV_PATH=%%i

:: Check if the environment exists
if not defined ENV_PATH (
    echo Error: Environment "%ENV_NAME%" not found!
    pause
    exit /b
)

:: Create a copy folder
mkdir env_copy 2>nul

:: Copy entire Conda environment
xcopy /E /I /Y "%ENV_PATH%" "ml_env\%ENV_NAME%"
echo Conda environment copied.

:: Create a ZIP file for transfer
powershell Compress-Archive -Path ml_env -DestinationPath env_copy.zip -Force
echo Environment archive created: ml_env.zip

echo Done! Transfer env_copy.zip to the new PC.
pause
