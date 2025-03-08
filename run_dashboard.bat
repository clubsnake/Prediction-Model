@echo off
echo Setting up environment...
set MPLBACKEND=Agg
set PYTHONPATH=%PYTHONPATH%;%CD%

echo Starting dashboard...
echo.
echo If you encounter module errors, please run: pip install -r requirements.txt
echo.
streamlit run src\dashboard\dashboard\dashboard_core.py

if %errorlevel% neq 0 (
    echo Error starting dashboard. Trying alternative method...
    python -m streamlit run src\dashboard\dashboard\dashboard_core.py
)

pause
