
@echo off
set TARGET_VENV=".venv"

REM Check if the virtual environment exists
if exist %TARGET_VENV% (
    call %TARGET_VENV%\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv %TARGET_VENV%
    call %TARGET_VENV%\Scripts\activate.bat
    pip install -r requirements.txt
)

REM Run the script with args
python demo.py %*
deactivate

