#!/bin/sh


# Define expected virtual environment folder name
TARGET_VENV=".venv"

# Check if the virtual environment exists
if [ -d "$TARGET_VENV" ]; then
    . "$TARGET_VENV/bin/activate"
else
    echo "Creating virtual environment..."
    python3 -m venv "$TARGET_VENV"
    . "$TARGET_VENV/bin/activate"
    pip3 install -r requirements.txt
fi


# Run the script
python3 "$PWD/demo.py" "$@"
deactivate

