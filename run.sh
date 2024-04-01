#!/bin/sh

# Try to activate a virtual env before running demo script
VENV_NAMES="env .env venv .venv"
for name in $VENV_NAMES; do
    VENV_PATH="$PWD/$name/bin/activate"
    if [ -e $VENV_PATH ]; then
        echo "Using virtualenv: $name"
        . "$PWD/$name/bin/activate"
        python3 "$PWD/rtsp_demo.py" "$@"
        deactivate
        exit 0
    fi
done

# Feedback if we can't find the virtual env
echo "Couldn't find virtual environment folder! Cannot run script..."

