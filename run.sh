#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "$0 <python_script>"
    exit 4
fi

if [ ! -f .env ]; then
    echo ".env file not found"
    exit 4
fi

set -a
source .env
set +a

SCRIPT="$1"
if [ ! -f "$SCRIPT" ]; then
    echo "Nonexistent python file, '$SCRIPT' not found"
    exit 1
fi

python3 "$SCRIPT"
