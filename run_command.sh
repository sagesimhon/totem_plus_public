#!/bin/bash

# source ~/.bashrc
# conda activate totems

# Check if the command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [arguments...]"
    exit 1
fi

# Command to run (passed as arguments)
COMMAND="$@"

# Function to run the command until it succeeds
run_command() {
    while ! $COMMAND; do
        echo "Command failed. Retrying in 5 seconds..."
        sleep 5
    done
}

# Run the command function
run_command

echo "Command succeeded!"
