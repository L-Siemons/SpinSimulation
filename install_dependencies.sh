#!/bin/bash

# install_dependencies.sh
# Script to install Python dependencies from requirements.txt
#
# How to use this script:
# 1. Place this script in the same directory as your requirements.txt file.
# 2. Make sure this script has execute permissions:
#    Run 'chmod +x install_dependencies.sh' in the terminal.
# 3. Execute the script:
#    Run './install_dependencies.sh' in the terminal.

#This process can be done using VS Code's integrated terminal.

echo "Starting installation of Python dependencies..."

# Check for Python and pip availability
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3."
    exit 1
fi

if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found. Please install pip3."
    exit 1
fi

# Install dependencies from requirements.txt
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "An error occurred during installation."
    exit 1
fi

