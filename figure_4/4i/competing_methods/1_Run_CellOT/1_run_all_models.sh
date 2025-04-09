#!/bin/bash

# Check if the file path is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <file_path>"
    exit 1
fi

file_path=$1

# Check if the file exists
if [ ! -f "$file_path" ]; then
    echo "File not found: $file_path"
    exit 1
fi

# Read the file and create a list
list=$(<"$file_path")

# Print the list
echo "$list"


while IFS= read -r element; do
    echo "Running ba.sh with argument: $element"
        sbatch 2_sbatch_scheduler.sh "$element"
done <<< "$list"