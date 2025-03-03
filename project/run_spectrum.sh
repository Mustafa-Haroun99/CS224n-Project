#!/bin/bash

# Define an array of values for --top-percent
top_percent_values=(25 50 75)

# Loop through each value and run the Python script
for percent in "${top_percent_values[@]}"; do
    echo "Running spectrum.py with --top-percent=$percent"
    python run_spectrum.py --model-name=gpt2 --top-percent $percent
done

echo "All experiments completed!"