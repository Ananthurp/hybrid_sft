#!/bin/bash
# This is our master script to run all training experiments sequentially.

set -e # This command ensures that the script will exit immediately if any command fails.

# Navigate to the root of the code repository where the scripts should be launched from
cd /data/ananthu/gem_project/code/GEM

echo "=========================================================="
echo "Starting Experiment 1: Qwen2-1.5B with GEM Loss"
echo "=========================================================="

# Run the first script
bash scripts/qwen2-1.5b/train_qwen_gem.sh

echo "=========================================================="
echo "SUCCESS: GEM Loss Training Completed."
echo "Starting Experiment 2: Qwen2-1.5B with Cross-Entropy Loss"
echo "=========================================================="

# Because of 'set -e', this next command will only run if the one above succeeded.
bash scripts/qwen2-1.5b/train_ce_qwen.sh

echo "=========================================================="
echo "All training experiments have completed successfully."
echo "=========================================================="