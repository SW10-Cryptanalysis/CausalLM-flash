#!/bin/bash
#SBATCH --job-name=convert_data
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4      # Use a CPU partition (change 'cpu' if AAU uses a different name for CPU nodes)
#SBATCH --cpus-per-task=32   # Grab 32 cores to rip through the files
#SBATCH --mem=64G            # Plenty of memory to hold the paths

set -e

# Go to the directory where the script is
cd /ceph/project/SW10-CausalLM/Llama-xFormers

# Add uv to path if necessary
export PATH="$HOME/.local/bin:$PATH"

echo "Starting conversion $@..."
uv run src/preprocess.py "$@"
echo "Finished completely!"
