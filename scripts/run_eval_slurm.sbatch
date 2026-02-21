#!/bin/bash
#SBATCH --job-name=prompt_energy_eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module purge
module load 2023r1
module load openmpi
module load cuda/12.1

# Activate Conda environment
echo "Loading Anaconda/Miniconda..."
module load miniconda3 || true

# Set environment path (user should export this before sbatch)
CONDA_ENV_PATH="${CONDA_ENV_PATH:?CONDA_ENV_PATH not set. Please export CONDA_ENV_PATH=/path/to/your/conda/env}"
echo "Activating environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# Verify Python environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Set project directory (default to current directory if not set)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# Set HuggingFace environment variables
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HF_ALLOW_CODE_EVAL=1

# Set Rust/Cargo path (if EnergiBridge is used)
export PATH="$HOME/.cargo/bin:$PATH"

# Read experiment parameters from environment
CONFIG_FILE="${CONFIG_FILE:-config/experiment.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Python binary: $PYTHON_BIN"
echo "=========================================="

# Check if EnergiBridge binary exists
ENERGI_BRIDGE_BIN="${ENERGI_BRIDGE_BIN:-$PROJECT_DIR/EnergiBridge/target/release/energibridge}"
if [[ ! -f "$ENERGI_BRIDGE_BIN" ]]; then
    echo "WARNING: EnergiBridge binary not found at $ENERGI_BRIDGE_BIN"
    echo "Please build EnergiBridge first or set ENERGI_BRIDGE_BIN environment variable"
    exit 1
fi

# Run the evaluation
echo "Starting evaluation..."
echo "=========================================="

python -u "$PROJECT_DIR/scripts/run_eval.py" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --python "$PYTHON_BIN"

EXIT_CODE=$?

echo "=========================================="
echo "Evaluation completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
