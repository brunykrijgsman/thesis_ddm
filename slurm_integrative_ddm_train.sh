#!/usr/bin/env bash
#SBATCH --job-name=ddm_train
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ddm_train-%j.out
#SBATCH --error=logs/ddm_train-%j.err

set -euf -o pipefail

# Load CUDA modules
echo "Loading CUDA modules..."
module load 2023

module purge

# UV version
UV_VERSION="0.9.5"

# Cache directory
export UV_CACHE_DIR=${TMPDIR}/uv_cache

# Set JAX as the Keras backend for BayesFlow
export KERAS_BACKEND=jax

# JAX GPU settings
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda,cpu

# Print job information
echo "=========================================="
echo "Job ID: ${SLURM_JOBID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Backend: JAX"
echo "=========================================="

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create logs directory
mkdir -p ${SLURM_SUBMIT_DIR}/logs

# Download uv if needed
if [ ! -e ${SLURM_SUBMIT_DIR}/uv ]; then
    echo "Downloading uv ${UV_VERSION}..."
    cd ${SLURM_SUBMIT_DIR}
    wget -q https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-musl.tar.gz -O - | tar xz --strip-components=1 -C . uv-x86_64-unknown-linux-musl/uv
    chmod +x uv
fi

cd ${SLURM_SUBMIT_DIR}

# Ensure any CPU-only JAX is removed before installing GPU wheel
./uv pip uninstall jax jaxlib -- -y || true

# Install the CUDA wheel (bundles CUDA & cuDNN)
./uv pip install -U "jax[cuda12]==0.6.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX GPU support
echo "=========================================="
echo "Checking JAX GPU support..."
./uv run python -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices())"
echo "=========================================="

# Run training
echo "Starting training script..."
./uv run python scripts/integrative_ddm_train.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
