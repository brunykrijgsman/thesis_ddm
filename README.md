# single_trial_nddm_compare
#### (Repository version 0.0.3)
Repository for comparing model fits of single-trial and integrative (neural) drift-diffusion models using BayesFlow and Stan

**Authors:**
**Bruny Krijgsman from Erasmus University of Rotterdam**
**Michael D. Nunez from the Psychological Methods group at the University of Amsterdam**

### Citation
Krijgsman, B., Steeghs-Turchina, M., Nunez, M. D. (2026, April 29). [Which model better predicts brain-behavior relationships during decision-making? A simulation-based comparison of single-trial directed and integrative drift-diffusion models](https://osf.io/preprints/psyarxiv/7d46a_v2). PsyArXiv. https://doi.org/10.31234/osf.io/7d46a_v2

## Installation

### Prerequisites

- uv (https://docs.astral.sh/uv/getting-started/installation/)

### Install dependencies

- uv takes care of the dependencies (python, packages)

```bash
uv run python scripts/directed_ddm_generate.py
uv run python scripts/directed_ddm_fit.py
uv run python scripts/directed_ddm_analyze.py
```

```bash
uv run python scripts/directed_ddm_fit_factorial_parallel.py --prefix cross_
```

Running in background process:
```bash
nohup uv run python scripts/directed_ddm_fit_factorial_parallel.py --prefix cross_ &

tail -f nohup.out
```

## Integrative DDM training (BayesFlow + Keras/JAX)

This section explains how to run `scripts/integrative_ddm_train.py`, what it needs, and what it produces.

### What the script does
- Simulates data via BayesFlow for an integrative DDM + P300 model (no external dataset required).
- Trains a BayesFlow workflow for 500 epochs using Keras with the JAX backend.
- Saves a model checkpoint and a pickled training history.
- Generates diagnostic plots for simulated data and training loss.

### Outputs and where to find them
On first successful run, the script creates these directories if missing:
- `integrative_model/checkpoints`
- `integrative_model/figures`

Files created (SEED defaults to 12):
- Model checkpoint: `integrative_model/checkpoints/checkpoint_integrative_ddm_seed_12_150epochs_150epochs.keras`
- Training history (pickle): `integrative_model/checkpoints/training_history_integrative_ddm_seed_12_150epochs.pkl`
- Simulated data checks:
  - `integrative_model/figures/sim_rt_distribution.png`
  - `integrative_model/figures/sim_P300_distribution.png`
- Validation data checks:
  - `integrative_model/figures/val_rt_distribution.png`
  - `integrative_model/figures/val_P300_distribution.png`
- Training loss plots:
  - `integrative_model/figures/loss_with_val_improved_seed_12_150epochs.png`
  - `integrative_model/figures/loss_plot_seed_12_150epochs.png`

On subsequent runs, if the checkpoint and history exist, the script loads them and only regenerates plots/analysis.

To force retraining from scratch, delete the existing checkpoint and history first, for example:
```bash
rm -f integrative_model/checkpoints/checkpoint_integrative_ddm_seed_12_variable_trials.keras \
      integrative_model/checkpoints/checkpoint_integrative_ddm_seed_12_variable_trials.pkl
```

### Requirements
- Python: 3.11 (the project is configured for `>=3.11,<3.12`).
- Package manager: `uv` (recommended) or another PEP 621-compatible tool.

Runtime dependencies (installed automatically when using `uv` with this repo):
- bayesflow (>=2.0.3)
- keras (installed via BayesFlow; used with the JAX backend)
- jax (>=0.6.1) [CPU by default]
- tensorflow (>=2.19.0) [for Keras 3 saving utilities]
- numpy (<=2.2.0)
- scipy (>=1.15.3)
- numba (>=0.61.2)
- matplotlib (>=3.10.3)
- pandas, seaborn (used by analysis/plotting)

Note: The top-level `pyproject.toml` already declares the core dependencies. If you encounter `ImportError` for `pandas` or `seaborn`, install them as well.

### Environment setup
The script configures Keras to use the JAX backend internally:
- It sets `KERAS_BACKEND=jax` before importing `keras`.

No additional environment variables are required for CPU execution. Optional for GPU:
- JAX GPU: install an appropriate `jax[cuda]` build for your CUDA/cuDNN stack (not required for CPU-only training).

### How to run
Using `uv` (recommended):
```bash
cd single_trial_nddm_compare
# Run training script
uv run scripts/integrative_ddm_train.py
# Simulate data from integrative model in multiple conditions
uv run scripts/integrative_ddm_generate_factorial_new_sigma.py
# Check the simulated data
uv run scripts/integrative_ddm_data_check.py
# Generate recovery results for different conditions
uv run scripts/integrative_ddm_analyze_results_factorial.py
```

Alternative with a virtual environment:
```bash
cd single_trial_nddm_compare
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
# Run training script
python scripts/integrative_ddm_train.py
# Simulate data from integrative model in multiple conditions
python scripts/integrative_ddm_generate_factorial_new_sigma.py
# Check the simulated data
python scripts/integrative_ddm_data_check.py
# Generate recovery results for different conditions
python scripts/integrative_ddm_analyze_results_factorial.py
```

### Reproducibility and configuration
- Random seed is set to 12 for NumPy, TensorFlow, and JAX inside the script.
- Training configuration is defined in the script/workflow:
  - epochs: 150
  - batch_size: 64
  - num_batches_per_epoch: 200
  - validation_data size: 10,000 simulated datasets

To change the seed, epochs, or paths, edit `scripts/integrative_ddm_train.py` (look for `SEED`, `model_name`, `CHECKPOINT_PATH`).

### Troubleshooting
- Keras backend error: ensure the `KERAS_BACKEND=jax` is set before `import keras`. The script does this automatically.
- Missing packages (e.g., `pandas`, `seaborn`): install them with `uv add pandas seaborn` or `pip install pandas seaborn`.
- JAX installation issues: prefer CPU wheels if you don't need GPU. For GPU, follow the official JAX installation guide matching your CUDA/cuDNN.

