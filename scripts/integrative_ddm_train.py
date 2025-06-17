# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

from pathlib import Path
import sys
import jax
import numpy as np
import tensorflow as tf

SEED = 12
np.random.seed(SEED)
tf.random.set_seed(SEED)
jax_key = jax.random.PRNGKey(SEED)

import keras

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from integrative_model.analysis import simulated_data_check, plot_training_history, analyze_training_performance, plot_bayesflow_loss
from integrative_model.workflow import create_workflow
from integrative_model.training_history import save_training_history, load_training_history

# =====================================================================================
# Setup paths and directories

# Get the directory of the current file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIVE_MODEL_DIR = PROJECT_ROOT / "integrative_model"
CHECKPOINTS_DIR = INTEGRATIVE_MODEL_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = INTEGRATIVE_MODEL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define a suffix for model and history files
model_name = f"integrative_ddm_seed_{SEED}"

# Set checkpoint path relative to current file - include seed in filename
CHECKPOINT_PATH = CHECKPOINTS_DIR / f"checkpoint_{model_name}.keras"

# =====================================================================================
# Main training 

# Initialize workflow
workflow, simulator, adapter = create_workflow()

# Simulation data setup 
sim_draws = simulator.sample(100)
simulated_data_check(sim_draws, 
                    FIGURES_DIR / 'sim_rt_distribution.png',
                    FIGURES_DIR / 'sim_P300_distribution.png')

print("Setting up validation data...")
val_data_size = 10000
val_data = simulator.sample(val_data_size) 
simulated_data_check(val_data, 
                    FIGURES_DIR / 'val_rt_distribution.png',
                    FIGURES_DIR / 'val_P300_distribution.png')

# Adapted summary_variables and inference_variables
adapted_sim_draws = adapter(sim_draws)
adapted_val_data = adapter(val_data)
print("Adapted inference_variables shape:", adapted_sim_draws["inference_variables"].shape)
print("Adapted summary_variables dtype:", adapted_sim_draws["summary_variables"].dtype)
print("Adapted inference_variables dtype:", adapted_sim_draws["inference_variables"].dtype)

# Check for existing checkpoint and history
history_pickle_path = CHECKPOINTS_DIR / f"training_history_{model_name}.pkl"
checkpoint_exists = CHECKPOINT_PATH.exists()
history_exists = history_pickle_path.exists()

if checkpoint_exists and history_exists:
    print("Loading checkpoint and training history...")
    approximator = keras.saving.load_model(CHECKPOINT_PATH)
    history_dict = load_training_history(history_pickle_path)
    print("Loaded existing model and training history.")
elif checkpoint_exists:
    print("Loading checkpoint (no history found)...")
    approximator = keras.saving.load_model(CHECKPOINT_PATH)
    history_dict = None
    print("Loaded existing model, but no training history found.")
else:
    print("No checkpoint found, creating new approximator...")

    history = workflow.fit_online(
        epochs=200, 
        batch_size=64, 
        num_batches_per_epoch=200, 
        validation_data=10000
    )

    print("Training complete.")

    # Save training history
    save_training_history(history, history_pickle_path)
    print(f"Saved training history as pickle: {history_pickle_path}")
    
    # Save the model
    workflow.approximator.save(CHECKPOINT_PATH)
    print(f"Saved model checkpoint: {CHECKPOINT_PATH}")
    
    # Convert to dictionary format for consistent plotting
    history_dict = history.history

# =====================================================================================
# Plotting and Analysis (always executed after loading or training)

# Generate all training plots and analysis directly
if history_dict is None:
    print("No training history available for plotting.")
else:
    print("Generating plots and analysis...")
    
    # Create Figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate main training plot
    main_plot_path = plot_training_history(history_dict, 
                                          str(FIGURES_DIR / f'loss_with_val_improved_seed_{SEED}.png'))
    results['main_plot_path'] = main_plot_path
    print(f"Saved improved loss plot: {main_plot_path}")
    
    # Perform analysis
    analysis_results = analyze_training_performance(history_dict, verbose=True)
    results['analysis'] = analysis_results
    
    # Generate bayesflow plot
    bf_plot_path = plot_bayesflow_loss(history_dict, 
                                      str(FIGURES_DIR / f'loss_plot_seed_{SEED}.png'))
    results['bayesflow_plot_path'] = bf_plot_path
    print(f"Saved bayesflow loss plot: {bf_plot_path}")
