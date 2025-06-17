# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf

SEED = 12
np.random.seed(SEED)
tf.random.set_seed(SEED)
jax_key = jax.random.PRNGKey(SEED)

# Import modules
import bayesflow as bf
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.approximators import ContinuousApproximator
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add JAX-specific imports for checkpointing
import keras
import os

# Import DDM simulator and plotting functions
import integrative_ddm_sim as ddm
import plots

# =====================================================================================
# Helper functions for training history management

def save_training_history(history, checkpoint_dir):
    """Save training history as pickle."""
    import pickle
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save as pickle - include seed in filename
    history_pickle_path = os.path.join(checkpoint_dir, f'training_history_seed{SEED}_mixed_new_sigma_beta.pkl')
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    return history_pickle_path

def load_training_history(checkpoint_dir, format='pickle'):
    """Load training history from saved files.
    
    Args:
        checkpoint_dir: Directory containing saved history files
        format: 'pickle', 'json', or 'csv'
    
    Returns:
        Dictionary containing training history
    """
    import pickle
    import json
    
    if format == 'pickle':
        history_path = os.path.join(checkpoint_dir, f'training_history_seed{SEED}_mixed_new_sigma_beta.pkl')
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    
    elif format == 'json':
        history_path = os.path.join(checkpoint_dir, f'training_history_seed{SEED}_mixed_new_sigma_beta.json')
        with open(history_path, 'r') as f:
            history_dict = json.load(f)
        # Convert lists back to numpy arrays
        for key, value in history_dict.items():
            if isinstance(value, list):
                history_dict[key] = np.array(value)
        return history_dict
    
    elif format == 'csv':
        history_path = os.path.join(checkpoint_dir, f'training_history_seed{SEED}_mixed_new_sigma_beta.csv')
        history_df = pd.read_csv(history_path)
        return history_df.to_dict('list')
    
    else:
        raise ValueError("Format must be 'pickle', 'json', or 'csv'")

# =====================================================================================
# Setup paths and directories

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set checkpoint path relative to current file - include seed in filename
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'checkpoints', f'jax_simple_integrative_ddm_checkpoint_seed{SEED}_mixed_new_sigma_beta.keras')
# Create checkpoint directory if it doesn't exist
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# =====================================================================================
# Main training 

# Initialize networks
print("Initializing networks...")

# Meta function for simulator
def meta():
    return dict(n_obs=100)

# Use these in make_simulator
print("Making simulator...")
simulator = make_simulator([ddm.prior, ddm.likelihood], meta_fn=meta)

# Simulation data setup 
sim_draws = simulator.sample(100)
# ddm.simulated_data_check(sim_draws)

print("Setting up validation data...")
val_data_size = 10000
val_data = simulator.sample(val_data_size) 
#ddm.simulated_data_check(val_data)

summary_network = bf.networks.SetTransformer(summary_dim=8)
inference_network = bf.networks.CouplingFlow()
adapter = (
    Adapter()
    .broadcast("n_obs", to="choicert")    
    .as_set(["choicert", "z"])
    .standardize(exclude=["n_obs"])
    .convert_dtype("float64", "float32")
    .concatenate(["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"], into="inference_variables")
    .concatenate(["choicert", "z"], into="summary_variables")
    .rename("n_obs", "inference_conditions")
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_network,
    summary_network=summary_network,
)

# Adapted summary_variables and inference_variables
adapted_sim_draws = adapter(sim_draws)
adapted_val_data = adapter(val_data)
print("Adapted inference_variables shape:", adapted_sim_draws["inference_variables"].shape)
print("Adapted summary_variables dtype:", adapted_sim_draws["summary_variables"].dtype)
print("Adapted inference_variables dtype:", adapted_sim_draws["inference_variables"].dtype)

# Check for existing checkpoint and history
history_pickle_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), f'training_history_seed{SEED}_mixed_new_sigma_beta.pkl')
checkpoint_exists = os.path.exists(CHECKPOINT_PATH)
history_exists = os.path.exists(history_pickle_path)

if checkpoint_exists and history_exists:
    print("Loading checkpoint and training history...")
    approximator = keras.saving.load_model(CHECKPOINT_PATH)
    history_dict = load_training_history(os.path.dirname(CHECKPOINT_PATH), format='pickle')
    print("Loaded existing model and training history.")
elif checkpoint_exists:
    print("Loading checkpoint (no history found)...")
    approximator = keras.saving.load_model(CHECKPOINT_PATH)
    history_dict = None
    print("Loaded existing model, but no training history found.")
else:
    print("No checkpoint found, creating new approximator...")

    history = workflow.fit_online(
        epochs=150, 
        batch_size=64, 
        num_batches_per_epoch=200, 
        validation_data=10000
    )

    print("Training complete.")

    # Save training history
    history_pickle_path = save_training_history(history, os.path.dirname(CHECKPOINT_PATH))
    print(f"Saved training history as pickle: {history_pickle_path}")
    
    # Save the model
    workflow.approximator.save(CHECKPOINT_PATH)
    print(f"Saved model checkpoint: {CHECKPOINT_PATH}")
    
    # Convert to dictionary format for consistent plotting
    history_dict = history.history

# =====================================================================================
# Plotting and Analysis (always executed after loading or training)

# Use the extracted plotting functions
results = plots.generate_training_plots_and_analysis(history_dict, SEED, CURRENT_DIR, verbose=True)
