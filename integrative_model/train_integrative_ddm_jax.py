# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# Import modules
import bayesflow as bf
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.approximators import ContinuousApproximator
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Add JAX-specific imports for checkpointing
import keras
import os

# Import DDM simulator
import integrative_ddm_sim as ddm

# Set checkpoint path
CHECKPOINT_PATH = 'checkpoints/jax_integrative_ddm_checkpoint.keras'

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

sim_draws = simulator.sample(100)
print("Simulator draws keys:", sim_draws.keys())
ddm.simulated_data_check(sim_draws)

# summary_network = MySummaryNet()
summary_network = bf.networks.SetTransformer(summary_dim=10)

# inference_network = MyInferenceNetwork()
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

# Adapted summary_variables and inference_variables
adapted = adapter(sim_draws)
print("Adapted summary_variables shape:", adapted["summary_variables"].shape)
print("Adapted inference_variables shape:", adapted["inference_variables"].shape)
print("Adapted summary_variables dtype:", adapted["summary_variables"].dtype)
print("Adapted inference_variables dtype:", adapted["inference_variables"].dtype)

# Use Keras optimizer with JAX backend instead of optax directly
optimizer = keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.1)

if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    approximator = keras.saving.load_model(CHECKPOINT_PATH)
else:
    print("No checkpoint found, creating new approximator...")

    # Create and compile approximator
    approximator = ContinuousApproximator(
        summary_network=summary_network,
        inference_network=inference_network,
        adapter=adapter
    )

    # Compile approximator 
    print("Compiling approximator...")
    approximator.compile(optimizer=optimizer)

    # Train the model
    print("Training the model...")
    history = approximator.fit(
        epochs=50, # TEMP: only 50 epochs instead of 500
        num_batches=200,
        batch_size=32,
        simulator=simulator,
        iterations_per_epoch=1000,
        n_obs=100
    )
    print("Training complete.")

    plot = bf.diagnostics.plots.loss(
        history=history
    )
    plot.savefig('Figures/loss_plot.png')

    # Create checkpoint directory if it doesn't exist
    approximator.save(CHECKPOINT_PATH)
