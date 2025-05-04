# =====================================================================================
# Import modules
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.approximators import ContinuousApproximator, Approximator
from bayesflow.simulators import make_simulator
from integrative_ddm_sim import ddm_prior, batch_simulator
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

# =====================================================================================
# Main training 

# Initialize networks
print("Initializing networks...")
summary_net = SummaryNetwork(input_shape=(100, 2))
inference_net = InferenceNetwork()
amortizer = ContinuousApproximator(
    adapter=summary_net,
    inference_network=inference_net,
)

# Define the optimizer and learning rate
learning_rate = 1e-4
optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.1)

# Create approximator instance
approximator = ContinuousApproximator(
    summary_network=summary_net,
    inference_network=inference_net,
    adapter=summary_net
)
# Compile approximator 
approximator.compile(optimizer=optimizer)

# Write single simulation function to pass into generative model
def full_simulator(batch_size = 32, n_obs = 100):
    params = ddm_prior(batch_size)
    sims = batch_simulator(params, n_obs)
    print(f"params shape: {params.shape}, dtype: {params.dtype}")
    print(f"sims shape: {sims.shape}, dtype: {sims.dtype}")
    return params, sims

# Use these in make_simulator
generative_model = make_simulator(full_simulator)

# Define varying trial count
def prior_N(n_min=60, n_max=120):
    return np.random.randint(n_min, n_max + 1)

# Define the checkpoint callback to save the model during training
checkpoint_callback = ModelCheckpoint(
    'checkpoints/integrative_ddm_epoch_{epoch:02d}.keras', 
    save_freq='epoch',  
    save_best_only=True,  
    verbose=1 
)

# Train the model
print("Training the model...")
losses = approximator.fit(
    epochs=500,
    num_batches=200,
    batch_size=32,
    simulator=generative_model,
    callbacks=[checkpoint_callback],
    iterations_per_epoch=1000,
    n_obs=100
)
print("Training complete.")

# After training, manually save the model in SavedModel format
approximator.save('checkpoints/integrative_ddm_model')

# Save training losses and plot
np.save('losses.npy', losses)

# Print final loss
print(f"Final loss: {losses[-1]}")
