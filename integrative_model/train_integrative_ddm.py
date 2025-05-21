# Force CPU on macOS before any TensorFlow usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Must be here right after import

# Now import everything else
from bayesflow.networks import InferenceNetwork
from bayesflow.approximators import ContinuousApproximator
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
from integrative_ddm_sim import ddm_prior, batch_simulator
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
import atexit

# Force CPU on macOS before any TensorFlow usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Minimize TensorFlow logs

# TensorFlow session cleanup
def cleanup():
    print("Cleaning up resources...")
    tf.keras.backend.clear_session()
    sys.stdout.flush()  # Ensure all prints are flushed before exit

atexit.register(cleanup)

# =====================================================================================
# Main training 

print("Initializing networks...")

# Custom summary network
class MySummaryNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(64, return_sequences=False)
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(7, activation='relu')  

    def call(self, inputs):
        tf.print("SummaryNet input shape:", tf.shape(inputs))  # Shows actual shape at runtime

        # Check that inputs are correctly shaped (should be rank 4)
        tf.debugging.assert_rank(inputs, 4, message="Expected input rank 4: (batch_outer, batch_inner, time, features)")

        # Use static shapes when possible
        batch_outer, batch_inner, time_steps, features = inputs.shape

        if None in [batch_outer, batch_inner, time_steps, features]:
            # Fallback to dynamic shape if static shape is missing
            shape = tf.shape(inputs)
            batch_outer = shape[0]
            batch_inner = shape[1]
            time_steps = shape[2]
            features = shape[3]

            x = tf.reshape(inputs, (batch_outer * batch_inner, time_steps, features))
            x = self.lstm(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return self.output_layer(x)  # Make sure to return the output
        else:
            # Handle the case where the input shape is static
            x = tf.reshape(inputs, (batch_outer * batch_inner, time_steps, features))
            x = self.lstm(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return self.output_layer(x)  # Make sure to return the output

    def compute_metrics(self, summary_variables, stage=None):
        outputs = self(summary_variables)
        return {"outputs": outputs}

# Custom inference network
class MyInferenceNetwork(InferenceNetwork):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(7)

    def build(self, xz_shape):
        print("xz_shape:", xz_shape)  # Debugging the shape value

        # Ensure xz_shape[-1] is a valid integer and not a tensor
        feature_dim = int(xz_shape[-1])  # Convert to integer if necessary
        self.dense1.build((None, feature_dim))  # Build layer with the correct shape
        self.dense2.build((None, self.dense1.units))
        self.output_layer.build((None, self.dense2.units))
        super().build(xz_shape)

    def call(self, xz, conditions=None, training=False):
        print("InferenceNet input shape (raw):", tf.shape(xz))
        if len(xz.shape) == 3:
            xz = tf.reshape(xz, (-1, xz.shape[-1]))
        tf.debugging.assert_equal(xz.shape[-1], 7, message="Expected 7 summary features.")
        x = self.dense1(xz)
        x = self.dense2(x)
        return self.output_layer(x)

# Initialize networks
summary_net = MySummaryNet()
inference_net = MyInferenceNetwork()

# Dummy pass to build networks
dummy_summary_input = tf.zeros((32, 32, 100, 2), dtype=tf.float32)
summary_net.build(dummy_summary_input.shape)  # Build the summary network with shape
summary_output = summary_net(dummy_summary_input)  # Ensure output is computed

# Add a check to confirm summary_output is valid
if summary_output is None:
    raise ValueError("Summary network did not return a valid output.")

# Get the actual output shape of the summary network
output_shape_summary_net = tf.shape(summary_output)  # This gets the shape of the summary network output

# Now use the output shape from the summary network to build the inference network
inference_net.build(output_shape_summary_net)  # Build the inference network using the correct shape

# Adapter setup
adapter = (
    Adapter()
    .to_array()
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .standardize()
    .rename("sims", "summary_variables")
    .rename("params", "inference_variables")
)

# Optimizer
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, clipnorm=1.1)

# Approximator
approximator = ContinuousApproximator(
    summary_network=summary_net,
    inference_network=inference_net,
    adapter=adapter
)

approximator.compile(optimizer=optimizer)

# Simulator
def full_simulator(batch_size=32, n_obs=100):
    params = ddm_prior(batch_size).astype(np.float32)
    sims = batch_simulator(params, n_obs).astype(np.float32)
    return {
        "params": params,
        "sims": sims
    }

generative_model = make_simulator(full_simulator)

# Trial count prior
# def prior_N(n_min=60, n_max=120):
#    return np.random.randint(n_min, n_max + 1)


# Optional gradient flow check
sample_batch = full_simulator(batch_size=2, n_obs=10)
adapted = adapter(sample_batch)

x = adapted["summary_variables"]        # shape: (2, 10, 2)
x = tf.expand_dims(x, axis=0)           # shape: (1, 2, 10, 2)

with tf.GradientTape() as tape:
    summary_out = summary_net(x)
    inf_out = inference_net(summary_out)
    target = tf.random.normal(tf.shape(inf_out), dtype=tf.float32)
    dummy_loss = tf.reduce_mean(tf.square(inf_out - target))

grads = tape.gradient(dummy_loss, approximator.trainable_variables)

if any(g is None for g in grads):
    print("Gradients:", grads)
    raise RuntimeError("Gradient flow broken: Some gradients are None.")
else:
    print("Gradient flow check passed.")

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    'checkpoints/integrative_ddm_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    save_best_only=True,
    verbose=1
)

# Print trainable variables
print("Trainable variables:")
for var in approximator.trainable_variables:
    print(var.name, var.shape)

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

# Save model
approximator.save('checkpoints/integrative_ddm_model')

# Save and report losses
np.save('losses.npy', losses)
print(f"Final loss: {losses[-1]}")