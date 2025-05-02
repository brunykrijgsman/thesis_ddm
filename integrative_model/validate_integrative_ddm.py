# =====================================================================================
from bayesflow.diagnostics import diagnostics
from model_components import prior, batch_simulator
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.approximators import ContinuousApproximator

# Reload trainer amortizer
summary_net = SummaryNetwork()
inference_net = InferenceNetwork({'n_params': 7})
amortizer = ContinuousApproximator(inference_net, summary_net)
amortizer.load('checkpoints/integrative_ddm')

# Load trained model
model = bayesflow.models.BayesianModel(prior, batch_simulator)
model.load_weights('checkpoints/integrative_ddm')

# Generate validation data
n_param_sets = 1000
n_samples = 1000
n_trials = 100

true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_trials).astype(np.float32)
param_samples = amortizer.samples(x, n_samples)
param_means = param_samples.mean(axis=0)

param_names = ['alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma']

# Plot parameter means
for i, param_name in enumerate(param_names):
    plt.figure(figsize=(10, 5))
    plt.hist(param_means[:, i], bins=30, edgecolor='black')
    plt.title(f'{param_name} distribution')
    plt.show()

diagnostics(true_params, param_means, param_names, filename='results/integrative_ddm_diagnostics')