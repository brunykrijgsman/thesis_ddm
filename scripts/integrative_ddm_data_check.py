import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Get data path
data_dir = PROJECT_ROOT / 'integrative_model' / 'data'

# Load one example data file to examine structure
example_file = 'integrative_ddm_data_SNR_low_COUP_high_DIST_gaussian.mat'
data_path = data_dir / example_file
data = loadmat(data_path)

print(f"Loading data from: {example_file}")
print("Available variables:", list(data.keys()))

# Check key variables
print('=== Model Parameters ===')
print('alpha shape:', data['alpha'].shape, 'mean:', np.mean(data['alpha']))
print('tau shape:', data['tau'].shape, 'mean:', np.mean(data['tau']))
print('beta shape:', data['beta'].shape, 'mean:', np.mean(data['beta']))
print('mu_delta shape:', data['mu_delta'].shape, 'mean:', np.mean(data['mu_delta']))
print('eta_delta shape:', data['eta_delta'].shape, 'mean:', np.mean(data['eta_delta']))
print('gamma shape:', data['gamma'].shape, 'mean:', np.mean(data['gamma']))
print('sigma shape:', data['sigma'].shape, 'mean:', np.mean(data['sigma']))
print()

print('=== Data Variables ===')
print('rt shape:', data['rt'].shape, 'mean:', np.mean(data['rt']))
print('acc shape:', data['acc'].shape, 'mean:', np.mean(data['acc']))
print('z (P300) shape:', data['z'].shape, 'mean:', np.mean(data['z']))
print('participant shape:', data['participant'].shape, 'unique participants:', len(np.unique(data['participant'])))
print()

print('=== Condition Info ===')
print('condition:', data['condition'][0] if 'condition' in data else 'N/A')
print('nparts:', data['nparts'][0][0] if 'nparts' in data else 'N/A')
print('ntrials:', data['ntrials'][0][0] if 'ntrials' in data else 'N/A')
print('N (total observations):', data['N'][0][0] if 'N' in data else 'N/A')
print()

# Create visualization plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Data Check: {example_file}', fontsize=14)

# Plot 1: Gamma distribution (coupling parameter)
axes[0, 0].hist(data['gamma'].flatten(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Gamma (DDM-P300 coupling)')
axes[0, 0].set_xlabel('gamma')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sigma distribution (P300 noise)
axes[0, 1].hist(data['sigma'].flatten(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Sigma (P300 noise)')
axes[0, 1].set_xlabel('sigma')
axes[0, 1].set_ylabel('Count')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: P300 (z) distribution
axes[0, 2].hist(data['z'].flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_title('P300 Signal (z)')
axes[0, 2].set_xlabel('z')
axes[0, 2].set_ylabel('Count')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: RT distribution
axes[1, 0].hist(data['rt'].flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Reaction Time')
axes[1, 0].set_xlabel('RT (seconds)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Accuracy distribution
axes[1, 1].hist(data['acc'].flatten(), bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Accuracy')
axes[1, 1].set_xlabel('Accuracy (0/1)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Alpha distribution (drift rate)
axes[1, 2].hist(data['alpha'].flatten(), bins=30, edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Alpha (drift rate)')
axes[1, 2].set_xlabel('alpha')
axes[1, 2].set_ylabel('Count')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()

# Compare across conditions
print('\n=== Comparing Across Conditions ===')
conditions = [
    'integrative_ddm_data_SNR_low_COUP_low_DIST_gaussian.mat',
    'integrative_ddm_data_SNR_low_COUP_high_DIST_gaussian.mat',
    'integrative_ddm_data_SNR_high_COUP_low_DIST_gaussian.mat',
    'integrative_ddm_data_SNR_high_COUP_high_DIST_gaussian.mat',
    'integrative_ddm_data_SNR_low_COUP_low_DIST_laplace.mat',
    'integrative_ddm_data_SNR_low_COUP_high_DIST_laplace.mat',
    'integrative_ddm_data_SNR_high_COUP_low_DIST_laplace.mat',
    'integrative_ddm_data_SNR_high_COUP_high_DIST_laplace.mat',
    'integrative_ddm_data_SNR_low_COUP_low_DIST_uniform.mat',
    'integrative_ddm_data_SNR_low_COUP_high_DIST_uniform.mat',
    'integrative_ddm_data_SNR_high_COUP_low_DIST_uniform.mat',
    'integrative_ddm_data_SNR_high_COUP_high_DIST_uniform.mat'
]

gamma_means = []
sigma_means = []
rt_means = []
acc_means = []

for condition in conditions:
    if (data_dir / condition).exists():
        cond_data = loadmat(data_dir / condition)
        gamma_means.append(np.mean(cond_data['gamma']))
        sigma_means.append(np.mean(cond_data['sigma']))
        rt_means.append(np.mean(cond_data['rt']))
        acc_means.append(np.mean(cond_data['acc']))
        print(f"{condition}: gamma={np.mean(cond_data['gamma']):.3f}, sigma={np.mean(cond_data['sigma']):.3f}, RT={np.mean(cond_data['rt']):.3f}, Acc={np.mean(cond_data['acc']):.3f}")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Condition Comparisons (Gaussian Error Distribution)', fontsize=14)

condition_labels = ['Low SNR\nLow Coup', 'Low SNR\nHigh Coup', 'High SNR\nLow Coup', 'High SNR\nHigh Coup']

axes[0, 0].bar(range(len(gamma_means)), gamma_means)
axes[0, 0].set_title('Mean Gamma (Coupling)')
axes[0, 0].set_xticks(range(len(condition_labels)))
axes[0, 0].set_xticklabels(condition_labels, rotation=45)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(range(len(sigma_means)), sigma_means)
axes[0, 1].set_title('Mean Sigma (Noise)')
axes[0, 1].set_xticks(range(len(condition_labels)))
axes[0, 1].set_xticklabels(condition_labels, rotation=45)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].bar(range(len(rt_means)), rt_means)
axes[1, 0].set_title('Mean Reaction Time')
axes[1, 0].set_xticks(range(len(condition_labels)))
axes[1, 0].set_xticklabels(condition_labels, rotation=45)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(range(len(acc_means)), acc_means)
axes[1, 1].set_title('Mean Accuracy')
axes[1, 1].set_xticks(range(len(condition_labels)))
axes[1, 1].set_xticklabels(condition_labels, rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.show() 