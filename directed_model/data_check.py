from scipy.io import loadmat
import matplotlib.pyplot as plt


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'ddmdata_SNR_low_COUP_high_DIST_gaussian.mat')
data = loadmat(data_path)
print(data.keys())

# Check the variables you care about
print('sigma_z:', data['sigma_z'])
print('mu_z:', data['mu_z'])
print('lambda_param:', data['lambda_param'])

# Plot histogram of sigma_z
plt.hist(data['sigma_z'].flatten(), bins=50, edgecolor='black')
plt.title('sigma_z distribution')
plt.xlabel('sigma_z')
plt.ylabel('Count')
plt.grid(True)
plt.show()

