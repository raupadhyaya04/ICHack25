import numpy as np
from scipy import stats
import pandas as pd

def get_data(filename):
    df = pd.read_csv(filename)
    return df


# Assumption Constants
supplier_mean__expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4
supplier_mean_product_expiry = 4

supplier_std_product_expiry = 10

distributer_mean_dc = 50
distributer_std_dc = 50

# Supplier distribution (Erlang)
# Erlang is a special case of Gamma distribution where shape parameter (k) is an integer
# We need to calculate shape (k) and scale (θ) parameters from mean and std
supplier_shape = int((supplier_mean_product_expiry ** 2) / (supplier_std_product_expiry ** 2))  # rounded to nearest integer
supplier_scale = supplier_std_product_expiry ** 2 / supplier_mean_product_expiry

# Distributor distribution (Negative Binomial)
# We need to calculate n and p parameters from mean and std
# For negative binomial: mean = n(1-p)/p and variance = n(1-p)/p^2
p = distributer_mean_dc / (distributer_std_dc ** 2)  # probability of success
n = distributer_mean_dc * p / (1 - p)  # number of successes

def generate_supplier_samples(size=1000):
    """Generate samples from supplier's Erlang distribution"""
    return np.random.gamma(supplier_shape, supplier_scale, size)

def generate_distributer_samples(size=1000):
    """Generate samples from distributor's Negative Binomial distribution"""
    return np.random.negative_binomial(n, p, size)

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    supplier_samples = generate_supplier_samples(1000)
    distributer_samples = generate_distributer_samples(1000)
    
    # Print summary statistics
    print("\nSupplier Distribution (Erlang):")
    print(f"Target Mean: {supplier_mean_product_expiry}")
    print(f"Actual Mean: {np.mean(supplier_samples):.2f}")
    print(f"Target Std: {supplier_std_product_expiry}")
    print(f"Actual Std: {np.std(supplier_samples):.2f}")
    
    print("\nDistributor Distribution (Negative Binomial):")
    print(f"Target Mean: {distributer_mean_dc}")
    print(f"Actual Mean: {np.mean(distributer_samples):.2f}")
    print(f"Target Std: {distributer_std_dc}")
    print(f"Actual Std: {np.std(distributer_samples):.2f}")


    # Can we use local populations




    