import numpy as np
import pandas as pd

def get_data(filename):
    df = pd.read_csv(filename)
    return df

# Expiration percentage per category
daily_expiration_percentages = {
    "Fresh Produce": 35,
    "Dairy": 25,
    "Meat and Fish": 20,
    "Bakery": 15,
    "Pantry Items": 4,
    "Other": 1
}

average_expiry_weight = 66  # kg per day (total)
std_by_category = 10  # kg (total std for all categories)

# Convert percentages into actual means (in kg)
category_means = {
    key: (value / 100) * average_expiry_weight
    for key, value in daily_expiration_percentages.items()
}

# Distribute std proportionally across categories
total_percentage = sum(daily_expiration_percentages.values())  # Should be 100%
category_stds = {
    key: (value / total_percentage) * std_by_category
    for key, value in daily_expiration_percentages.items()
}

def calculate_shape_and_scale(mean, std):
    if std == 0:
        raise ValueError("Standard deviation cannot be zero.")

    supplier_shape = (mean ** 2) / (std ** 2)  # Shape parameter
    supplier_scale = (std ** 2) / mean  # Scale parameter

    return supplier_shape, supplier_scale

def generate_supplier_samples(size, mean, std):
    supplier_shape, supplier_scale = calculate_shape_and_scale(mean, std)
    return np.random.gamma(supplier_shape, supplier_scale, size)

def get_distributions():
    distributions = {}

    for category, mean in category_means.items():
        std = category_stds[category]  # Get standard deviation for this category
        supplier_shape, supplier_scale = calculate_shape_and_scale(mean, std)
        distributions[category] = np.random.gamma(supplier_shape, supplier_scale, 1000)

    return distributions

# Example usage
if __name__ == "__main__":

    df = get_data('./data_processing/data.csv')
    print(df.head(10))

    '''
    # Generate supplier samples for a specific category
    fresh_produce_mean = category_means["Fresh Produce"]
    fresh_produce_std = category_stds["Fresh Produce"]
    supplier_samples = generate_supplier_samples(1000, fresh_produce_mean, fresh_produce_std)
    
    # Generate distributions for all categories
    category_distributions = get_distributions()
    
    print("Sample from Fresh Produce:", supplier_samples[:5])
    print("Generated distributions for all categories.")
    '''