# %%
import pandas as pd

# %%
data = pd.read_csv('final_data.csv', index_col=0)

# %%
data

# %%
filtered = data[data["type"] == "Foodbank"].reset_index()
filtered

# %%
import matplotlib.pyplot as plt

plt.plot(filtered["demand"])

# %%
filtered["demand"].describe()

# %%
# delete "lat", "lng", "type","demand", "supply", "waste" columns
filtered.drop(columns=["lat", "lng", "type","demand", "supply", "waste"], inplace=True)

# %%
filtered

# %%
filtered["name"].to_list()

# %%
import numpy as np
from scipy.stats import gamma
import pandas as pd

# Parameters
N = 249                             # Total number of samples
p_zero = 0.85                       # Proportion of zeros
desired_total_sum = 25000           # Desired total sum

# Initialize 'filtered' DataFrame with 'name' column
# Assume 'names_list' is your list of names corresponding to each sample
names_list = [f"Sample {i+1}" for i in range(N)]  # Replace with your actual names

filtered = pd.DataFrame({'name': filtered["name"].to_list()})

# Compute adjusted mean and standard deviation
mu_total = desired_total_sum / N    # New mean to achieve the desired total sum
original_CV = 680.29 / 166.71       # Original coefficient of variation (CV ≈ 4.08)
sigma_total = original_CV * mu_total
sigma_total_sq = sigma_total ** 2

# Precompute constants outside the loop
n_zeros = int(p_zero * N)           # Number of zeros
n_nonzeros = N - n_zeros            # Number of non-zeros
p_nonzero = 1 - p_zero
mu_zero = 0

# Mean of non-zero data
total_sum = mu_total * N            # Total sum of data
mu_nonzero = total_sum / n_nonzeros

# Variance of non-zero data using the law of total variance
delta_mu_zero = mu_zero - mu_total
delta_mu_nonzero = mu_nonzero - mu_total

sigma_zero_sq = 0                   # Variance of zeros is zero
sigma_nonzero_sq = (
    sigma_total_sq
    - p_zero * delta_mu_zero ** 2
    - p_nonzero * delta_mu_nonzero ** 2
) / p_nonzero

sigma_nonzero = np.sqrt(sigma_nonzero_sq)

# Gamma distribution parameters
shape = (mu_nonzero ** 2) / sigma_nonzero_sq
scale = sigma_nonzero_sq / mu_nonzero

# Generate zeros and gamma-distributed non-zero values outside the loop
zeros = np.zeros(n_zeros)
max_value = 6640  # Maximum value to clip the gamma samples

for i in range(366):
    # Generate gamma-distributed non-zero values for the day
    gamma_samples = gamma.rvs(a=shape, scale=scale, size=n_nonzeros)

    # Clip values to the maximum value
    gamma_samples = np.clip(gamma_samples, a_min=None, a_max=max_value)

    # Combine zeros and gamma samples
    data = np.concatenate([zeros, gamma_samples])

    # Shuffle the data
    np.random.shuffle(data)

    # Assign data to the DataFrame, ensuring the 'name' column is preserved
    filtered[f"Day {i+1}"] = data

    # Optional: Print the total sum for each day to verify
    print(f"Day {i+1} Total Sum: {np.sum(data):.2f}")

# Optional: Display the first few rows of the DataFrame
print(filtered.head())


# %%
filtered

# %%
filtered.drop(columns=["Day 366"], inplace=True)

# %%
supermarkets = pd.read_csv('generated_timeseries.csv', index_col=0)
supermarkets["waste"] = supermarkets["excess_kg"] * 1.7798
supermarkets.drop(columns=["excess_kg"], inplace=True)

supermarkets["waste"] = supermarkets["waste"] / 70

# %%
data = pd.read_csv('final_data.csv', index_col=0)
filtered_up = data[data["type"] == "Supermarket"].reset_index()
filtered_up["waste"] = filtered_up["waste"] / 45

# %%
import pandas as pd
import numpy as np

# Assuming you have a DataFrame 'filtered_up' with 'name' and 'waste' columns
# For illustration, here's how you might define 'filtered_up':
# filtered_up = pd.DataFrame({
#     'name': ['Store A', 'Store B', 'Store C', 'Store D'],
#     'waste': [100, 200, 0, 150]  # Notice that 'Store C' has a mean waste of 0
# })

# Extract names and waste values
names = filtered_up['name'].tolist()
waste_means = filtered_up['waste'].values  # Mean for each name

# Number of days
N_days = 365

# Initialize the DataFrame with the 'name' column
data = pd.DataFrame({'name': names})

# Calculate variances:
# - For entries where waste_mean is 0, variance is set to 5
# - For other entries, variance is 0.1 * waste_mean
waste_variances = np.where(waste_means == 0, 5, 0.1 * waste_means)

# Standard deviations are the square root of variances
waste_stddevs = np.sqrt(waste_variances)

# Generate samples for each day
for day in range(1, N_days + 1):
    day_name = f'Day {day}'
    # Sample from normal distribution for each name
    samples = np.random.normal(loc=waste_means, scale=waste_stddevs)
    # Ensure no negative values (since waste can't be negative)
    samples = np.clip(samples, a_min=0, a_max=None)
    # Add the samples to the DataFrame
    data[day_name] = samples

# Display the first few rows of the DataFrame


# %%
data

# %%
filtered

# %%
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger


lambda_penalty = 0.4
distance_penalty = 0.2


def milp_optimization(supply, demand, distance):
    model = LpProblem("MIP_Optimization", LpMaximize)
    x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat=LpInteger) for i in supply for j in demand}

    model += lpSum(x[i, j] for i in supply for j in demand) \
             - lambda_penalty * lpSum((supply[i] - lpSum(x[i, j] for j in demand)) for i in supply) \
             - distance_penalty * lpSum(distance[i, j] * x[i, j] for i in supply for j in demand)

    for i in supply:
        model += lpSum(x[i, j] for j in demand) <= supply[i]
    for j in demand:
        model += lpSum(x[i, j] for i in supply) <= demand[j]

    model.solve()
    output = {}
    for key in x:
        val = x[key].varValue
        #if val != 0.0:
        output[key] = val
    return output

# %%
geospatial_data = pd.read_csv('final_data.csv', index_col=0)
geospatial_data = geospatial_data.drop(columns=["demand", "supply", "waste"])

# %%
geospatial_data

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger

# Set penalty parameters
lambda_penalty = 0.4
distance_penalty = 2.0

# Define the optimization function
def milp_optimization(supply, demand, distance_dict):
    # Create the model
    model = LpProblem("MIP_Optimization", LpMaximize)
    
    # Decision variables: Amount allocated from supermarket i to food bank j
    x = {
        (i, j): LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in supply for j in demand
    }
    
    # Objective function
    model += (
        lpSum(x[i, j] for i in supply for j in demand)
        - lambda_penalty * lpSum(
            (supply[i] - lpSum(x[i, j] for j in demand)) for i in supply
        )
        - distance_penalty * lpSum(
            distance_dict.get((i, j), 0) * x[i, j] for i in supply for j in demand
        )
    )
    
    # Supply constraints: Total allocations from a supermarket cannot exceed its supply
    for i in supply:
        model += lpSum(x[i, j] for j in demand) <= supply[i]
    
    # Demand constraints: Total allocations to a food bank cannot exceed its demand
    for j in demand:
        model += lpSum(x[i, j] for i in supply) <= demand[j]
    
    # Solve the model
    model.solve()
    
    # Extract the results
    output = {}
    for (i, j) in x:
        val = x[(i, j)].varValue
        if val > 0:
            output[(i, j)] = val
    return output

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points.
    Input coordinates are in decimal degrees.
    Output distance is in kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

# Assuming 'geospatial_data' DataFrame is already loaded
# Ensure 'name' is a column; reset index if necessary
geospatial_data = geospatial_data.reset_index(drop=False)

# Clean names to remove leading/trailing whitespace
geospatial_data['name'] = geospatial_data['name'].str.strip()

# Separate supermarkets and food banks
supermarkets = geospatial_data[geospatial_data['type'] == 'Supermarket'].copy()
foodbanks = geospatial_data[geospatial_data['type'] == 'Foodbank'].copy()

# Compute distances between every supermarket and every food bank
# Prepare arrays for vectorized computation
supermarkets_coords = supermarkets[['name', 'lat', 'lng']].reset_index(drop=True)
foodbanks_coords = foodbanks[['name', 'lat', 'lng']].reset_index(drop=True)

# Create a list to store distances
distance_records = []

# Compute distances
for idx_sup, sup_row in supermarkets_coords.iterrows():
    sup_name = sup_row['name']
    sup_lat = sup_row['lat']
    sup_lng = sup_row['lng']
    
    # Compute distances to all food banks
    distances = haversine_distance(
        sup_lat,
        sup_lng,
        foodbanks_coords['lat'].values,
        foodbanks_coords['lng'].values
    )
    
    # Store distances
    for idx_fb, fb_row in foodbanks_coords.iterrows():
        fb_name = fb_row['name']
        distance = distances[idx_fb]
        distance_records.append({
            'supermarket': sup_name,
            'foodbank': fb_name,
            'distance_km': distance
        })

# Convert to DataFrame
distance_df = pd.DataFrame(distance_records)

# Create a distance dictionary for quick lookup
distance_dict = distance_df.set_index(['supermarket', 'foodbank'])['distance_km'].to_dict()



N_days = 30

# Run the optimization for each day
results = {}  # To store results for each day

for day in range(1, N_days + 1):
    day_name = f'Day {day}'
    print(f"Processing {day_name}...")
    
    # Prepare supply for the day
    supply_series = data.set_index('name')[day_name]
    supply_series = supply_series[supply_series > 0]  # Filter out zero or negative supplies
    supply = supply_series.to_dict()
    
    # Prepare demand for the day
    demand_series = filtered.set_index('name')[day_name]
    demand_series = demand_series[demand_series > 0]  # Filter out zero or negative demands
    demand = demand_series.to_dict()
    
    # Check if there's supply and demand
    if not supply or not demand:
        print(f"No supply or demand on {day_name}, skipping optimization.")
        continue
    
    # Run the optimization
    result = milp_optimization(supply, demand, distance_dict)
    results[day_name] = result
    
    
    # Optional: Display the allocations
    # print(f"Allocations for {day_name}:")
    # for (i, j), qty in result.items():
    #     print(f" - Supply from {i} to {j}: {qty} units")

# Compile all allocations into a single DataFrame
allocation_records = []
for day_name, allocations in results.items():
    for (i, j), qty in allocations.items():
        allocation_records.append({
            'day': day_name,
            'supermarket': i,
            'foodbank': j,
            'quantity': qty
        })

allocations_df = pd.DataFrame(allocation_records)

# Optionally, save the results to a CSV file
# allocations_df.to_csv('allocations_all_days.csv', index=False)

# Display the first few rows of the allocations
print(allocations_df.head())


# %%
allocations_df[allocations_df["foodbank"] == "Bounds Green"]

# %%
allocations_df[allocations_df["day"] == "Day 1"]

# %%
# Assuming 'pos', 'supermarket_names', and 'foodbank_names' are already defined

# For each day:
unique_days = allocations_df['day'].unique()
for day in unique_days:
    print(f"Processing {day}...")
    daily_allocations = allocations_df[allocations_df['day'] == day].copy()
    
    # Group allocations to avoid duplicates
    edge_data = daily_allocations.groupby(['supermarket', 'foodbank'])['quantity'].sum().reset_index()
    
    # Build edge list and weights
    edges = []
    edge_weights = {}
    for _, row in edge_data.iterrows():
        s = row['supermarket']
        f = row['foodbank']
        quantity = row['quantity']
        edges.append((s, f))
        edge_weights[(s, f)] = quantity
    
    # Verify number of edges
    print(f"Number of transactions on {day}: {len(edges)}")
    
    # Create a new graph for the day's allocations
    G_day = nx.Graph()
    G_day.add_nodes_from(supermarket_names, node_type='supermarket')
    G_day.add_nodes_from(foodbank_names, node_type='foodbank')
    G_day.add_edges_from(edges)
    
    # Calculate foodbank satisfaction levels (if applicable)
    # ... (same as before)
    
    # Create edge traces
    edge_traces = []
    for (s, f), quantity in edge_weights.items():
        x0, y0 = pos[s]
        x1, y1 = pos[f]
        width = quantity / 20  # Adjust scaling_factor as needed
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=width, color='#888'),
            hoverinfo='text',
            text=f"{s} to {f}: {quantity:.2f}",
            mode='lines')
        edge_traces.append(edge_trace)
    
    # Prepare node traces
    # Nodes involved in transactions
    supermarkets_involved = set(edge_data['supermarket'])
    foodbanks_involved = set(edge_data['foodbank'])
    
    # Node sizes
    supermarket_sizes = [15 if node in supermarkets_involved else 10 for node in supermarket_names]
    foodbank_sizes = [15 if node in foodbanks_involved else 10 for node in foodbank_names]
    
    # Supermarket trace
    supermarket_trace = go.Scatter(
        x=[pos[node][0] for node in supermarket_names],
        y=[pos[node][1] for node in supermarket_names],
        mode='markers',
        hoverinfo='text',
        text=supermarket_names,
        marker=dict(
            color='blue',
            size=supermarket_sizes,
            line_width=1),
        name='Supermarkets')
    
    # Foodbank trace
    foodbank_trace = go.Scatter(
        x=[pos[node][0] for node in foodbank_names],
        y=[pos[node][1] for node in foodbank_names],
        mode='markers',
        hoverinfo='text',
        text=foodbank_names,
        marker=dict(
            color='green',
            size=foodbank_sizes,
            line_width=1),
        name='Foodbanks')
    
    # Create figure
    fig = go.Figure(data=edge_traces + [supermarket_trace, foodbank_trace])
    
    # Update layout
    fig.update_layout(
        title=f'Food Distribution Network - {day}',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    # Show the figure
    fig.show()



