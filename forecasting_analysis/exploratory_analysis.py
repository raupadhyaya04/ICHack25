# %%
pip install osmnx matplotlib

# %%
!pip install --upgrade osmnx


# %%
import osmnx as ox

place_name = "London, England, United Kingdom"
gdf = ox.geocode_to_gdf(place_name)

if not gdf.empty:
    boundary_polygon = gdf.geometry.values[0]
    print("Successfully fetched London boundary!")
else:
    print("No valid polygon found for London.")


# %%
import matplotlib.pyplot as plt

# Plot the boundary of London
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color='lightblue', edgecolor='blue')
plt.title("Boundary of London")
plt.show()


# %%
import osmnx as ox
import matplotlib.pyplot as plt

# Step 1: Get the boundary polygon for London
boundary_gdf = ox.geocode_to_gdf("London, England, United Kingdom")
boundary_polygon = boundary_gdf.geometry.values[0]

print("Boundary polygon for London:")
print(boundary_polygon)
print(type(boundary_polygon))

# Step 2: Define the tags for supermarkets
tags = {"shop": "supermarket"}

# Step 3: Retrieve supermarkets within the London polygon
supermarkets = ox.features.features_from_polygon(boundary_polygon, tags)

# Step 4: Inspect the retrieved data
print(supermarkets.head())

# Step 5: Visualize the supermarkets
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the London boundary
boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black')

# Plot the supermarkets
supermarkets.plot(ax=ax, color='red', markersize=5)

plt.title('Supermarkets in London')
plt.show()


# %%
import requests
import pandas as pd

# Define the Overpass API URL
overpass_url = "http://overpass-api.de/api/interpreter"

# Overpass QL query to get all supermarkets in Greater London
overpass_query = """
[out:json][timeout:120];
area["name"="Greater London"]["boundary"="administrative"]->.searchArea;
(
  node["shop"="supermarket"](area.searchArea);
  way["shop"="supermarket"](area.searchArea);
  relation["shop"="supermarket"](area.searchArea);
);
out center;
"""

# Send the request to the Overpass API
response = requests.get(overpass_url, params={'data': overpass_query})

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
else:
    print(f"Error: {response.status_code}")
    exit()

# Process the data
elements = data['elements']
supermarkets = []
for el in elements:
    tags = el.get('tags', {})
    name = tags.get('name', 'Unnamed')
    if el['type'] == 'node':
        lat = el.get('lat')
        lon = el.get('lon')
    elif 'center' in el:
        lat = el['center']['lat']
        lon = el['center']['lon']
    else:
        lat = None
        lon = None
    supermarkets.append({
        'name': name,
        'latitude': lat,
        'longitude': lon,
        'osm_id': el.get('id')
    })

# Create a DataFrame
df = pd.DataFrame(supermarkets)

# Save to CSV
# df.to_csv('london_supermarkets.csv', index=False)

print(f"Saved data to 'london_supermarkets.csv'. Found {len(df)} supermarkets.")


# %% [markdown]
# # Onto the map

# %%
supermarkets = pd.read_csv('london_supermarkets.csv')
foodbanks = pd.read_csv('givefood_foodbanks.csv')

print(f"Number of supermarkets: {len(supermarkets)}")
print(f"Number of food banks: {len(foodbanks)}")

# %%
# Now we need the London boundary polygon to filter the data
place_name = "London, England, United Kingdom"
gdf = ox.geocode_to_gdf(place_name)

if not gdf.empty:
    boundary_polygon = gdf.geometry.values[0]
    print("Successfully fetched London boundary!")

print(supermarkets.columns)
print(foodbanks.columns)

# %%
# Split the 'lat_lng' column into 'latitude' and 'longitude' columns
foodbanks[['latitude', 'longitude']] = foodbanks['lat_lng'].str.split(',', expand=True)

# Remove any leading/trailing whitespaces and convert to float
foodbanks['latitude'] = foodbanks['latitude'].str.strip().astype(float)
foodbanks['longitude'] = foodbanks['longitude'].str.strip().astype(float)



# %%
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Step 1: Obtain London's boundary polygon
london_boundary = ox.geocode_to_gdf('Greater London, United Kingdom')

# Optional: Visualize the boundary
# london_boundary.plot()
# plt.show()

# Step 2: Parse 'lat_lng' into 'latitude' and 'longitude'
foodbanks[['latitude', 'longitude']] = foodbanks['lat_lng'].str.split(',', expand=True)
foodbanks['latitude'] = foodbanks['latitude'].astype(float)
foodbanks['longitude'] = foodbanks['longitude'].astype(float)

# Step 3: Create GeoDataFrame for foodbanks
foodbanks['geometry'] = [Point(xy) for xy in zip(foodbanks['longitude'], foodbanks['latitude'])]
foodbanks_gdf = gpd.GeoDataFrame(foodbanks, geometry='geometry', crs='EPSG:4326')

# Step 4: Ensure CRS match
foodbanks_gdf = foodbanks_gdf.to_crs(london_boundary.crs)

# Step 5: Spatial join to filter foodbanks within London
foodbanks_in_london = gpd.sjoin(foodbanks_gdf, london_boundary, how='inner', predicate='within')
foodbanks_in_london = foodbanks_in_london.drop(columns=['index_right'])

# Step 6: Results
print(f"Total foodbanks: {len(foodbanks_gdf)}")
print(f"Foodbanks within London: {len(foodbanks_in_london)}")

# Step 7: Save or use the filtered data
# foodbanks_in_london.to_csv('foodbanks_in_london.csv', index=False)


# %%
foodbanks_in_london

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Step 1: Prepare the Data

# Ensure foodbanks data is a GeoDataFrame
if not isinstance(foodbanks_in_london, gpd.GeoDataFrame):
    foodbanks_in_london['geometry'] = [Point(xy) for xy in zip(foodbanks_in_london['longitude'], foodbanks_in_london['latitude'])]
    foodbanks_gdf = gpd.GeoDataFrame(foodbanks_in_london, geometry='geometry', crs='EPSG:4326')
else:
    foodbanks_gdf = foodbanks_in_london

# Prepare supermarkets data
supermarkets['geometry'] = [Point(xy) for xy in zip(supermarkets['longitude'], supermarkets['latitude'])]
supermarkets_gdf = gpd.GeoDataFrame(supermarkets, geometry='geometry', crs='EPSG:4326')

# Ensure CRS match
london_boundary = london_boundary.to_crs('EPSG:4326')
foodbanks_gdf = foodbanks_gdf.to_crs('EPSG:4326')
supermarkets_gdf = supermarkets_gdf.to_crs('EPSG:4326')

# Step 2: Plot the Data

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot London boundary
london_boundary.plot(ax=ax, color='white', edgecolor='black', linewidth=1)

# Plot foodbanks
foodbanks_gdf.plot(ax=ax, 
                   markersize=50, 
                   color='red', 
                   marker='o', 
                   label='Foodbanks')

# Plot supermarkets
supermarkets_gdf.plot(ax=ax, 
                      markersize=20, 
                      color='blue', 
                      marker='^', 
                      label='Supermarkets')

# Add legend
plt.legend(prop={'size': 12})

# Add title
plt.title('Foodbanks and Supermarkets in London', fontsize=15)

# Remove axes
ax.set_axis_off()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# %%
import pandas as pd
import geopandas as gpd

# Step 1: Prepare the Foodbanks DataFrame

# Extract latitude and longitude if necessary
if 'latitude' not in foodbanks_in_london.columns or 'longitude' not in foodbanks_in_london.columns:
    foodbanks_in_london['longitude'] = foodbanks_in_london.geometry.x
    foodbanks_in_london['latitude'] = foodbanks_in_london.geometry.y

# Add 'type' column
foodbanks_in_london['type'] = 'Foodbank'

# Ensure 'name' column exists
# If the 'name' column is missing or named differently, adjust accordingly
# For this example, we'll proceed assuming 'name' exists

# Create subset
foodbanks_subset = foodbanks_in_london[['name', 'latitude', 'longitude', 'type']].copy()

# Step 2: Prepare the Supermarkets DataFrame

# Extract latitude and longitude if necessary
if 'latitude' not in supermarkets_gdf.columns or 'longitude' not in supermarkets_gdf.columns:
    supermarkets_gdf['longitude'] = supermarkets_gdf.geometry.x
    supermarkets_gdf['latitude'] = supermarkets_gdf.geometry.y

# Add 'type' column
supermarkets_gdf['type'] = 'Supermarket'

# Ensure 'name' column exists
# For this example, we'll proceed assuming 'name' exists

# Create subset
supermarkets_subset = supermarkets_gdf[['name', 'latitude', 'longitude', 'type']].copy()

# Step 3: Combine the DataFrames
combined_df = pd.concat([foodbanks_subset, supermarkets_subset], ignore_index=True)

# Step 4: Verify the Combined DataFrame
print(combined_df.head())

# Step 5: (Optional) Save to CSV
combined_df.to_csv('combined_locations.csv', index=False)


# %%
foodbanks_subset = foodbanks_in_london[['organisation_name', 'latitude', 'longitude', 'type']].copy()
foodbanks_subset

# %%
foodbannks = pd.read_csv('givefood_foodbanks.csv')
demand = pd.read_csv('givefood_items.csv', index_col=0)

# %%
demand['created'] = pd.to_datetime(demand['created'])

# Only keep values in 2024
demand = demand[demand['created'].dt.year == 2024]

# sort by organisation name
demand = demand.sort_values(by='organisation_name')

# only keep need in the type column
demand = demand[demand['type'] == 'need']

demand

# %% [markdown]
# ## Making the assumption above that each item costs ~1, can google and see median is 2-3 pounds

# %%
len(demand["item"].unique())

# %%
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Assuming 'foodbanks_subset' and 'demand' DataFrames are already defined

# Step 1: Extract latitude and longitude from 'lat_lng' column in 'demand' DataFrame
demand[['latitude', 'longitude']] = demand['lat_lng'].str.split(',', expand=True).astype(float)

# Step 2: Rename the latitude and longitude columns for clarity
demand.rename(columns={'latitude': 'latitude_demand', 'longitude': 'longitude_demand'}, inplace=True)
foodbanks_subset.rename(columns={'latitude': 'latitude_foodbank', 'longitude': 'longitude_foodbank'}, inplace=True)

# Step 3: Reset index on 'demand' DataFrame to make 'organisation_name' a column
demand_reset = demand.reset_index()

# Step 4: Merge DataFrames on 'organisation_name'
merged_df = pd.merge(
    demand_reset,
    foodbanks_subset,
    on='organisation_name',
    how='inner'
)

# Step 5: Define Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return distances

# Step 6: Calculate distances between demand and foodbank locations
merged_df['distance'] = haversine_distance(
    merged_df['latitude_demand'],
    merged_df['longitude_demand'],
    merged_df['latitude_foodbank'],
    merged_df['longitude_foodbank']
)

# Step 7: Filter matches based on proximity (e.g., within 5 km)
distance_threshold = 1  # kilometers
filtered_df = merged_df[merged_df['distance'] <= distance_threshold]

# Step 8: Aggregate the demand for each foodbank location
demand_per_foodbank = filtered_df.groupby(
    ['organisation_name', 'latitude_foodbank', 'longitude_foodbank']
).size().reset_index(name='demand_count')

# Step 9: Merge the demand counts back into 'foodbanks_subset'
foodbanks_with_demand = pd.merge(
    foodbanks_subset,
    demand_per_foodbank,
    on=['organisation_name', 'latitude_foodbank', 'longitude_foodbank'],
    how='left'
)

# Fill NaN values in 'demand_count' with 0
foodbanks_with_demand['demand_count'] = foodbanks_with_demand['demand_count'].fillna(0)

# The 'foodbanks_with_demand' DataFrame now contains the demand counts for each foodbank


# %%
supermarkets_subset

# %%
foodbanks_with_demand

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Step 1: Prepare the Data

# Ensure 'foodbanks_with_demand' is a GeoDataFrame
if not isinstance(foodbanks_with_demand, gpd.GeoDataFrame):
    foodbanks_with_demand['geometry'] = [Point(xy) for xy in zip(foodbanks_with_demand['longitude_foodbank'], foodbanks_with_demand['latitude_foodbank'])]
    foodbanks_gdf = gpd.GeoDataFrame(foodbanks_with_demand, geometry='geometry', crs='EPSG:4326')
else:
    foodbanks_gdf = foodbanks_with_demand

# Ensure 'supermarkets_subset' is a GeoDataFrame
if not isinstance(supermarkets_subset, gpd.GeoDataFrame):
    supermarkets_subset['geometry'] = [Point(xy) for xy in zip(supermarkets_subset['longitude'], supermarkets_subset['latitude'])]
    supermarkets_gdf = gpd.GeoDataFrame(supermarkets_subset, geometry='geometry', crs='EPSG:4326')
else:
    supermarkets_gdf = supermarkets_subset

# Ensure CRS match
london_boundary = london_boundary.to_crs('EPSG:4326')
foodbanks_gdf = foodbanks_gdf.to_crs('EPSG:4326')
supermarkets_gdf = supermarkets_gdf.to_crs('EPSG:4326')

# Step 2: Plot the Data

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot London boundary
london_boundary.plot(ax=ax, color='white', edgecolor='black', linewidth=1)

# Plot supermarkets
supermarkets_gdf.plot(ax=ax, 
                      markersize=20, 
                      color='blue', 
                      marker='^', 
                      # Remove label to prevent duplicate legend entries
                      # label='Supermarkets'
                      )

# Plot foodbanks with demand as color map
foodbanks_plot = foodbanks_gdf.plot(ax=ax, 
                                    markersize=50, 
                                    column='demand_count',  # Use 'demand_count' for color mapping
                                    cmap='OrRd',          # Changed color map to 'YlGnBu' (Yellow-Green-Blue)
                                    legend=True,            # Show legend (color bar)
                                    legend_kwds={'label': "Demand Count (£)", 'shrink': 0.5},  # Customize legend
                                    marker='o')

# Add custom legend entries
from matplotlib.lines import Line2D

# Create custom legend handles
supermarket_handle = Line2D([], [], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Supermarkets')
foodbank_handle = Line2D([], [], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Foodbanks')

# Combine legend handles
legend_handles = [supermarket_handle, foodbank_handle]

# Add the legend to the plot
ax.legend(handles=legend_handles, title='Locations', loc='upper left')

# Add title
plt.title('Foodbanks and Supermarkets in London\nFoodbanks Colored by Demand', fontsize=15)

# Remove axes
ax.set_axis_off()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# ## Proportional mean to sample the demand for each supermarket

# %% [markdown]
# # Find demand for supermarket based on area

# %%
import geopandas as gpd
import pandas as pd
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the GeoJSON file directly from the URL
geojson_url = 'https://github.com/martinjc/UK-GeoJSON/blob/master/json/administrative/eng/lad.json?raw=true'
gdf_boroughs = gpd.read_file(geojson_url)

boroughs = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City of London", "Croydon",
    "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", 
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", 
    "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
]

# Filter the GeoDataFrame to include only the London boroughs
gdf_london = gdf_boroughs[gdf_boroughs['LAD13NM'].isin(boroughs)]

spending_values = [
    249.55, 623.70, 266.93, 504.57, 628.61, 1627.08, 36.24, 832.83,
    763.69, 317.73, 481.42, 451.62, 894.34, 250.95, 468.56, 236.15,
    975.12, 596.55, 755.53, 714.16, 504.39, 772.13, 530.12, 423.88,
    625.13, 279.99, 523.05, 952.85, 469.89, 1184.48, 274.26, 1070.51, 1626.71
]

# Create a DataFrame for the spending data
spending_df = pd.DataFrame({
    'borough': boroughs,
    'spending': spending_values
})

# Clean borough names in both DataFrames to ensure they match
gdf_london['LAD13NM_clean'] = gdf_london['LAD13NM'].str.lower().str.strip()
spending_df['borough_clean'] = spending_df['borough'].str.lower().str.strip()

# Correct mismatches in borough names
spending_df['borough_clean'] = spending_df['borough_clean'].replace({
    'hammersmith and fulham': 'hammersmith and fulham',
})

gdf_london['LAD13NM_clean'] = gdf_london['LAD13NM_clean'].replace({
    'city of london corporation': 'city of london',
    'westminster city council': 'westminster',
})

# # Merge the spending data with the GeoDataFrame
# merged_df = gdf_london.merge(
#     spending_df,
#     left_on='LAD13NM_clean',
#     right_on='borough_clean'
# )


# Assuming df1 is your first DataFrame and df2 is your second DataFrame

# Merge the DataFrames on the clean borough names
merged_df = pd.merge(gdf_london, spending_df, left_on='LAD13NM_clean', right_on='borough_clean')

# Select only the required columns
final_df = merged_df[['LAD13NM', 'geometry', 'spending']]

final_df


# %%
# Import necessary libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# For reproducibility of the random samples
np.random.seed(0)

# --- Step 1: Prepare the Data ---

# Assuming 'boroughs_df' is your first DataFrame with borough data
# and 'stores_df' is your second DataFrame with store data

# Convert 'stores_df' into a GeoDataFrame
supermarkets_subset['geometry'] = gpd.points_from_xy(supermarkets_subset['longitude'], supermarkets_subset['latitude'])
stores_gdf = gpd.GeoDataFrame(supermarkets_subset, geometry='geometry')

# Ensure both GeoDataFrames have the same Coordinate Reference System (CRS)
# WGS84 Latitude/Longitude: EPSG:4326
stores_gdf.set_crs(epsg=4326, inplace=True)
final_df.set_crs(epsg=4326, inplace=True)

# Optional: Validate and clean borough geometries
final_df['geometry'] = final_df['geometry'].buffer(0)

# --- Step 2: Perform a Spatial Join ---

# Spatially join stores with boroughs to find which borough each store is in
stores_in_boroughs = gpd.sjoin(
    stores_gdf,
    final_df[['LAD13NM', 'geometry', 'spending']],
    how='left',
    predicate='within'
)

# --- Step 3: Generate Demand Counts ---

# Define a function to sample from the normal distribution
def sample_demand(spending_value):
    if pd.isnull(spending_value) or spending_value <= 0:
        return None  # Cannot sample without a valid spending value
    variance = spending_value * 0.1
    std_dev = np.sqrt(variance)
    sample = np.random.normal(loc=spending_value, scale=std_dev)
    return max(sample, 0)  # Ensure the demand count is not negative

# Apply the function to generate 'demand_count' for each store
stores_in_boroughs['demand_count'] = stores_in_boroughs['spending'].apply(sample_demand)

# Display the updated DataFrame
print(stores_in_boroughs[['name', 'LAD13NM', 'spending', 'demand_count']])


# %%
stores_in_boroughs['spending'].describe()

# %%
# stock column sampled from normal dist with mean = demand_count * 0.3 and std = demand_count * 0.4 but keep it positive and two decimal places
stores_in_boroughs['stock'] = np.random.normal(loc=stores_in_boroughs['demand_count'] + stores_in_boroughs['demand_count'] *0.2, scale=stores_in_boroughs['demand_count'] * 0.2)
stores_in_boroughs['stock'] = stores_in_boroughs['stock'].apply(lambda x: max(x, 0))  # Ensure stock is not negative
stores_in_boroughs['stock'] = stores_in_boroughs['stock'].round(2)  # Round to two decimal places

stores_in_boroughs

# %%
foodbanks_with_demand["stock"] = None

# %%
# Import necessary libraries
import pandas as pd

# Read the supermarket dat
# Standardize column names in foodbanks
foodbanks_with_demand.rename(columns={
    'organisation_name': 'name',
    'latitude_foodbank': 'lat',
    'longitude_foodbank': 'lng',
    'stock': 'supply',
    'demand_count': 'demand'
}, inplace=True)

# Standardize column names in supermarkets
stores_in_boroughs.rename(columns={
    'latitude': 'lat',
    'longitude': 'lng',
    'stock': 'supply',
    'demand_count': 'demand'
}, inplace=True)


# Select the required columns
supermarkets = stores_in_boroughs[['name', 'lat', 'lng', 'type', 'demand', 'supply']]
foodbanks = foodbanks_with_demand[['name', 'lat', 'lng', 'type', 'demand', 'supply']]

# Combine the datasets
combined_df = pd.concat([supermarkets, foodbanks], ignore_index=True)

combined_df


# %%
combined_df["waste"] = combined_df["supply"] - combined_df["demand"]

combined_df['waste'] = combined_df['waste'].apply(lambda x: max(x, 0))  # Ensure stock is not negative

combined_df.to_csv('clearer_ish_data.csv', index=False)

# %%
combined_df = pd.read_csv('clearer_ish_data.csv')

# %%
combined_df['supply'] = combined_df['supply'] * 10
combined_df['demand'] = combined_df['demand'] * 10

combined_df["waste"] = combined_df["supply"] - combined_df["demand"]

combined_df['waste'] = combined_df['waste'].apply(lambda x: max(x, 0))  # Ensure stock is not negative

combined_df

# %%
combined_df.to_csv('final_data.csv', index=False)

# %%
# Import pandas
import pandas as pd

# Assuming your first DataFrame is 'stores_df' and the second is 'foodbanks_df'

# --- Step 1: Standardize Column Names ---

# Rename columns in the foodbanks DataFrame to match the stores DataFrame
foodbanks_df_renamed = foodbanks_with_demand.rename(columns={
    'organisation_name': 'name',
    'latitude_foodbank': 'latitude',
    'longitude_foodbank': 'longitude',
    'demand_count': 'demand'
})

# Rename 'demand_count' to 'demand' in the stores DataFrame as well
stores_df_renamed = stores_in_boroughs.rename(columns={
    'demand_count': 'demand'
})

# --- Step 2: Concatenate the DataFrames ---

# Select the necessary columns from each DataFrame
stores_df_selected = stores_df_renamed[['name', 'latitude', 'longitude', 'type', 'demand']]
foodbanks_df_selected = foodbanks_df_renamed[['name', 'latitude', 'longitude', 'type', 'demand']]

# Concatenate the DataFrames
combined_df = pd.concat([stores_df_selected, foodbanks_df_selected], ignore_index=True)

# --- Step 3: View the Combined DataFrame ---

# Display the combined DataFrame
combined_df

# %%
combined_df.to_csv('everythingwithdemand.csv', index=False)

# %%
supermarkets_subset

# %%
foodbanks_with_demand["demand_count"]

# %%
combined_df.to_csv('supermarkets_foodbanks_foodbankdemand.csv', index=False)

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pandas as pd

# Load the GeoJSON file directly from the URL
geojson_url = 'https://github.com/martinjc/UK-GeoJSON/blob/master/json/administrative/eng/lad.json?raw=true'
gdf_boroughs = gpd.read_file(geojson_url)

# Inspect the GeoDataFrame
gdf_boroughs

boroughs = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City of London", "Croydon",
    "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", 
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", 
    "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
]

# Filter the GeoDataFrame to include only the London boroughs
gdf_london = gdf_boroughs[gdf_boroughs['LAD13NM'].isin(boroughs)]

# List of boroughs and their corresponding average spending values
boroughs = [
    "Barking & Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City of London", "Croydon",
    "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith & Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington & Chelsea", "Kingston upon Thames", "Lambeth", 
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", 
    "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
]
spending_values = [
    249.55, 623.70, 266.93, 504.57, 628.61, 1627.08, 36.24, 832.83, 763.69, 317.73, 481.42, 451.62, 
    894.34, 250.95, 468.56, 236.15, 975.12, 596.55, 755.53, 714.16, 504.39, 772.13, 530.12, 423.88, 
    625.13, 279.99, 523.05, 952.85, 469.89, 1184.48, 274.26, 1070.51, 1626.71
]

# Create a DataFrame from the boroughs and spending data
df_spending = pd.DataFrame({
    "borough": boroughs,
    "spending": spending_values
})

# Load the shapefile for London's boroughs
gdf = ox.geocode_to_gdf("London, England")

# Ensure the borough names are correctly matched
gdf = gdf[gdf['borough'].notna()]

# Merge the spending data with the geometries (ensure the boroughs match)
gdf = gdf.set_index("borough").join(df_spending.set_index("borough"))

# Plot the map with color-coding for spending values
fig, ax = plt.subplots(figsize=(10, 10))

# Plotting with a color map
gdf.plot(column='spending', ax=ax, legend=True,
         legend_kwds={'label': "Average Spending on Food in London Boroughs",
                      'orientation': "horizontal"},
         cmap='coolwarm', edgecolor='black')

# Title and labels
ax.set_title('Average Food Spending in London Boroughs', fontsize=15)
ax.set_axis_off()

# Show the plot
plt.show()


# %%
import geopandas as gpd
import pandas as pd

# Load the GeoJSON file directly from the URL
geojson_url = 'https://github.com/martinjc/UK-GeoJSON/blob/master/json/administrative/eng/lad.json?raw=true'
gdf_boroughs = gpd.read_file(geojson_url)

# Inspect the GeoDataFrame
gdf_boroughs

# %%
boroughs = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City of London", "Croydon",
    "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow", "Havering",
    "Hillingdon", "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", 
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", 
    "Tower Hamlets", "Waltham Forest", "Wandsworth", "Westminster"
]

# Filter the GeoDataFrame to include only the London boroughs
gdf_london = gdf_boroughs[gdf_boroughs['LAD13NM'].isin(boroughs)]
gdf_london

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Your spending data
boroughs = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey",
    "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth",
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
    "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest",
    "Wandsworth", "Westminster"
]

spending_values = [
    249.55, 623.70, 266.93, 504.57, 628.61, 1627.08, 36.24, 832.83,
    763.69, 317.73, 481.42, 451.62, 894.34, 250.95, 468.56, 236.15,
    975.12, 596.55, 755.53, 714.16, 504.39, 772.13, 530.12, 423.88,
    625.13, 279.99, 523.05, 952.85, 469.89, 1184.48, 274.26, 1070.51, 1626.71
]

# Create a DataFrame for the spending data
spending_df = pd.DataFrame({
    'borough': boroughs,
    'spending': spending_values
})

# Clean borough names in both DataFrames to ensure they match
gdf_london['LAD13NM_clean'] = gdf_london['LAD13NM'].str.lower().str.strip()
spending_df['borough_clean'] = spending_df['borough'].str.lower().str.strip()

# Correct mismatches in borough names
spending_df['borough_clean'] = spending_df['borough_clean'].replace({
    'hammersmith and fulham': 'hammersmith and fulham',
})

gdf_london['LAD13NM_clean'] = gdf_london['LAD13NM_clean'].replace({
    'city of london corporation': 'city of london',
    'westminster city council': 'westminster',
})

# Merge the spending data with the GeoDataFrame
merged_df = gdf_london.merge(
    spending_df,
    left_on='LAD13NM_clean',
    right_on='borough_clean'
)

# Create abbreviations for boroughs
borough_abbr = {
    'Barking and Dagenham': 'B&D',
    'Barnet': 'Barnet',
    'Bexley': 'Bexley',
    'Brent': 'Brent',
    'Bromley': 'Bromley',
    'Camden': 'Camden',
    'City of London': 'City',
    'Croydon': 'Croydon',
    'Ealing': 'Ealing',
    'Enfield': 'Enfield',
    'Greenwich': 'Greenwich',
    'Hackney': 'Hackney',
    'Hammersmith and Fulham': 'H&F',
    'Haringey': 'Haringey',
    'Harrow': 'Harrow',
    'Havering': 'Havering',
    'Hillingdon': 'Hillingdon',
    'Hounslow': 'Hounslow',
    'Islington': 'Islington',
    'Kensington and Chelsea': 'K&C',
    'Kingston upon Thames': 'Kingston',
    'Lambeth': 'Lambeth',
    'Lewisham': 'Lewisham',
    'Merton': 'Merton',
    'Newham': 'Newham',
    'Redbridge': 'Redbridge',
    'Richmond upon Thames': 'Richmond',
    'Southwark': 'Southwark',
    'Sutton': 'Sutton',
    'Tower Hamlets': 'T. Hamlets',
    'Waltham Forest': 'W. Forest',
    'Wandsworth': 'Wandsworth',
    'Westminster': 'Westminster'
}

# Map abbreviations to the merged DataFrame
merged_df['borough_abbr'] = merged_df['borough'].map(borough_abbr)

# Ensure the merge was successful
if merged_df.empty:
    print("Merge resulted in an empty DataFrame. Please check for mismatched borough names.")
else:
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot the map without automatic legend
    merged_df.plot(
        column='spending',
        cmap='OrRd',
        linewidth=0.5,
        ax=ax,
        edgecolor='0.6',
        legend=False  # Disable the automatic legend
    )

    # Remove axis for a cleaner look
    ax.axis('off')

    # Add a title
    ax.set_title('Supermarket Spending by London Borough (2024)', fontdict={'fontsize': 20}, pad=20)

    # Calculate centroid coordinates for annotations
    merged_df['coords'] = merged_df['geometry'].apply(lambda x: x.representative_point().coords[0])

    # Annotate boroughs with abbreviations
    for idx, row in merged_df.iterrows():
        plt.annotate(
            row['borough_abbr'],        # Use the abbreviation
            xy=row['coords'],
            xytext=(0, 0),
            textcoords='offset points',
            horizontalalignment='center',
            verticalalignment='center',
            fontweight='bold',
            fontsize=7,
            color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.7)
        )

    # Adjust the layout to make space for the colorbar
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

    # Create colorbar on the right side
    # Adjust the position of the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.02)  # Reduced pad from 0.1 to 0.02

    # Normalize the color scale
    vmin = merged_df['spending'].min()
    vmax = merged_df['spending'].max()
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # Dummy array for the scalar mappable

    # Add the colorbar
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Supermarket Spending (£)', fontsize=12)

    # Adjust colorbar tick label font size
    cbar.ax.tick_params(labelsize=10)

    # Display the map
    plt.show()



