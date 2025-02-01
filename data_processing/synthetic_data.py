import pandas as pd
import numpy as np
from scipy.stats import erlang
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from calendar import month_name

# Category information dictionary
category_info = {
    "Fresh Produce": {
        "expiration_days": 4,
        "daily_weight_kg": 20.0
    },
    "Dairy": {
        "expiration_days": 7,
        "daily_weight_kg": 15.0
    },
    "Meat and Fish": {
        "expiration_days": 5,
        "daily_weight_kg": 12.0
    },
    "Bakery": {
        "expiration_days": 3,
        "daily_weight_kg": 10.0
    },
    "Pantry Items": {
        "expiration_days": 180,
        "daily_weight_kg": 5.0
    },
    "Personal Care": {
        "expiration_days": 365,
        "daily_weight_kg": 0.21
    },
    "Cleaning Products": {
        "expiration_days": 730,
        "daily_weight_kg": 0.15
    }
}

def get_data(filename):
    """Read and process the input CSV file."""
    try:
        df = pd.read_csv(filename)
        df['excess'] = df['supply'] - df['demand']
        new_df = df[['name', 'excess']].copy()
        new_df['excess'] = new_df['excess'].clip(lower=0)
        # Remove any rows with NaN values
        new_df = new_df.dropna()
        return new_df
    except Exception as e:
        print(f"Error reading data file: {e}")
        return pd.DataFrame({'name': [], 'excess': []})

def generate_mock_timeseries(df, category_info, avg_price=1.7789, year=2023):
    """Generate mock time series data for food waste."""
    
    def calculate_weight(excess_value):
        """Calculate weight parameter ensuring it's positive."""
        if pd.isna(excess_value) or excess_value <= 0:
            return 0.1
        return max(excess_value / avg_price, 0.1)

    def generate_category_samples(weight, days):
        """Generate daily samples for one category."""
        try:
            scale = max(weight/2, 0.1)
            return erlang.rvs(a=2, scale=scale, size=days)
        except ValueError as e:
            print(f"Error generating samples with weight {weight}: {e}")
            return np.ones(days) * 0.1

    # Generate dates for the entire year
    start_date = datetime.date(year, 1, 1)
    dates = [start_date + datetime.timedelta(days=x) for x in range(365)]
    
    result_df = pd.DataFrame()
    
    # Process each supermarket
    for _, row in df.iterrows():
        # Skip if excess is NaN
        if pd.isna(row['excess']):
            continue
            
        excess = max(row['excess'], 0)
        weight = calculate_weight(excess)
        
        # Sample for each category
        daily_samples = []
        selected_categories = list(category_info.keys())[:6]
        
        for category in selected_categories:
            category_samples = generate_category_samples(weight, 365)
            # Apply category-specific scaling
            category_samples *= category_info[category]['daily_weight_kg'] / 100
            daily_samples.append(category_samples)
        
        # Calculate daily averages across categories
        daily_averages = np.mean(daily_samples, axis=0)
        
        # Create temporary dataframe for this supermarket
        temp_df = pd.DataFrame({
            'date': dates,
            'supermarket': row['name'],
            'excess_kg': daily_averages
        })
        
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
    # Convert date column to datetime
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Ensure all values are positive
    result_df['excess_kg'] = result_df['excess_kg'].clip(lower=0)
    
    return result_df

def visualize_timeseries(timeseries_df):
    """
    Create various visualizations for the time series data.
    """
    # Set the style using seaborn
    sns.set_style("whitegrid")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Time Series Plot for all supermarkets
    plt.subplot(2, 2, 1)
    for supermarket in timeseries_df['supermarket'].unique():
        data = timeseries_df[timeseries_df['supermarket'] == supermarket]
        plt.plot(data['date'], data['excess_kg'], label=supermarket, alpha=0.7)
    
    plt.title('Daily Excess Food Waste by Supermarket', fontsize=12, pad=20)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Excess (kg)', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # 2. Monthly Average Waste
    plt.subplot(2, 2, 2)
    monthly_avg = timeseries_df.copy()
    monthly_avg['month'] = monthly_avg['date'].dt.month
    monthly_data = monthly_avg.groupby('month')['excess_kg'].mean()
    
    sns.barplot(x=range(1, 13), y=monthly_data.values, color='skyblue')
    plt.title('Average Monthly Food Waste', fontsize=12, pad=20)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Average Excess (kg)', fontsize=10)
    plt.xticks(range(12), [month_name[i][:3] for i in range(1, 13)], rotation=45)
    
    # 3. Box Plot by Supermarket
    plt.subplot(2, 2, 3)
    sns.boxplot(x='supermarket', y='excess_kg', data=timeseries_df)
    plt.title('Distribution of Daily Food Waste by Supermarket', fontsize=12, pad=20)
    plt.xlabel('Supermarket', fontsize=10)
    plt.ylabel('Excess (kg)', fontsize=10)
    plt.xticks(rotation=45)
    
    # 4. Heatmap of Weekly Patterns
    plt.subplot(2, 2, 4)
    weekly_data = timeseries_df.copy()
    weekly_data['dayofweek'] = weekly_data['date'].dt.dayofweek
    weekly_data['week'] = weekly_data['date'].dt.isocalendar().week
    
    pivot_table = weekly_data.pivot_table(
        values='excess_kg',
        index='week',
        columns='dayofweek',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_table, cmap='YlOrRd')
    plt.title('Weekly Patterns of Food Waste', fontsize=12, pad=20)
    plt.xlabel('Day of Week', fontsize=10)
    plt.ylabel('Week Number', fontsize=10)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Additional statistics
    print("\nSummary Statistics:")
    print("==================")
    print(f"Total waste recorded: {timeseries_df['excess_kg'].sum():.2f} kg")
    print(f"Average daily waste: {timeseries_df['excess_kg'].mean():.2f} kg")
    print(f"Maximum daily waste: {timeseries_df['excess_kg'].max():.2f} kg")
    print(f"Minimum daily waste: {timeseries_df['excess_kg'].min():.2f} kg")
    
    # Calculate and display total waste by supermarket
    total_by_supermarket = timeseries_df.groupby('supermarket')['excess_kg'].sum()
    print("\nTotal Waste by Supermarket:")
    print("===========================")
    for supermarket, total in total_by_supermarket.items():
        print(f"{supermarket}: {total:.2f} kg")

def main():
    """Main function to run the data generation process."""
    try:
        # Read input data
        df = get_data('./data_processing/clearer_ish_data.csv')
        
        if df.empty:
            print("No data to process")
            return
        
        # Generate time series data
        timeseries_df = generate_mock_timeseries(df, category_info)
        
        # Basic validation
        if timeseries_df.empty:
            print("No time series data generated")
            return
            
        # Print summary statistics
        print("\nTime Series Data Summary:")
        print(f"Total records: {len(timeseries_df)}")
        print(f"Date range: {timeseries_df['date'].min()} to {timeseries_df['date'].max()}")
        print(f"Average excess (kg): {timeseries_df['excess_kg'].mean():.2f}")
        print("\nSample of generated data:")
        print(timeseries_df.head())
        
        # Optionally save the results
        timeseries_df.to_csv('generated_timeseries.csv', index=False)
        print("\nData saved to 'generated_timeseries.csv'")
        
    except Exception as e:
        print(f"An error occurred in main execution: {e}")

    return timeseries_df

if __name__ == "__main__":
    t = main()
    visualize_timeseries(t)