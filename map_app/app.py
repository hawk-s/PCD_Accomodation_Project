from flask import Flask, render_template, jsonify
import pandas as pd
import eurostat

app = Flask(__name__)

# --- DATA LOAD AND PREPARATION ---
# Import Eurostat data
data_analysed = eurostat.get_data_df('tour_ce_omn12')

# Replace month column to make it usable in analysis
data_analysed["month"] = data_analysed["month"].str.replace(r'M', "", regex=True)

# Rename the column for easier use
data_analysed = data_analysed.rename(columns={"geo\\TIME_PERIOD": "GEO"})

# Add years to the rows
data_analysed = data_analysed.melt(
    id_vars=['freq', 'indic_to', 'c_resid', 'month', 'unit', 'GEO'],
    var_name='TIME_PERIOD',
    value_name='bookings'
)

# Define main subsets based on 'indic_to'
Stay_df = data_analysed[data_analysed['indic_to'] == 'STY']
LengthOfStay_df = data_analysed[data_analysed['indic_to'] == 'LSTY']
NightsSpend_df = data_analysed[data_analysed['indic_to'] == 'NGT_SP']

# Create dictionary for subsets
subset_dict = {}
Yvars = {
    "Stay": Stay_df,
    "LengthOfStay": LengthOfStay_df,
    "NightsSpend": NightsSpend_df
}

for name, df in Yvars.items():
    subset_dict[f"{name}_DOM"] = df[df["c_resid"] == "DOM"]
    subset_dict[f"{name}_FOR"] = df[df["c_resid"] == "FOR"]
    subset_dict[f"{name}_TOTAL"] = df[df["c_resid"] == "TOTAL"]

# Split data into monthly and annual periods
transformed_data = {}
for name, df in subset_dict.items():
    monthly = df[df["month"] != 'TOTAL'].copy()
    annual = df[df["month"] == 'TOTAL'].copy()

    # Transform 'month' and 'TIME_PERIOD' to datetime
    monthly['month'] = pd.to_datetime(monthly['TIME_PERIOD'] + "-" + monthly['month'], format="%Y-%m")
    annual['month'] = pd.to_datetime(annual['TIME_PERIOD'], format="%Y")


    transformed_data[f"{name}_monthly"] = monthly
    transformed_data[f"{name}_annual"] = annual

# Clean data by removing NaN values
transformed_data_cleaned = {key: df.dropna() for key, df in transformed_data.items()}

# Filter datasets
stay_data = {key: df for key, df in transformed_data_cleaned.items() if 'Stay_' in key and 'LengthOfStay_' not in key}
length_of_stay_data = {key: df for key, df in transformed_data_cleaned.items() if 'LengthOfStay_' in key}
nights_spend_data = {key: df for key, df in transformed_data_cleaned.items() if 'NightsSpend_' in key}

# --- HELPER FUNCTION ---
def get_geojson_data(dataset, year, month):
    # Combine all the datasets (if dataset is a dictionary of DataFrames)
    combined = pd.concat(dataset.values(), ignore_index=True)

    print("Combined Data:")
    print(combined.head())
    print("Filter Year:", year)
    print("Filter Month:", month)

    print("Available Years:", combined['TIME_PERIOD'].unique())
    print("Available Months:", combined['month'].dt.month_name().unique())

    print("Unique Months in Data:", combined['month'].unique())

    # Ensure 'month' is a datetime object (if it's not already)
    combined['month'] = pd.to_datetime(combined['month'], errors='coerce')

    # Extract the year and month from the 'month' column
    combined['year'] = combined['month'].dt.year
    combined['month_name'] = combined['month'].dt.strftime('%B')  # This will give the full month name

    # Filter data by the selected year and month
    filtered = combined[(combined['year'] == year) & (combined['month_name'] == month)]

    # Debug: Print filtered DataFrame
    print("Filtered Data:")
    print(filtered)

    # Create GeoJSON features from filtered data
    geojson_features = []
    for _, row in filtered.iterrows():
        geojson_features.append({
            "type": "Feature",
            "properties": {
                "country": row["GEO"],
                "value": row["bookings"]
            },
            "geometry": {
                "type": "Point",
                "coordinates": [0, 0]  # Replace with actual coordinates if available
            }
        })

    return {"type": "FeatureCollection", "features": geojson_features}


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/<metric>/<year>/<month>')
def get_data(metric, year, month):
    # Choose the correct dataset based on the metric
    if metric == "stays":
        dataset = stay_data
    elif metric == "length_of_stay":
        dataset = length_of_stay_data
    elif metric == "nights_spent":
        dataset = nights_spend_data
    else:
        return jsonify({"error": "Invalid metric"}), 400

    # Get GeoJSON data
    geojson_data = get_geojson_data(dataset, int(year), month)
    return jsonify(geojson_data)

if __name__ == '__main__':
    app.run(debug=True)
