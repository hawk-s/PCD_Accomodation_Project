from flask import Flask, render_template, jsonify, request
import pandas as pd
import eurostat
import json
from shapely.geometry import shape, Point
from pyproj import Transformer


app = Flask(__name__)

# Load GeoJSON data for regions and countries
with open('map_app copy/nuts_2021.geojson', 'r', encoding='utf-8') as f:
    geo_coordinates = json.load(f)

# --- DATA LOAD AND PREPARATION ---
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
Yvars = {"Stay": Stay_df, "LengthOfStay": LengthOfStay_df, "NightsSpend": NightsSpend_df}
for name, df in Yvars.items():
    subset_dict[f"{name}_DOM"] = df[df["c_resid"] == "DOM"]
    subset_dict[f"{name}_FOR"] = df[df["c_resid"] == "FOR"]
    subset_dict[f"{name}_TOTAL"] = df[df["c_resid"] == "TOTAL"]

# Split data into monthly and annual periods
transformed_data = {}
for name, df in subset_dict.items():
    monthly = df[df["month"] != 'TOTAL'].copy()
    annual = df[df["month"] == 'TOTAL'].copy()
    monthly['month'] = pd.to_datetime(monthly['TIME_PERIOD'] + "-" + monthly['month'], format="%Y-%m")
    annual['month'] = pd.to_datetime(annual['TIME_PERIOD'], format="%Y")
    transformed_data[f"{name}_monthly"] = monthly
    transformed_data[f"{name}_annual"] = annual

transformed_data_cleaned = {key: df.dropna() for key, df in transformed_data.items()}
stay_data = {key: df for key, df in transformed_data_cleaned.items() if 'Stay_' in key}
length_of_stay_data = {key: df for key, df in transformed_data_cleaned.items() if 'LengthOfStay_' in key}
nights_spend_data = {key: df for key, df in transformed_data_cleaned.items() if 'NightsSpend_' in key}

# --- HELPER FUNCTION ---
# Create a transformer for converting from EPSG:3035 (typical for Eurostat) to EPSG:4326 (WGS84)
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def get_geojson_data(dataset, year, month):
    if isinstance(dataset, dict):
        combined = pd.concat(dataset.values(), ignore_index=True)
    else:
        combined = dataset  # If dataset is already a DataFrame

    combined['month'] = pd.to_datetime(combined['month'], errors='coerce')
    combined['year'] = combined['month'].dt.year
    combined['month_name'] = combined['month'].dt.strftime('%B')

    if month.lower() == "total":  # Handle annual data
        filtered = combined[combined['year'] == year]
    else:  # Handle monthly data
        filtered = combined[(combined['year'] == year) & (combined['month_name'] == month)]

    filtered = filtered[filtered["GEO"].str.len() > 2]
    filtered = filtered[~filtered["GEO"].isin(["EU27_2020"])]

    geojson_features = []
    for _, row in filtered.iterrows():
        geo_data = next((f for f in geo_coordinates['features'] if f['properties']['NUTS_ID'] == row["GEO"]), None)
        if geo_data:
            geometry = shape(geo_data["geometry"])
            if geometry.is_valid:
                centroid = geometry.centroid
                lon, lat = transformer.transform(centroid.x, centroid.y)
                region_name = geo_data['properties'].get('NUTS_NAME', 'Unknown Region')
                geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "region": region_name,
                        "value": row["bookings"]
                    },
                    "geometry": {"type": "Point", "coordinates": [lon, lat]}
                })
        else:
            print(f"No match for GEO: {row['GEO']}")

    geojson_data = {"type": "FeatureCollection", "features": geojson_features}
    print(json.dumps(geojson_data, indent=2))  # Debugging output
    return geojson_data






# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')  # Remove hardcoded geojson_data


@app.route('/data/<metric>/<year>/<month>/<category>')
def get_data(metric, year, month, category):
    dataset_map = {
        "stays": stay_data,
        "length_of_stay": length_of_stay_data,
        "nights_spent": nights_spend_data
    }
    dataset_group = dataset_map.get(metric)

    if not dataset_group:
        print(f"Invalid metric: {metric}")
        return jsonify({"error": "Invalid metric"}), 400

    if month.lower() == "yearly":
        period = "annual"
    else:
        period = "monthly"

    dataset_key = f"Stay_{category.upper()}_{period}" if metric == "stays" else \
                  f"LengthOfStay_{category.upper()}_{period}" if metric == "length_of_stay" else \
                  f"NightsSpend_{category.upper()}_{period}"

    dataset = dataset_group.get(dataset_key)
    if dataset is None or dataset.empty:
        print(f"Invalid category or period: {category}, {period}")
        return jsonify({"error": "Invalid category or period"}), 400

    if isinstance(dataset, pd.DataFrame):
        dataset = {"data": dataset}  # Wrap in dictionary for compatibility

    geojson_data = get_geojson_data(dataset, int(year), month)
    return jsonify(geojson_data)




if __name__ == '__main__':
    app.run(debug=True)