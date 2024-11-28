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

    #filtered = filtered[filtered["GEO"].str.len() > 2]
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
                
                # Get the mapped name from nuts_mapping
                nuts_mapped_name = nuts_mapping.get(row["GEO"], "Unknown Name")

                # Only add mapped_name if it is different from region_name
                mapped_name = nuts_mapped_name if nuts_mapped_name != region_name else None

                geojson_feature = {
                    "type": "Feature",
                    "properties": {
                        "region": region_name,
                        "value": row["bookings"]
                    },
                    "geometry": {"type": "Point", "coordinates": [lon, lat]}
                }

                if mapped_name:
                    geojson_feature["properties"]["mapped_name"] = mapped_name

                geojson_features.append(geojson_feature)
                
        else:
            print(f"No match for GEO: {row['GEO']}")

    geojson_data = {"type": "FeatureCollection", "features": geojson_features}
    print(json.dumps(geojson_data, indent=2))  # Debugging output
    return geojson_data

# Load NUTS mapping data
with open('nuts_mapping.json', 'r', encoding='utf-8') as f:
    nuts_mapping = json.load(f)


# --- ROUTES ---
@app.route('/')
def home():
    return render_template('home.html')

# Route to render the country-level map page
@app.route('/map/countries')
def countries_map():
    return render_template('countries_map.html')

# Route to render the region-level map page
@app.route('/map/regions')
def regions_map():
    return render_template('regions_map.html')

@app.route('/data/countries/<metric>/<year>/<month>/<category>')
def get_countries_data(metric, year, month, category):
    dataset_map = {
        "stays": stay_data,
        "length_of_stay": length_of_stay_data,
        "nights_spent": nights_spend_data
    }
    dataset_group = dataset_map.get(metric)

    if not dataset_group:
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
        return jsonify({"error": "Invalid category or period"}), 400

    # Filter to include only countries with 2-character GEO codes
    dataset = dataset[dataset['GEO'].str.len() == 2]

    geojson_data = get_geojson_data(dataset, int(year), month)
    return jsonify(geojson_data)


@app.route('/data/regions/<metric>/<year>/<month>/<category>')
def get_regions_data(metric, year, month, category):
    dataset_map = {
        "stays": stay_data,
        "length_of_stay": length_of_stay_data,
        "nights_spent": nights_spend_data
    }
    dataset_group = dataset_map.get(metric)

    if not dataset_group:
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
        return jsonify({"error": "Invalid category or period"}), 400

    #filter to include only regions with 4-character GEO codes:
    dataset = dataset[dataset['GEO'].str.len() == 4]

    geojson_data = get_geojson_data(dataset, int(year), month)
    return jsonify(geojson_data)

# --- Route for Top 10 Visited Countries Page ---
@app.route('/top10/countries')
def top10_countries():
    return render_template('top10_countries.html')

@app.route('/top10/countries/data', methods=['GET'])
def top10_countries_data():
    metric = request.args.get('metric', 'stays')
    year = request.args.get('year', '2022')
    month = request.args.get('month', 'January')
    category = request.args.get('category', 'DOM')

    dataset_map = {
        "stays": stay_data,
        "length_of_stay": length_of_stay_data,
        "nights_spent": nights_spend_data
    }
    dataset_group = dataset_map.get(metric)

    dataset_key = f"Stay_{category.upper()}_monthly" if metric == "stays" else \
                  f"LengthOfStay_{category.upper()}_monthly" if metric == "length_of_stay" else \
                  f"NightsSpend_{category.upper()}_monthly"

    dataset = dataset_group.get(dataset_key)
    if dataset is None or dataset.empty:
        return jsonify([])

    # Filter data based on year and month
    dataset['year'] = dataset['month'].dt.year
    dataset['month_name'] = dataset['month'].dt.strftime('%B')
    filtered_data = dataset[(dataset['year'] == int(year)) & (dataset['month_name'] == month)]

    # Filter to include only countries with 2-character GEO codes for top10_countries_data
    filtered_data = filtered_data[filtered_data['GEO'].str.len() == 2]

    # Group by common GEO code and get the top 10 visited countries
    top_countries = (filtered_data.groupby('GEO')['bookings'].sum()
                    .reset_index()
                    .sort_values(by='bookings', ascending=False).head(10))

    # Replace NUTS codes with country names
    top_countries['GEO'] = top_countries['GEO'].map(nuts_mapping)

    # Convert result to JSON format
    top_countries_list = top_countries.to_dict(orient='records')
    return jsonify(top_countries_list)

# --- Route for Top 10 Visited Regions Page ---
@app.route('/top10/regions')
def top10_regions():
    return render_template('top10_regions.html')

@app.route('/top10/regions/data', methods=['GET'])
def top10_regions_data():
    metric = request.args.get('metric', 'stays')
    year = request.args.get('year', '2022')
    month = request.args.get('month', 'January')
    category = request.args.get('category', 'DOM')

    dataset_map = {
        "stays": stay_data,
        "length_of_stay": length_of_stay_data,
        "nights_spent": nights_spend_data
    }
    dataset_group = dataset_map.get(metric)

    dataset_key = f"Stay_{category.upper()}_monthly" if metric == "stays" else \
                  f"LengthOfStay_{category.upper()}_monthly" if metric == "length_of_stay" else \
                  f"NightsSpend_{category.upper()}_monthly"

    dataset = dataset_group.get(dataset_key)
    if dataset is None or dataset.empty:
        return jsonify([])

    # Filter data based on year and month
    dataset['year'] = dataset['month'].dt.year
    dataset['month_name'] = dataset['month'].dt.strftime('%B')
    filtered_data = dataset[(dataset['year'] == int(year)) & (dataset['month_name'] == month)]

    # Filter to include only regions with 4-character GEO codes for top10_regions_data
    filtered_data = filtered_data[filtered_data['GEO'].str.len() == 4]

    # Group by common GEO code and get the top 10 visited regions
    top_regions = (filtered_data.groupby('GEO')['bookings'].sum()
                   .reset_index()
                   .sort_values(by='bookings', ascending=False).head(10))

    # Replace NUTS codes with region names
    top_regions['GEO'] = top_regions['GEO'].map(nuts_mapping)

    # Convert result to JSON format
    top_regions_list = top_regions.to_dict(orient='records')
    return jsonify(top_regions_list)

if __name__ == '__main__':
    app.run(debug=True)
