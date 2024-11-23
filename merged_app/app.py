from flask import Flask, render_template, jsonify, request
import pandas as pd
import eurostat
import json
from shapely.geometry import shape, Point
from pyproj import Transformer
import os

#app = Flask(__name__)

app = Flask(__name__, static_folder='/static')
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)





# Load GeoJSON data for regions and countries
with open('merged_app/nuts_2021.geojson', 'r', encoding='utf-8') as f:
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















##############################################################
######### ADDED FOR APP2 #####################################
##############################################################
# Assuming 'transformed_data' is the dictionary that holds your data, apply the country-level filter
country_view = {}

# Loop through each DataFrame in Yvars and apply the filter for 'GEO' length - each NUTS country code only has 2 string characters
for df_name, df in transformed_data.items():
    # Filter DataFrame to only include rows where 'GEO' has a string length of 2 (country-level only)
    country_view[df_name] = df[df["GEO"].str.len() == 2]















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

    # Filter to include only regions with 4-character GEO codes
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






####################################################################################################################
# NEW CODE STARTS HERE # 
####################################################################################################################
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import io
import base64


def heatmap(data, selected_country, selected_year, title1, title2):
    '''
    Plan: 2 heatmaps nebeneinander für DOM und FOR

    x-Achse: Monate 1-12
    y-Achse: Subregions (für AT: AT1, AT11, AT12, AT2, ...)

    Werte: für ein ausgewähltes Jahr (User)
    '''
    if data not in [Stay_df, NightsSpend_df, LengthOfStay_df]:
        raise ValueError("Invalid Dataframe")

    # Filter for required subregion data without the country itself
    heatmap_data = data[data["GEO"].str.contains(selected_country, case=False, na=False) &
                        (data["TIME_PERIOD"] == selected_year)]
    heatmap_data = heatmap_data[heatmap_data["GEO"] != selected_country]

    # Get all the subregions (abbreviations)
    subregions = heatmap_data['GEO'].unique()

    # Split the data for domestic and foreign visitors
    heatmap_data_domestic = heatmap_data[heatmap_data["c_resid"] == "DOM"]
    heatmap_data_foreign = heatmap_data[heatmap_data["c_resid"] == "FOR"]

    # Pivot the data to create a matrix for each subregion and month
    heatmap_data_domestic_pivot = heatmap_data_domestic.pivot_table(index='GEO', columns='month', values='bookings', aggfunc='sum', fill_value=0)
    heatmap_data_foreign_pivot = heatmap_data_foreign.pivot_table(index='GEO', columns='month', values='bookings', aggfunc='sum', fill_value=0)

    # Drop the "TOTAL" column if it exists (excludes column 13 if it's present)
    heatmap_data_domestic_pivot = heatmap_data_domestic_pivot.drop(columns='TOTAL', errors='ignore')
    heatmap_data_foreign_pivot = heatmap_data_foreign_pivot.drop(columns='TOTAL', errors='ignore')

    # Create a figure with two subplots side by side (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the first heatmap (total stays of domestic visitors)
    sns.heatmap(heatmap_data_domestic_pivot / 1000, annot=True, fmt=".1f", cmap="YlGnBu", 
                cbar_kws={'label': 'Total Stays of Domestic Visitors (in 1000s)'}, annot_kws={'size': 8}, ax=axes[0])
    axes[0].set_title(f"{title1} in {selected_country}", fontsize=16)

    # Plot the second heatmap (total stays of foreign visitors)
    sns.heatmap(heatmap_data_foreign_pivot / 1000, annot=True, fmt=".1f", cmap="YlGnBu", 
                cbar_kws={'label': 'Total Stays of Foreign Visitors (in 1000s)'}, annot_kws={'size': 8}, ax=axes[1])
    axes[1].set_title(f"{title2} in {selected_country}", fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Convert the plot to a PNG image and encode it in base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to avoid memory issues

    return img_base64


def generate_comparison_chart(subregion_data, subregion_1, subregion_2):
    # Fetch data for the two subregions
    subregion_1_data = subregion_data[subregion_1]['monthly_stays']
    subregion_2_data = subregion_data[subregion_2]['monthly_stays']
    
    # Plot comparison line chart
    months = [f"Month {i+1}" for i in range(12)]
    plt.figure(figsize=(10, 6))
    plt.plot(months, subregion_1_data, label=subregion_1, marker='o')
    plt.plot(months, subregion_2_data, label=subregion_2, marker='o')
    
    plt.title(f"Comparison of {subregion_1} and {subregion_2}")
    plt.xlabel("Month")
    plt.ylabel("Total Stays")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Convert the plot to a PNG image and encode it in base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to avoid memory issues
    
    return img_base64


############## methods for country level statistics:

'''
    method to get total stays in the selected year
'''
def get_total_stays_in_selected_year(selected_country, selected_year):
    total_stays = Stay_df[(Stay_df['GEO'] == selected_country) &
                          (Stay_df['c_resid'] == 'TOTAL') &
                          (Stay_df['month'] == 'TOTAL') &
                          (Stay_df['TIME_PERIOD'] == selected_year)]
    # Ensure there's exactly one row matching the criteria and return the value of 'bookings'
    if not total_stays.empty:
        return float(total_stays["bookings"].iloc[0])
    else:
        return None  # Return None if no matching row is found


'''
    method to calculate total stays from 2018 to 2024
'''
def get_total_stays(selected_country):
    total_stays = Stay_df[(Stay_df['GEO'] == selected_country) &
                          (Stay_df['c_resid'] == 'TOTAL') &
                          (Stay_df['month'] == 'TOTAL')]["bookings"].sum()
    return int(total_stays)


'''
    method to calculate percentage of foreign visitors
'''
def get_percentage_foreign_visitors(selected_country, selected_year):
    # Foreign visitors
    foreign_visitors = Stay_df[(Stay_df['GEO'] == selected_country) &
                                (Stay_df['c_resid'] == 'FOR') &
                                (Stay_df['month'] == 'TOTAL') &
                                (Stay_df['TIME_PERIOD'] == selected_year)]['bookings'].sum()
    
    # Total visitors
    total_visitors = Stay_df[(Stay_df['GEO'] == selected_country) &
                              (Stay_df['c_resid'] == 'TOTAL') &
                              (Stay_df['month'] == 'TOTAL') &
                              (Stay_df['TIME_PERIOD'] == selected_year)]['bookings'].sum()
    
    # Calculate percentage
    if total_visitors > 0:
        percentage_foreign = (foreign_visitors / total_visitors) * 100
        percentage_foreign = round(percentage_foreign, 2)  # Round to 2 decimal places
    else:
        percentage_foreign = 0
    return percentage_foreign


'''
    method for most popular month in a specific year
'''
def get_most_popular_month(selected_country, selected_year):
    # Filter out the rows where month == 'TOTAL'
    monthly_stays = Stay_df[(Stay_df['GEO'] == selected_country) &
                            (Stay_df['TIME_PERIOD'] == selected_year) &
                            (Stay_df['month'] != 'TOTAL')] \
                            .groupby('month')['bookings'].sum()
    
    # Find the month with the maximum number of bookings
    most_popular_month = monthly_stays.idxmax()
    most_popular_month_stays = monthly_stays.max()

    # Month mapping from number to month name
    month_mapping = {
        '01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
    }
    
    # Convert month number to month name
    most_popular_month_name = month_mapping.get(most_popular_month, 'Unknown Month')
    
    return f"{most_popular_month_name} ({int(most_popular_month_stays)} stays)"



'''
    method to calculate average length of stays
'''
def calculate_avg_length_of_stays(selected_country, selected_year):
    
    # Make copies of the original DataFrames
    data_stays = Stay_df.copy(deep=True)
    data_length = LengthOfStay_df.copy(deep=True)
    data_nights = NightsSpend_df.copy(deep=True)

    # Rename 'bookings' column to 'length' in data_length DataFrame
    data_length.rename(columns={'bookings': 'length'}, inplace=True)

    # Merge the dataframes on the specified columns
    stays_length = pd.merge(data_stays, data_length, how='inner', 
                            on=['GEO', 'month', 'TIME_PERIOD', 'c_resid'])

    # Drop the unwanted columns
    stays_length = stays_length.drop(['freq_x', 'unit_x', 'freq_y', 'unit_y'], axis=1)

    # Calculate the average length of stay
    stays_length["avg_length_of_stay"] = stays_length["length"] / stays_length["bookings"]

    # Filter the data for the selected country and year
    stays_length = stays_length[
        (stays_length['GEO'] == selected_country) & 
        (stays_length['TIME_PERIOD'] == selected_year) & 
        (stays_length['month'] == "TOTAL")
    ]

    # Group by c_resid to get the average length of stay for domestic and foreign visitors
    avg_by_resid = stays_length.groupby('c_resid')['avg_length_of_stay'].mean().reset_index()

    # Calculate the total average for all visitors
    total_avg_length = stays_length['avg_length_of_stay'].mean()

    # Round the results to 4 decimal places
    avg_by_resid_result = avg_by_resid.set_index('c_resid')['avg_length_of_stay'].to_dict()
    avg_by_resid_result['total'] = round(total_avg_length, 4)

    # Round the individual group averages to 3 decimal places
    avg_by_resid_result = {key: round(value, 3) for key, value in avg_by_resid_result.items()}

    # Formatting output in the desired format
    #print(f"Average Length of Stay:")
    #print(f"Domestic visitors: {avg_by_resid_result.get('DOM', 'N/A')} days")
    #print(f"Foreign visitors: {avg_by_resid_result.get('FOR', 'N/A')} days")
    #print(f"Total: {avg_by_resid_result.get('total', 'N/A')} days")
    
    return avg_by_resid_result


def find_subregion_with_highest_avg_length_of_stay(selected_country, selected_year):
    # Make copies of the original DataFrames
    data_stays = Stay_df.copy(deep=True)
    data_length = LengthOfStay_df.copy(deep=True)

    # Rename 'bookings' column to 'length' in data_length DataFrame
    data_length.rename(columns={'bookings': 'length'}, inplace=True)

    # Merge the dataframes on the specified columns
    stays_length = pd.merge(data_stays, data_length, how='inner', 
                            on=['GEO', 'month', 'TIME_PERIOD', 'c_resid'])

    # Drop the unwanted columns
    stays_length = stays_length.drop(['freq_x', 'unit_x', 'freq_y', 'unit_y'], axis=1)

    # Calculate the average length of stay, but handle NaN values (avoid division by zero)
    stays_length["avg_length_of_stay"] = stays_length["length"] / stays_length["bookings"]
    
    # Handle NaN values in the 'avg_length_of_stay' column: either drop rows or replace NaNs
    stays_length.dropna(subset=['avg_length_of_stay'], inplace=True)  # Drop rows with NaN in 'avg_length_of_stay'

    # Filter the data for the selected country, year, and the month "TOTAL"
    stays_length = stays_length[(
        stays_length['GEO'].str.contains(selected_country)) & 
        (stays_length['TIME_PERIOD'] == selected_year) & 
        (stays_length['month'] == "TOTAL") & 
        (stays_length['GEO'] != selected_country) &  # Exclude selected country
        (stays_length['c_resid'] == "TOTAL")]

    # Check if the stays_length DataFrame has any valid rows after filtering
    if stays_length.empty:
        return "No subregion data available for the selected filters."

    # Find the row with the highest avg_length_of_stay
    max_row = stays_length.loc[stays_length['avg_length_of_stay'].idxmax()]

    # Extract the subregion name and the corresponding value
    max_avg_length_subregion = max_row['GEO']
    max_avg_length_value = round(max_row['avg_length_of_stay'], 3)  # Round to 3 digits

    return f"{max_avg_length_subregion} ({max_avg_length_value})"


'''
    try for second visualization
'''
# Function to generate the line chart for the top 3 subregions with y-axis in thousands
def generate_nights_spent_line_charts(selected_country, selected_year):
    # Filter NightsSpend_df for the selected year, country, and exclude month = "TOTAL"
    df = NightsSpend_df[(NightsSpend_df['GEO'].str.contains(selected_country)) & 
                        (NightsSpend_df['TIME_PERIOD'] == selected_year) & 
                        (NightsSpend_df['month'] != 'TOTAL')]  # Exclude month = "TOTAL"
    
    # Rename bookings column to nights
    df = df.rename(columns={"bookings" : "nights"})
    
    # Filter out the selected country from the subregions (exclude the selected country itself)
    df = df[df['GEO'] != selected_country]
    
    # Group by subregion, month, and residency type (DOM/FOR) to calculate total nights spent
    df_grouped = df.groupby(['GEO', 'month', 'c_resid'])['nights'].sum().reset_index()

    # Get the top 3 subregions by total nights spent (across both domestic and foreign visitors)
    total_nights_per_subregion = df_grouped.groupby('GEO')['nights'].sum().nlargest(3).index
    top_subregions = df_grouped[df_grouped['GEO'].isin(total_nights_per_subregion)]

    # Prepare the line chart for domestic and foreign visitors
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    for i, c_resid in enumerate(['DOM', 'FOR']):
        ax = axes[i]
        for subregion in total_nights_per_subregion:
            subregion_data = top_subregions[(top_subregions['GEO'] == subregion) & 
                                            (top_subregions['c_resid'] == c_resid)]
            # Scale the nights by dividing by 1000 (to show in thousands)
            ax.plot(subregion_data['month'], subregion_data['nights'] / 1000, label=subregion)

        ax.set_title(f'Nights Spent by {c_resid} Visitors (Top 3 Subregions)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Nights Spent (in thousands)')
        ax.legend(title='Subregion')
        
        # Adjust y-axis ticks to show rounded values (for better readability)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
    # Save the plot to a PNG in memory
    img_bytes = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    return img_base64

#################################################################################################

@app.route('/country-insights', methods=['GET', 'POST'])
def index():
    # Set the default country (e.g., Germany 'DE')
    selected_country = request.form.get('country', 'AT')
    selected_year = request.form.get('year', '2024')  # Get the selected year from the form

    # Extract unique countries from the filtered country data (i.e., only countries with 2-character GEO codes)
    countries = set()
    for df_name, df in country_view.items():
        countries = countries.union(df['GEO'].unique())
    countries = sorted(countries)

    # Extract available years dynamically from the 'TIME_PERIOD' column in your data
    years = set()
    for df_name, df in country_view.items():
        years = years.union(df['TIME_PERIOD'].unique())
    years = sorted([str(year)[:4] for year in years])  # Convert to string and extract the year part

    # Generate Heatmap for Stays per Month
    stays_heatmap_base64 = heatmap(Stay_df, selected_country, selected_year, 'Monthly Stays (Domestic)', 'Monthly Stays (Foreign)')

    # Generate Line Chart for Nighs Spent by Top 3 Subregions (DOM/FORE)
    nights_spent_line_chart_base64 = generate_nights_spent_line_charts(selected_country, selected_year)

    # Calculate statistics for country-level statistics:
    total_stays = get_total_stays(selected_country)
    total_stays_in_selected_year = get_total_stays_in_selected_year(selected_country, selected_year)
    percentage_foreign_visitors = get_percentage_foreign_visitors(selected_country, selected_year)
    most_popular_month = get_most_popular_month(selected_country, selected_year)
    avg_length_of_stays = calculate_avg_length_of_stays(selected_country, selected_year)
    subregion_with_highest_avg_length = find_subregion_with_highest_avg_length_of_stay(selected_country, selected_year)

    # Collect statistics into a dictionary to pass to the template
    country_level_stats = {
        'total_stays': total_stays,
        'total_stays_in_selected_year': total_stays_in_selected_year,
        'percentage_foreign_visitors': percentage_foreign_visitors,
        'most_popular_month': most_popular_month,
        'avg_length_of_stays': avg_length_of_stays,
        'subregion_with_highest_avg_length': subregion_with_highest_avg_length,
    }

    # Collect subregion data to pass to the template (assuming `Stay_df` has this data)
    subregion_data = {}  # Add subregion data here based on your dataframe
    for subregion in Stay_df['GEO'].unique():
        subregion_data[subregion] = {
            'monthly_stays': Stay_df[Stay_df['GEO'] == subregion]['bookings'].tolist(),
        }

    return render_template(
        'index_app2.html',
        countries=countries,
        selected_country=selected_country,
        years=years,  # Pass the available years to the template
        stays_heatmap=stays_heatmap_base64,
        nights_spent_line_chart=nights_spent_line_chart_base64,  # Pass the nights spent line chart to the template
        selected_year=selected_year,  # Pass the selected year to the template
        country_level_stats=country_level_stats,  # Pass the calculated statistics
        subregion_data=subregion_data  # Pass the subregion data to the template
    )





############################################################################
##################### 3rd app ##############################################
############################################################################

 
import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid threading issues on macOS
import matplotlib.pyplot as plt
import time
import random




os.environ["PYTHONIOENCODING"] = "utf-8"






# Flask app initialization
'''
app = Flask(__name__, static_folder='/static')
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
'''




# Load predefined datasets
def load_datasets():
    """Load X and y from predefined CSV files."""
    X = pd.read_csv("merged_app/merged_dataX.csv", index_col=0)
    y = pd.read_csv("merged_app/merged_dataY.csv", index_col=0)
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()  # Ensure y is a 1D array
    return X, y

# Helper function for stratified splitting
def stratified_split(X, y, test_size=0.2, random_state=42, n_bins=5):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()
    return train_test_split(X, y, stratify=y_binned, test_size=test_size, random_state=random_state)

# Helper function for nested CV and hyperparameter tuning
def nested_cv(X_train, y_train, models, param_grids, inner_cv):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_results = {"Model": [], "Mean Test R²": [], "Mean Test MAE": []}

    for model_name, model in models.items():
        if param_grids[model_name]:
            randomized_search = RandomizedSearchCV(
                model,
                param_grids[model_name],
                scoring="r2",
                cv=inner_cv,
                n_iter=10,
                random_state=42,
                n_jobs=-1,
            )
            nested_cv_r2 = cross_val_score(randomized_search, X_train, y_train, cv=outer_cv, scoring="r2", n_jobs=-1)
            nested_cv_mae = cross_val_score(randomized_search, X_train, y_train, cv=outer_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        else:
            nested_cv_r2 = cross_val_score(model, X_train, y_train, cv=outer_cv, scoring="r2", n_jobs=-1)
            nested_cv_mae = cross_val_score(model, X_train, y_train, cv=outer_cv, scoring="neg_mean_absolute_error", n_jobs=-1)

        nested_results["Model"].append(model_name)
        nested_results["Mean Test R²"].append(np.mean(nested_cv_r2))
        nested_results["Mean Test MAE"].append(-np.mean(nested_cv_mae))

    return pd.DataFrame(nested_results)

# Helper function to generate a unique filename for each plot
def generate_unique_filename():
    timestamp = int(time.time())  # Current time as timestamp
    random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))  # Random 6-char string
    filename = f"cv_results_{timestamp}_{random_string}.png"
    return filename




@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    return render_template("index_app3.html")


@app.route("/process", methods=["GET", "POST"])
def process_data():
    # Load predefined datasets
    X, y = load_datasets()

    # Get selected cross-validation type
    cv_type = request.form.get('cv_type', 'simple')  # Default to 'simple' if not provided

    # Stratified train-test split
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Model definitions and hyperparameter grids
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
    }
    param_grids = {
        "Linear Regression": {},
        "Lasso Regression": {"alpha": [0.01, 0.1, 1, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    }

    # Define inner cross-validation (for hyperparameter tuning)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    if cv_type == 'nested':
        # Perform nested cross-validation
        results_df = nested_cv(X_train, y_train, models, param_grids, inner_cv)
    else:
        # Perform standard cross-validation
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        results_df = {"Model": [], "Mean Test R²": [], "Mean Test MAE": []}
        for model_name, model in models.items():
            cv_r2 = cross_val_score(model, X_train, y_train, cv=outer_cv, scoring="r2", n_jobs=-1)
            cv_mae = cross_val_score(model, X_train, y_train, cv=outer_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            results_df["Model"].append(model_name)
            results_df["Mean Test R²"].append(np.mean(cv_r2))
            results_df["Mean Test MAE"].append(-np.mean(cv_mae))
        results_df = pd.DataFrame(results_df)

    # Debug: Print the DataFrame to check data before plotting
    print("Results DataFrame:\n", results_df)

    # Generate a unique filename for the plot
    plot_filename = generate_unique_filename()

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(models))
    ax.bar(x - width / 2, results_df["Mean Test R²"], width, label="Mean Test R²")
    ax.bar(x + width / 2, results_df["Mean Test MAE"], width, label="Mean Test MAE")
    ax.set_xlabel("Models")
    ax.set_ylabel("Scores")
    ax.set_title(f"{cv_type.capitalize()} Cross-Validation Results")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45)
    ax.legend()

    # Ensure the static directory exists and is accessible
    static_folder_path = '/static'
    os.makedirs(static_folder_path, exist_ok=True)

    # Save plot to the static directory
    plot_path = os.path.join(static_folder_path, plot_filename)
    plt.tight_layout()  # Apply tight layout before saving
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    # Debug: Verify if the plot is saved correctly
    if os.path.exists(plot_path):
        print(f"Plot saved successfully at {plot_path}")
    else:
        print("Error: Plot was not saved.")

    # Select best model and evaluate on test set
    best_model_name = results_df.loc[results_df["Mean Test R²"].idxmax(), "Model"]
    best_model = models[best_model_name]
    if param_grids[best_model_name]:
        best_model = RandomizedSearchCV(
            best_model, param_grids[best_model_name], scoring="r2", cv=inner_cv, n_iter=10, random_state=42, n_jobs=-1
        ).fit(X_train, y_train).best_estimator_
    else:
        best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    test_performance = {
        "R²": r2_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "MAPE": mape_test,
    }

    return render_template(
        "results_app3.html",
        results=results_df.to_html(),
        plot_filename=plot_filename,  # Pass the unique filename
        best_model=best_model_name,
        test_performance=test_performance,
    )









if __name__ == '__main__':
    app.run(debug=True)
