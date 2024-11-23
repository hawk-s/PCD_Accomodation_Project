# Import necessary libraries for Flask
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# --- Your Original Code --- #
# Add the code you provided here for data preprocessing...

# DATA LOAD
# Import the necessary libraries and get the tour accommodation data
import eurostat
import pandas as pd
data_analysed = eurostat.get_data_df('tour_ce_omn12')

# DATA PREPARATION, EXPLORATION AND TRANSFORMATION

# Replace the month column to make it usuable in our analysis
data_analysed["month"] = data_analysed["month"].str.replace(r'M',"")
# Rename the column to facilitate its use throughout the analysis
data_analysed = data_analysed.rename(columns = {"geo\\TIME_PERIOD": "GEO"})
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

# Dictionary to store each subset with descriptive names
subset_dict = {}

# List of main subsets and corresponding descriptive keys
Yvars = {
    "Stay": Stay_df,
    "LengthOfStay": LengthOfStay_df,
    "NightsSpend": NightsSpend_df
}

# Loop through each main subset and create subsets based on 'c_resid'
for name, df in Yvars.items():
    subset_dict[f"{name}_DOM"] = df[df["c_resid"] == "DOM"]
    subset_dict[f"{name}_FOR"] = df[df["c_resid"] == "FOR"]
    subset_dict[f"{name}_TOTAL"] = df[df["c_resid"] == "TOTAL"]


subset_keys_list = list(subset_dict.keys())

# Optional: check the keys in the dictionary to see the subsets created
print(subset_dict.keys())

# Identify missing values and duplicates

# Define a function to check for missing values and duplicates in each subset table
def identify_data_issues(data_dict):
    for key, df in data_dict.items():
        print(f"\nData check for: {key}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("Missing values per column:")
        print(missing_values[missing_values > 0])  # Only show columns with missing values
        
        # Check for duplicates
        duplicate_rows = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_rows}")

# Run the function on the dictionary of subsets
identify_data_issues(subset_dict)

# Split the dataset into periods (months and years) to conduct an analysis on different levels


# Iterate over each subset in Yvars dictionary
transformed_data = {}
for name, df in subset_dict.items():
    
    # Separate monthly and annual rows
    monthly = df[df["month"] != 'TOTAL'].copy()  # Copy to avoid warnings when modifying
    annual = df[df["month"] == 'TOTAL'].copy()

    # Convert 'TIME_PERIOD' and 'month' columns to datetime format
    monthly['month'] = pd.to_datetime(monthly['TIME_PERIOD'] + "-" + monthly['month'], format="%Y-%m").dt.to_period("M")
    annual['month'] = pd.to_datetime(annual['TIME_PERIOD'], format="%Y").dt.to_period("Y")
    
    # Store the transformed DataFrames in the dictionary with meaningful keys
    transformed_data[f"{name}_monthly"] = monthly
    transformed_data[f"{name}_annual"] = annual

# Country - level exploration - start to explore the data on a country-level basis

country_view = {}

# Loop through each DataFrame in Yvars and apply the filter for 'GEO' length - each NUTS country code only has 2 string characters
for df_name, df in transformed_data.items():
    # Filter DataFrame to only include rows where 'GEO' has a string length of 2
    country_view[df_name] = df[df["GEO"].str.len() == 2]

# FURTHER ANALYSIS OF THE MISSING VALUES

# Function to analyze missing data by Year and Country for country-level filtered data
def analyze_missing_data_country_level(filtered_data_dict):
    # Lists to collect missing data information
    country_level_info = []
    year_level_info = []

    # Loop through each subset in the filtered dictionary (country-level only)
    for key, df in filtered_data_dict.items():
        # Find rows with any missing values
        missing_rows = df[df.isna().any(axis=1)]
        
        # Check for missing values by Country (GEO)
        for geo in df['GEO'].unique():
            geo_missing_count = len(missing_rows[missing_rows['GEO'] == geo])
            country_level_info.append({
                'Subset': key,
                'GEO': geo,
                'Missing_Count': geo_missing_count
            })

        # Check for missing values by Year (TIME_PERIOD)
        for year in df['TIME_PERIOD'].unique():
            year_missing_count = len(missing_rows[missing_rows['TIME_PERIOD'] == year])
            year_level_info.append({
                'Subset': key,
                'TIME_PERIOD': year,
                'Missing_Count': year_missing_count
            })

    # Create separate DataFrames for country-level and year-level missing data
    country_level_df = pd.DataFrame(country_level_info)
    year_level_df = pd.DataFrame(year_level_info)

    # Return the DataFrames
    return country_level_df, year_level_df

# Assuming 'transformed_data' is the dictionary that holds your data, apply the country-level filter
country_view = {}

# Loop through each DataFrame in Yvars and apply the filter for 'GEO' length - each NUTS country code only has 2 string characters
for df_name, df in transformed_data.items():
    # Filter DataFrame to only include rows where 'GEO' has a string length of 2 (country-level only)
    country_view[df_name] = df[df["GEO"].str.len() == 2]

# Run the missing data analysis on the filtered country-level data
country_level_df, year_level_df = analyze_missing_data_country_level(country_view)

'''
# Display the missing data summaries
print("Country-Level Missing Data:")
print(country_level_df)
print("\nYear-Level Missing Data:")
print(year_level_df)
'''
# DATA CLEANING
# Apply dropna to each DataFrame in the dictionary
transformed_data_cleaned = {
    key: df.dropna()  # Remove NaN values from each DataFrame
    for key, df in transformed_data.items()
}


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
        return int(total_stays["bookings"].iloc[0])
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

@app.route('/', methods=['GET', 'POST'])
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
        'index.html',
        countries=countries,
        selected_country=selected_country,
        years=years,  # Pass the available years to the template
        stays_heatmap=stays_heatmap_base64,
        nights_spent_line_chart=nights_spent_line_chart_base64,  # Pass the nights spent line chart to the template
        selected_year=selected_year,  # Pass the selected year to the template
        country_level_stats=country_level_stats,  # Pass the calculated statistics
        subregion_data=subregion_data  # Pass the subregion data to the template
    )



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

