from flask import Flask, render_template, request
import eurostat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load Eurostat dataset for tourism
tourism_data = eurostat.get_data_df('tour_ce_omn12')

# Filter dataset for country-level data (two-letter codes)
country_data = tourism_data[tourism_data['geo\\TIME_PERIOD'].str.len() == 2]

# Extract unique country codes from the dataset
unique_countries = country_data['geo\\TIME_PERIOD'].unique()

# Function to generate line chart for multiple countries
def generate_combined_chart(selected_countries, selected_years):
    # Filter data for the selected countries and years
    data_selected = country_data[country_data['geo\\TIME_PERIOD'].isin(selected_countries)]

    # Ensure the 'month' column is ordered correctly
    data_selected['month'] = pd.Categorical(data_selected['month'], categories=[f"M{str(i).zfill(2)}" for i in range(1, 13)], ordered=True)

    # Filter the data by selected years, keeping only columns for those years
    year_columns = [col for col in data_selected.columns if col.startswith(tuple(selected_years))]
    data_filtered = data_selected[['geo\\TIME_PERIOD', 'month'] + year_columns]

    # Melt the data to long format for plotting
    data_long = data_filtered.melt(id_vars=['geo\\TIME_PERIOD', 'month'], var_name='Year', value_name='Value')

    # Extract the year from the column name to use as the year label in the plot
    data_long['Year'] = data_long['Year'].str.extract('(\d{4})').astype(int)

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='month', y='Value', hue='geo\\TIME_PERIOD', style='Year', data=data_long, marker="o")
    plt.title("Monthly Trends for Selected Countries and Years")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend(title="Country - Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Convert the plot to a PNG image and encode it in base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return img_base64


@app.route('/')
def index():
    # Main navigation page
    return render_template('index.html')

@app.route('/combined_chart', methods=['GET', 'POST'])
def combined_chart():
    selected_countries = request.form.getlist('country') if request.method == 'POST' else ['AT', 'DE']
    selected_years = request.form.getlist('year') if request.method == 'POST' else ['2020', '2021']
    
    # Generate the plot image based on selected countries and years
    plot_img = generate_combined_chart(selected_countries, selected_years)
    
    # Extract unique years from the dataset for the dropdown
    all_years = [col[:4] for col in country_data.columns[6:] if col.startswith('20')]  # Assuming years start with '20'
    unique_years = sorted(set(all_years))

    return render_template('combined_chart.html', plot_img=plot_img, countries=unique_countries, selected_countries=selected_countries, years=unique_years, selected_years=selected_years)

if __name__ == '__main__':
    app.run(debug=True)
