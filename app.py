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
def generate_line_chart(selected_countries):
    # Filter data for the selected countries only
    data_selected = country_data[country_data['geo\\TIME_PERIOD'].isin(selected_countries)]

    # Ensure the 'month' column is ordered correctly
    data_selected['month'] = pd.Categorical(data_selected['month'], categories=[f"M{str(i).zfill(2)}" for i in range(1, 13)], ordered=True)

    # Melt the data for plotting
    data_long = data_selected.melt(id_vars=['geo\\TIME_PERIOD', 'month'], value_vars=data_selected.columns[6:], 
                                   var_name='Year', value_name='Value')

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='month', y='Value', hue='geo\\TIME_PERIOD', style='Year', data=data_long, marker="o")
    plt.title("Monthly Trends Over the Years for Selected Countries")
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

# Function to generate trends chart for a single country
def generate_trends_chart(country_code):
    # Filter data based on the selected country
    data_region = tourism_data[tourism_data['geo\\TIME_PERIOD'] == country_code]

    # Ensure the 'month' column is ordered as a categorical type
    data_region['month'] = pd.Categorical(data_region['month'], categories=[f"M{str(i).zfill(2)}" for i in range(1, 13)], ordered=True)

    # Melt the data to long format for Seaborn
    data_long = data_region.melt(id_vars=['month'], value_vars=data_region.columns[6:], var_name='Year', value_name='Value')

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='month', y='Value', hue='Year', data=data_long, marker="o")
    plt.title(f"Monthly Trends Over the Years for Region {country_code}")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend(title="Year")

    # Convert the plot to a PNG image and encode it in base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/')
def index():
    # Main navigation page
    return render_template('index.html')

@app.route('/line_chart', methods=['GET', 'POST'])
def line_chart():
    selected_countries = request.form.getlist('country') if request.method == 'POST' else ['AT', 'DE']
    plot_img = generate_line_chart(selected_countries)
    return render_template('line_chart.html', plot_img=plot_img, countries=unique_countries, selected_countries=selected_countries)

@app.route('/trends_chart', methods=['GET', 'POST'])
def trends_chart():
    country_code = request.form.get('country') if request.method == 'POST' else 'AT'
    plot_img = generate_trends_chart(country_code)
    return render_template('trends_chart.html', plot_img=plot_img, countries=unique_countries, selected_country=country_code)

if __name__ == '__main__':
    app.run(debug=True)
