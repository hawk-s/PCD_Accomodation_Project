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

# Helper function to generate the plot for a given country
def generate_plot(country_code):
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

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default country is Austria (AT) or the first in the list
    country_code = 'AT' if 'AT' in unique_countries else unique_countries[0]

    if request.method == 'POST':
        country_code = request.form.get('country')

    # Generate the plot for the selected country
    plot_img = generate_plot(country_code)

    # Render the HTML page with the plot and dropdown
    return render_template('index.html', plot_img=plot_img, countries=unique_countries, selected_country=country_code)

if __name__ == '__main__':
    app.run(debug=True)
