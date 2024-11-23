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

# Flask app initialization
app = Flask(__name__, static_folder='/Users/arnovanheule/Documents/FinalProject/static')
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load predefined datasets
def load_datasets():
    """Load X and y from predefined CSV files."""
    X = pd.read_csv("/Users/arnovanheule/Documents/FinalProject/data/merged_dataX.csv", index_col=0)
    y = pd.read_csv("/Users/arnovanheule/Documents/FinalProject/data/merged_dataY.csv", index_col=0)
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

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

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
    static_folder_path = '/Users/arnovanheule/Documents/FinalProject/static'
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


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
