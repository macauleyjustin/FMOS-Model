import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
import re
import logging
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import pandas as pd
import json
import os
import threading
import http.server
import socketserver
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. SET UP LOGGING ---
logging.basicConfig(
    filename='advanced_snow_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# --- 2. PERSISTENCE FUNCTIONS (Unchanged + Config) ---
def load_historical_weather(file_path='historical_weather.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.fillna(0)  # Impute NaN to avoid scaling errors
        logging.info(f"Loaded {len(df)} historical weather records.")
        return df
    logging.info("No historical weather file found. Starting fresh.")
    return pd.DataFrame(columns=['date', 'temperature_2m_min', 'snowfall_sum', 'precip'])

def save_historical_weather(df, file_path='historical_weather.csv'):
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {len(df)} historical weather records.")

def load_teleconnections(file_path='teleconnections.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info("Loaded saved teleconnection data.")
        return {k: defaultdict(dict, v) if k not in ['oni', 'mjo_amp'] else v for k, v in data.items()}
    return {'oni': {}, 'ao': defaultdict(dict), 'nao': defaultdict(dict), 'pna': defaultdict(dict), 'pdo': defaultdict(dict), 'qbo': defaultdict(dict), 'ssn': defaultdict(dict), 'mjo_amp': {}}

def save_teleconnections(data, file_path='teleconnections.json'):
    serializable = {k: dict(v) if k not in ['oni', 'mjo_amp'] else v for k, v in data.items()}
    with open(file_path, 'w') as f:
        json.dump(serializable, f)
    logging.info("Saved teleconnection data.")

def load_config(file_path='config.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded config: Location = {config.get('location')}")
        return config.get('location', None)
    return None

def save_config(location, file_path='config.json'):
    with open(file_path, 'w') as f:
        json.dump({'location': location}, f)
    logging.info(f"Saved config: Location = {location}")

# --- Additional Feature: Fetch Current Weather ---
def fetch_current_weather(lat, lon):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat, 'longitude': lon,
        'current': 'temperature_2m,weather_code,snowfall',
        'timezone': 'auto'
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()['current']
        temp_c = data['temperature_2m']
        temp_f = temp_c * 9/5 + 32
        snowfall_cm = data.get('snowfall', 0)
        snowfall_in = snowfall_cm / 2.54
        weather_code = data['weather_code']  # WMO code
        weather_desc = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 51: "Light drizzle", 61: "Slight rain", 71: "Slight snow",
            # Add more as needed
        }.get(weather_code, "Unknown")
        return f"Temperature: {temp_f:.1f}°F | Snowfall: {snowfall_in:.1f} in | Conditions: {weather_desc}"
    except Exception as e:
        logging.error(f"Error fetching current weather: {e}")
        return "Current weather unavailable."

# --- Additional Feature: Generate Historical Snow Plot ---
def generate_snow_plot(historical_df):
    if historical_df.empty:
        return None
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df['year'] = historical_df['date'].dt.year
    yearly_snow = historical_df.groupby('year')['snowfall_sum'].sum() / 2.54  # to inches
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_snow.index, yearly_snow.values, marker='o')
    plt.title('Historical Yearly Snowfall (inches)')
    plt.xlabel('Year')
    plt.ylabel('Total Snowfall')
    plt.grid(True)
    plot_path = 'historical_snow.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# --- New Feature: Generate First Snow Plot ---
def generate_first_snow_plot(first_snow_dict, predicted_doy, current_year):
    if not first_snow_dict:
        return None
    years = sorted(first_snow_dict.keys())
    doys = [first_snow_dict[y] for y in years]
    plt.figure(figsize=(10, 5))
    plt.scatter(years, doys, label='Historical First Snow DOY', color='blue')
    if predicted_doy is not None:
        plt.scatter(current_year, predicted_doy, label='Predicted First Snow DOY', color='red', marker='x')
    plt.title('Historical and Predicted First Snowfall Days')
    plt.xlabel('Year')
    plt.ylabel('Day of Year (DOY)')
    plt.legend()
    plt.grid(True)
    plot_path = 'first_snow_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# --- HTML Generation Function (Updated with new features) ---
def generate_html(location, driver_status_string, first_snow_message, seasonal_message, last_prediction_date, next_prediction_date, current_weather, plot_path, first_snow_plot_path, predicted_timestamp_ms):
    plot_img = f'<img src="{plot_path}" alt="Historical Snowfall Plot" style="max-width:100%;">' if plot_path else ''
    first_plot_img = f'<img src="{first_snow_plot_path}" alt="First Snowfall Plot" style="max-width:100%;">' if first_snow_plot_path else ''
    countdown_script = ""
    if predicted_timestamp_ms is not None:
        countdown_script = f"""
        <div id="countdown"></div>
        <script>
        const targetDate = new Date({predicted_timestamp_ms});
        function updateCountdown() {{
          const now = new Date().getTime();
          const distance = targetDate - now;
          if (distance < 0) {{
            document.getElementById("countdown").innerHTML = "Snowfall has arrived!";
            return;
          }}
          const days = Math.floor(distance / (1000 * 60 * 60 * 24));
          const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
          const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
          const seconds = Math.floor((distance % (1000 * 60)) / 1000);
          document.getElementById("countdown").innerHTML = `${{days}}d ${{hours}}h ${{minutes}}m ${{seconds}}s till next snowfall`;
        }}
        setInterval(updateCountdown, 1000);
        updateCountdown();
        </script>
        """
    else:
        countdown_script = "<p>No prediction available for countdown.</p>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Snowfall Predictions for {location}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Snowfall Predictions for {location}</h1>
        <p>Last prediction made: {last_prediction_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Next prediction: {next_prediction_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Current Weather</h2>
        <p>{current_weather}</p>
        
        <h2>Current Climate Drivers</h2>
        <table>
            <tr><th>Driver</th><th>Value</th></tr>
            {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(driver_status_string.items())])}
        </table>
        
        <h2>First Snowfall Forecast</h2>
        <p>{first_snow_message}</p>
        {countdown_script}
        
        <h2>Seasonal Snowfall Outlook</h2>
        <p>{seasonal_message}</p>
        
        <h2>Historical Snowfall Trend</h2>
        {plot_img}
        
        <h2>First Snowfall Days Trend</h2>
        {first_plot_img}
    </body>
    </html>
    """
    with open('index.html', 'w') as f:
        f.write(html_content)
    logging.info("Updated index.html with latest predictions.")

# --- Web Server Function ---
def run_server():
    PORT = 8080  # Changed to avoid port conflict
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        logging.info(f"Serving web server at http://localhost:{PORT}")
        httpd.serve_forever()

# --- 3. TELECONNECTION & DRIVER DATA FETCHERS ---

def _fetch_historical_oni():
    ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    logging.info("Fetching historical ENSO (ONI) data from NOAA...")
    try:
        response = requests.get(ONI_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        header_line = None
        data_start_index = -1
        for i, line in enumerate(lines):
            if "SEAS" in line and "YR" in line and "ANOM" in line:
                header_line = line.split()
                data_start_index = i
