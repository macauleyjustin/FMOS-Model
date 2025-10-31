#!/usr/bin/env python3
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
import re
import logging
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 1. SET UP LOGGING ---
logging.basicConfig(
    filename='snow_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)


# --- 2. COMPLETELY REWRITTEN: ENSO (ONI) DATA FETCHER ---
def _fetch_historical_oni():
    """
    Fetches the Oceanic Niño Index (ONI) from NOAA.
    Returns a dict of {year: oni_value} for the 'SON' season (Sept-Oct-Nov)
    and the most recent ONI value available.
    """
    ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    logging.info("Fetching historical ENSO (ONI) data from NOAA...")
    try:
        response = requests.get(ONI_URL, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        
        header_line = None
        data_start_index = -1
        
        # Find the header line by looking for "SEAS" and "YR"
        for i, line in enumerate(lines):
            if "SEAS" in line and "YR" in line and "ANOM" in line:
                header_line = line.split()
                data_start_index = i + 1
                break
        
        if header_line is None:
            raise ValueError("Could not find header (SEAS, YR, ANOM) in ONI data file.")
            
        # Find the column indices
        seas_col = header_line.index("SEAS")
        yr_col = header_line.index("YR")
        anom_col = header_line.index("ANOM")
        
        historical_son_data = {}
        latest_oni_value = None
        latest_oni_year = 0
        latest_oni_period = ""

        for line in lines[data_start_index:]:
            if not line.strip():
                continue
            
            parts = line.split()
            if not parts or len(parts) <= max(seas_col, yr_col, anom_col):
                continue
                
            try:
                year = int(parts[yr_col])
                season = parts[seas_col]
                anomaly_str = parts[anom_col]
                
                if anomaly_str != '-999.9':
                    anomaly = float(anomaly_str)
                    
                    # Store the 'SON' (Sept-Oct-Nov) value for historical training
                    if season == "SON":
                        historical_son_data[year] = anomaly
                    
                    # Store the most recent valid value for prediction
                    latest_oni_value = anomaly
                    latest_oni_year = year
                    latest_oni_period = season
                        
            except (ValueError, IndexError):
                continue # Skip malformed lines

        if latest_oni_value is None:
            raise ValueError("Could not parse any valid ONI values.")

        logging.info(f"Successfully parsed ONI data. Most recent: {latest_oni_period} {latest_oni_year} (ONI: {latest_oni_value:.2f})")
        return historical_son_data, latest_oni_value

    except Exception as e:
        logging.error(f"Failed to fetch or parse ONI data: {e}")
        return {}, 0.0

# --- 3. ALL HELPER FUNCTIONS ---

def get_lat_lon(location):
    """Fetches latitude and longitude from a location name."""
    try:
        if ',' in location and all('.' in part for part in location.replace(',', ' ').split()):
            return map(float, location.split(','))
        
        query_location = location
        if re.match(r'^\d{5}$', location.strip()):
            logging.info(f"Detected 5-digit zip code. Forcing US search for NWS API.")
            query_location = f"{location}, USA"

        headers = {'User-Agent': 'FMOS_ML_Logical/1.0 (https://example.com; contact@example.com)'}
        url = f"https://nominatim.openstreetmap.org/search?q={query_location}&format=json"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            logging.info(f"Found location: {data[0]['display_name']}")
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            raise ValueError(f"No location found for '{query_location}'")
            
    except Exception as e:
        logging.error(f"Error getting coordinates: {e}")
        return 42.7325, -84.5555

def get_historical_data_for_ml(lat, lon, years=30):
    """
    Fetches 30 years of daily data PLUS historical ENSO data.
    """
    logging.info(f"Fetching {years} years of climate data for ML model training...")
    
    historical_son_data, current_oni = _fetch_historical_oni()
    
    current_year = datetime.now().year
    start_date = f"{current_year - years}-01-01"
    end_date = f"{current_year - 1}-12-31"
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_min,snowfall_sum',
        'temperature_unit': 'fahrenheit',
        'precipitation_unit': 'inch',
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        daily_data = data['daily']
        times = daily_data['time']
        temp_mins = daily_data['temperature_2m_min']
        snow_sums = daily_data['snowfall_sum']
        
        X_train, y_train_snow = [], []
        first_snow_doy_list = []
        year_data = {}
        
        for i, date_str in enumerate(times):
            if temp_mins[i] is None or snow_sums[i] is None:
                continue
            
            # --- THIS IS THE FIX ---
            # Strip whitespace from the date string before parsing
            date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            # --- END OF FIX ---
            
            doy = date.timetuple().tm_yday
            year = date.year
            
            oni_val = historical_son_data.get(year, 0.0)
            
            if date.month >= 8 or date.month <= 5:
                X_train.append([doy, oni_val])
                y_train_snow.append(1 if snow_sums[i] > 0.1 else 0)

            if year not in year_data:
                year_data[year] = []
            year_data[year].append((date, temp_mins[i], snow_sums[i]))

        for year, readings in year_data.items():
            readings.sort(key=lambda x: x[0])
            for date, temp, snow in readings:
                if date.month >= 8:
                    if temp <= 32.0 and snow > 0.1:
                        first_snow_doy_list.append(date.timetuple().tm_yday)
                        break 
        
        if not X_train:
            raise ValueError("No historical data found to train model.")
            
        logging.info(f"Found {len(first_snow_doy_list)} historical first-snow events.")
        logging.info(f"Training ML model on {len(X_train)} daily records (now including ENSO data).")
        
        return np.array(X_train), np.array(y_train_snow), np.array(first_snow_doy_list), current_oni
        
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}", exc_info=True) # Added exc_info for more detail
        return None, None, None, 0.0

def train_snow_probability_model(X_train, y_train):
    """
    Trains a Logistic Regression model to predict the probability of snow
    given Day of Year (DOY) and ONI value.
    """
    model = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear')
    
    doy_features = X_train[:, 0]
    oni_features = X_train[:, 1].reshape(-1, 1) 
    
    doy_radians = (doy_features * (2 * np.pi / 365.25))
    X_cyclic_doy = np.column_stack((
        np.sin(doy_radians),
        np.cos(doy_radians)
    ))
    
    scaler = StandardScaler()
    X_scaled_oni = scaler.fit_transform(oni_features)
    
    X_features_final = np.column_stack((X_cyclic_doy, X_scaled_oni))
    
    model.fit(X_features_final, y_train)
    
    return model, scaler

def predict_first_snow_with_ml(model, scaler, current_doy, current_oni, threshold=0.25):
    """
    Uses the trained ML model to find the first day after
    current_doy where snow probability exceeds the threshold.
    """
    scaled_oni_val = scaler.transform([[current_oni]])[0, 0]
    
    for day_offset in range(1, 180):
        doy = current_doy + day_offset
        if doy > 365:
            doy -= 365
        
        doy_rad = doy * (2 * np.pi / 365.25)
        X_test = np.array([[
            np.sin(doy_rad),
            np.cos(doy_rad),
            scaled_oni_val
        ]])
        
        probability = model.predict_proba(X_test)[0, 1]
        
        if probability >= threshold:
            return current_doy + day_offset
            
    return None 

def fetch_nws_forecast_with_snow(lat, lon):
    """Fetches the 7-day NWS forecast."""
    logging.info("Fetching 7-day NWS forecast...")
    try:
        headers = {'User-Agent': 'FMOS_ML_Logical/1.0 (https://example.com; contact@example.com)'}
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        response = requests.get(points_url, headers=headers, timeout=10)
        response.raise_for_status()
        forecast_url = response.json()['properties']['forecastGridData']
        
        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_resp.raise_for_status()
        props = forecast_resp.json()['properties']
        
        snow_data = props['snowfallAmount']['values']
        temp_data = props['minTemperature']['values']
        
        daily_temps = {}
        for entry in temp_data:
            start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
            if entry['value'] is not None:
                daily_temps[start_time.date()] = entry['value'] * 9/5 + 32

        daily_snow = {}
        for entry in snow_data:
            start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
            if entry['value'] is not None:
                date_key = start_time.date()
                if date_key not in daily_snow:
                    daily_snow[date_key] = 0.0
                daily_snow[date_key] += entry['value'] * 39.3701

        all_dates = sorted(list(set(daily_temps.keys()) | set(daily_snow.keys())))
        
        for date in all_dates:
            if date < datetime.now(timezone.utc).date():
                continue
                
            temp = daily_temps.get(date, 33.0)
            snow = daily_snow.get(date, 0.0)
            
            if snow > 0.1 and temp <= 33.0:
                logging.info(f"NWS Forecast Found: Snow on {date.strftime('%B %d')}")
                return date
                
        logging.info("No snow found in 7-day NWS forecast.")
        return None
        
    except Exception as e:
        logging.warning(f"Error fetching NWS forecast: {e}")
        return None

def kl_divergence(params, prior_mu, prior_sigma, likelihood_mu, likelihood_sigma):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    kl = 0.5 * (2 * np.log(prior_sigma) - 2 * log_sigma + \
                (sigma**2 + (mu - prior_mu)**2) / prior_sigma**2 - 1)
    return kl

def variational_inference(prior_mu, prior_sigma, likelihood_mu, likelihood_sigma=15.0):
    """Blends the prior and likelihood."""
    try:
        initial_params = [likelihood_mu, np.log(likelihood_sigma)]
        bounds = [(max(1, prior_mu - 60), min(365, prior_mu + 60)), (np.log(1.0), np.log(30.0))]
        
        res = minimize(kl_divergence, initial_params, 
                       args=(prior_mu, prior_sigma, likelihood_mu, likelihood_sigma),
                       bounds=bounds, method='L-BFGS-B')
        
        return min(365, max(1, res.x[0]))
    except:
        return min(365, max(1, (prior_mu + likelihood_mu) / 2))

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # --- STAGE 1: ONE-TIME SETUP ---
    try:
        location = input("Enter location (e.g., 'Lansing, MI' or lat,lon): ").strip()
        lat, lon = get_lat_lon(location)
        logging.info(f"Using coordinates: {lat:.4f}, {lon:.4f}")
        print("="*50)
        
        logging.info("--- STAGE 1: MODEL TRAINING (ONE-TIME) ---")
        
        X_train, y_train, historical_doys, current_oni = get_historical_data_for_ml(lat, lon)
        
        if historical_doys is None or len(historical_doys) == 0:
            logging.critical("Fatal error: Could not retrieve any historical data. Exiting.")
            exit()
        
        ml_model, scaler = train_snow_probability_model(X_train, y_train)
        
        oni_status = "Neutral"
        if current_oni >= 0.5: oni_status = f"El Niño ({current_oni:.2f})"
        elif current_oni <= -0.5: oni_status = f"La Niña ({current_oni:.2f})"
        logging.info(f"Current ENSO Status: {oni_status}")
        
        logging.info("ML model trained successfully. Entering continuous prediction loop.")
        
        prior_mu = np.median(historical_doys)
        prior_sigma = max(np.std(historical_doys) if len(historical_doys) > 1 else 15.0, 5.0)
        
    except Exception as e:
        logging.critical(f"Fatal error during setup: {e}", exc_info=True) # Added exc_info for more detail
        exit()

    # --- STAGE 2: CONTINUOUS PREDICTION LOOP ---
    logging.info("--- STAGE 2: STARTING HOURLY PREDICTION LOOP ---")
    
    while True:
        try:
            current_date = datetime.now()
            logging.info(f"--- Running new prediction cycle ---")
            
            # 1. Get Real 7-Day Forecast
            nws_snow_date = fetch_nws_forecast_with_snow(lat, lon)
            
            if nws_snow_date:
                days_until = (nws_snow_date - current_date.date()).days
                logging.info("\n" + "="*50 +
                             "\n=== FINAL PREDICTION (FROM NWS) ===" +
                             f"\nFirst Snowfall: {nws_snow_date.strftime('%A, %B %d, %Y')}" +
                             f"\nDays from now: {days_until}" +
                             "\nConfidence: HIGH (Based on NWS forecast)" +
                             "\n" + "="*50
                            )
                logging.info(f"Prediction complete. Sleeping for 1 hour...")
                time.sleep(3600)
                continue 

            # 2. No NWS forecast, run statistical model
            logging.info("No NWS snow. Moving to statistical ML model...")
            
            current_doy = current_date.timetuple().tm_yday
            current_year = current_date.year

            ml_onset_days = predict_first_snow_with_ml(
                ml_model, scaler, current_doy, current_oni, threshold=0.25
            )
            
            if ml_onset_days is None:
                logging.warning("ML model did not find a likely snow day in the next 180 days.")
                logging.info(f"Prediction complete. Sleeping for 1 hour...")
                time.sleep(3600)
                continue

            ml_predicted_doy = (current_doy + ml_onset_days)
            if ml_predicted_doy > 365:
                ml_predicted_doy -= 365
                
            logging.info("\n--- Statistical Model Predictions ---" +
                         f"\nHistorical Average (Prior):   DOY {prior_mu:.1f} ± {prior_sigma:.1f}" +
                         f"\nML Model (ENSO-Aware):  DOY {ml_predicted_doy:.1f}"
                        )
            
            # B) Bayesian Fusion
            final_fused_doy = variational_inference(prior_mu, prior_sigma, ml_predicted_doy)
            predicted_doy = int(final_fused_doy)

            # C) Calculate Final Date
            predicted_year = current_year
            if predicted_doy < current_doy:
                 predicted_year += 1
            
            predicted_date = datetime(predicted_year, 1, 1) + timedelta(days=predicted_doy - 1)
            days_until = (predicted_date - current_date).days
            
            logging.info("\n" + "="*50 +
                         "\n=== FINAL PREDICTION (STATISTICAL) ===" +
                         f"\nFirst Snowfall: {predicted_date.strftime('%B %d, %Y')}" +
                         f"\nDays from now: {days_until}" +
                         f"\nConfidence: MEDIUM (Based on {len(historical_doys)} years of climate data)" +
                         f"\nDriver: {oni_status}" +
                         "\n" + "="*50
                        )

            # 3. Sleep for 1 hour
            logging.info(f"Prediction complete. Sleeping for 1 hour...")
            time.sleep(3600) # 3600 seconds = 1 hour
        
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Exiting continuous loop.")
            break 
            
        except Exception as e:
            logging.error(f"Error in prediction loop: {e}", exc_info=True) # Added exc_info for more detail
            logging.info("Attempting to recover. Sleeping for 15 minutes...")
            time.sleep(900)

    logging.info("Script has shut down.")
