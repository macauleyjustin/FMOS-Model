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


# --- 2. TELECONNECTION & DRIVER DATA FETCHERS ---

def _fetch_historical_oni():
    """
    Fetches the Oceanic Niño Index (ONI) from NOAA.
    Returns:
    - {year: oni_value_for_SON}
    - latest_oni_value
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
                season = parts[seas_col]
                year = int(parts[yr_col])
                anomaly_str = parts[anom_col]
                
                if anomaly_str != '-99.9':
                    anomaly = float(anomaly_str)
                    
                    if season == "SON":
                        historical_son_data[year] = anomaly
                    
                    latest_oni_value = anomaly
                    latest_oni_year = year
                    latest_oni_period = season
                        
            except (ValueError, IndexError):
                continue

        if latest_oni_value is None:
            raise ValueError("Could not parse any valid ONI values.")

        logging.info(f"Successfully parsed ONI. Most recent: {latest_oni_period} {latest_oni_year} (ONI: {latest_oni_value:.2f})")
        return historical_son_data, latest_oni_value

    except Exception as e:
        logging.error(f"Failed to fetch or parse ONI data: {e}")
        return {}, 0.0

# --- General Parser for AO, NAO, PNA, PDO ---
def _fetch_historical_teleconnection(url, name):
    """
    Generic parser for NOAA's monthly teleconnection files (AO, NAO, PNA, PDO).
    Returns:
    - {year: {month: value}}
    - latest_value
    """
    logging.info(f"Fetching historical {name} data from NOAA...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        historical_data = defaultdict(dict)
        latest_value = 0.0
        
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            try:
                year = int(parts[0])
                if year < 1900 or len(parts) != 13: # Basic sanity check
                    continue
                    
                for month_idx, value_str in enumerate(parts[1:], 1):
                    value = float(value_str)
                    if value > -99.9: 
                        historical_data[year][month_idx] = value
                        latest_value = value
            except ValueError:
                continue # Skip header lines or malformed data

        if not historical_data:
             raise ValueError(f"No data parsed for {name}. Check file format.")

        logging.info(f"Successfully parsed {name}. Most recent value: {latest_value:.2f}")
        return historical_data, latest_value
    
    except Exception as e:
        logging.error(f"Failed to fetch or parse {name} data: {e}", exc_info=True)
        return defaultdict(dict), 0.0


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

# --- MASSIVELY UPDATED FUNCTION (THE FIX) ---
def get_historical_data_for_ml(lat, lon, years=30):
    """
    Fetches 30 years of daily data PLUS historical drivers (ENSO, AO, NAO, PNA, etc.)
    This now makes ONE call to Open-Meteo for all weather data in metric.
    """
    logging.info(f"--- STAGE 1A: Fetching all historical climate driver data ---")
    
    # 1. Fetch all teleconnection data
    historical_oni_data, current_oni = _fetch_historical_oni()
    historical_ao_data, current_ao = _fetch_historical_teleconnection(
        "https://psl.noaa.gov/data/correlation/ao.data", "AO")
    historical_nao_data, current_nao = _fetch_historical_teleconnection(
        "https://psl.noaa.gov/data/correlation/nao.data", "NAO")
    historical_pna_data, current_pna = _fetch_historical_teleconnection(
        "https://psl.noaa.gov/data/correlation/pna.data", "PNA")
    historical_pdo_data, current_pdo = _fetch_historical_teleconnection(
        "https://psl.noaa.gov/data/correlation/pdo.data", "PDO")
    
    current_drivers = {
        "oni": current_oni, "ao": current_ao, "nao": current_nao, "pna": current_pna, "pdo": current_pdo
    }

    # 2. Fetch daily Open-Meteo data
    logging.info(f"Fetching {years} years of daily climate data for ML model training...")
    current_year = datetime.now().year
    start_date = f"{current_year - years}-01-01"
    end_date = f"{current_year - 1}-12-31"
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': 'temperature_2m_min,snowfall_sum',
        'timezone': 'auto'
    }
    
    try:
        logging.info("Fetching all historical weather data from Open-Meteo...")
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        daily_data = data.get('daily', {})
        if not daily_data:
            raise ValueError("No daily data found in Open-Meteo response")

        # --- Merge the data ---
        times = daily_data.get('time', [])
        temp_mins_c = daily_data.get('temperature_2m_min', []) # In Celsius
        snow_sums_cm = daily_data.get('snowfall_sum', []) # In CM 
        
        # --- Fix: Ensure all lists have the same length ---
        min_len = min(len(times), len(temp_mins_c), len(snow_sums_cm))
        if len(times) != min_len:
             logging.warning(f"Data length mismatch in Open-Meteo response. Truncating to {min_len} days.")
             times = times[:min_len]
             temp_mins_c = temp_mins_c[:min_len]
             snow_sums_cm = snow_sums_cm[:min_len]

        X_train_first_snow, y_train_first_snow = [], []
        first_snow_doy_list = []
        seasonal_snow_totals = defaultdict(float)
        seasonal_driver_features = {} 
        year_data = {}
        first_snow_years_found = set() # --- FIX: More efficient check ---
        
        for i, date_str in enumerate(times):
            # Check for missing data (None)
            if (temp_mins_c[i] is None or snow_sums_cm[i] is None):
                continue
            
            # --- Convert units ---
            temp_f = temp_mins_c[i] * 9/5 + 32
            snow_in = snow_sums_cm[i] / 2.54 # cm to inches (FIXED)
            
            date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            doy = date.timetuple().tm_yday
            year = date.year
            month = date.month
            
            oni_val = historical_oni_data.get(year, 0.0)
            ao_val = historical_ao_data.get(year, {}).get(month, 0.0)
            nao_val = historical_nao_data.get(year, {}).get(month, 0.0)
            pna_val = historical_pna_data.get(year, {}).get(month, 0.0)
            pdo_val = historical_pdo_data.get(year, {}).get(month, 0.0)

            if date.month >= 8 or date.month <= 5: # "Snow Season"
                X_train_first_snow.append([
                    doy, 
                    oni_val, 
                    ao_val,
                    nao_val,
                    pna_val,
                    pdo_val
                ])
                y_train_first_snow.append(1 if snow_in > 0.1 else 0)

            season_year = year if month >= 8 else year - 1
            if snow_in > 0:
                seasonal_snow_totals[season_year] += snow_in 
            
            if month == 10: # October
                if (season_year in historical_oni_data and 
                    month in historical_ao_data.get(season_year, {}) and
                    month in historical_nao_data.get(season_year, {}) and
                    month in historical_pna_data.get(season_year, {}) and
                    month in historical_pdo_data.get(season_year, {})):
                    
                    seasonal_driver_features[season_year] = [
                        historical_oni_data[season_year],
                        historical_ao_data[season_year][month],
                        historical_nao_data[season_year][month],
                        historical_pna_data[season_year][month],
                        historical_pdo_data[season_year][month]
                    ]

            if year not in year_data: year_data[year] = []
            year_data[year].append((date, temp_f, snow_in))

        # --- FIX: More robust first-snow-day finding logic ---
        for year in range(current_year - years, current_year):
            season_year = year
            if season_year in first_snow_years_found:
                continue

            # Check fall (Aug-Dec) of this year
            if season_year in year_data:
                for date, temp, snow in year_data[season_year]:
                    if date.month >= 8: # Start looking in Aug
                        if temp <= 32.0 and snow > 0.1:
                            first_snow_doy_list.append(date.timetuple().tm_yday)
                            first_snow_years_found.add(season_year)
                            break
            
            # Check winter/spring (Jan-May) of *next* year
            if season_year + 1 in year_data and season_year not in first_snow_years_found:
                 for date, temp, snow in year_data[season_year+1]:
                     if date.month <= 5: # Stop in May
                         if temp <= 32.0 and snow > 0.1:
                             # This is still part of the 'season_year' season
                             first_snow_doy_list.append(date.timetuple().tm_yday + 365)
                             first_snow_years_found.add(season_year)
                             break
        # --- End Fix ---
        
        X_train_seasonal, y_train_seasonal = [], []
        for year, drivers in seasonal_driver_features.items():
            if year in seasonal_snow_totals:
                X_train_seasonal.append(drivers)
                y_train_seasonal.append(seasonal_snow_totals[year])

        if not X_train_first_snow:
            raise ValueError("No historical data found to train model.")
            
        logging.info(f"Found {len(first_snow_doy_list)} historical first-snow events.")
        logging.info(f"Training 'First Snow' model on {len(X_train_first_snow)} daily records.")
        logging.info(f"Training 'Seasonal Total' model on {len(X_train_seasonal)} seasons.")
        
        model_1_data = (np.array(X_train_first_snow), np.array(y_train_first_snow))
        model_2_data = (np.array(X_train_seasonal), np.array(y_train_seasonal))
        
        return model_1_data, model_2_data, np.array(first_snow_doy_list), current_drivers
        
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}", exc_info=True)
        return None, None, None, {}

def fetch_current_ground_state(lat, lon):
    """Fetches the most recent soil moisture and snow depth from Open-Meteo."""
    # --- FIX: Removed as soil_moisture and snow_depth are not available in daily params; return defaults ---
    logging.info("Skipping fetch of current ground state (not available in daily API). Using defaults.")
    return 0.0, 0.0

def train_first_snow_model(X_train, y_train):
    """
    Trains a Random Forest model for "First Snow"
    """
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    # --- Features are:
    # 0: doy, 1: oni, 2: ao, 3: nao, 4: pna, 5: pdo
    doy_features = X_train[:, 0]
    
    doy_radians = (doy_features * (2 * np.pi / 365.25))
    X_cyclic_doy = np.column_stack((np.sin(doy_radians), np.cos(doy_radians)))
    
    # --- Use a dictionary for scalers (cleaner) ---
    scalers = {}
    scaled_features_list = []
    
    # Scale all features *except* the first one (doy)
    feature_names = ['oni', 'ao', 'nao', 'pna', 'pdo']
    for i, name in enumerate(feature_names):
        feature_data = X_train[:, i+1].reshape(-1, 1)
        scaler = StandardScaler()
        scaled_features_list.append(scaler.fit_transform(feature_data))
        scalers[name] = scaler
    
    X_features_final = np.column_stack([X_cyclic_doy] + scaled_features_list)
    
    model.fit(X_features_final, y_train)
    
    scores = cross_val_score(model, X_features_final, y_train, cv=5, scoring='roc_auc')
    logging.info(f"First Snow Model CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    
    return model, scalers

def train_seasonal_snow_model(X_train, y_train):
    """
    Trains a Random Forest model for "Total Seasonal Snowfall".
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    logging.info(f"Seasonal Snow Model CV MAE: {-scores.mean():.3f} ± {scores.std():.3f}")
    
    return model

def predict_first_snow_with_ml(model, scalers, current_doy, current_drivers, threshold=0.25):
    """
    Uses the trained ML model to find the first day after
    current_doy where snow probability exceeds the threshold.
    """
    (current_oni, current_ao, current_nao, current_pna, current_pdo) = current_drivers
    
    # --- Scale current drivers using the scaler dictionary ---
    scaled_oni_val = scalers['oni'].transform([[current_oni]])[0, 0]
    scaled_ao_val = scalers['ao'].transform([[current_ao]])[0, 0]
    scaled_nao_val = scalers['nao'].transform([[current_nao]])[0, 0]
    scaled_pna_val = scalers['pna'].transform([[current_pna]])[0, 0]
    scaled_pdo_val = scalers['pdo'].transform([[current_pdo]])[0, 0]
    
    scaled_drivers_list = [scaled_oni_val, scaled_ao_val, scaled_nao_val, scaled_pna_val, scaled_pdo_val]

    for day_offset in range(1, 180):
        doy = current_doy + day_offset
        if doy > 365:
            doy -= 365
        
        doy_rad = doy * (2 * np.pi / 365.25)
        X_cyclic_doy = [np.sin(doy_rad), np.cos(doy_rad)]
        
        X_test = np.array([X_cyclic_doy + scaled_drivers_list])
        
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
        
        # --- Use the 'forecastGridData' endpoint for quantitative snow data ---
        grid_url = response.json()['properties']['forecastGridData']
        grid_response = requests.get(grid_url, headers=headers, timeout=10)
        grid_response.raise_for_status()
        props = grid_response.json()['properties']
        
        # Get snowfall (in mm) and temp (in Celsius)
        snow_data = props.get('snowfallAmount', {}).get('values', [])
        temp_data = props.get('minTemperature', {}).get('values', [])
        
        if not snow_data or not temp_data:
            logging.warning("NWS forecast data missing 'snowfallAmount' or 'minTemperature'.")
            return None

        # Process data into daily values
        daily_forecasts = {}

        for entry in temp_data:
            try:
                start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
                date_key = start_time.date()
                if entry['value'] is not None:
                    if date_key not in daily_forecasts:
                        daily_forecasts[date_key] = {'temp_c': -99, 'snow_mm': 0.0}
                    daily_forecasts[date_key]['temp_c'] = entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS temp data: {e}")

        for entry in snow_data:
            try:
                start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
                date_key = start_time.date()
                if entry['value'] is not None:
                    if date_key not in daily_forecasts:
                        daily_forecasts[date_key] = {'temp_c': -99, 'snow_mm': 0.0}
                    daily_forecasts[date_key]['snow_mm'] += entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS snow data: {e}")
        
        today = datetime.now(timezone.utc).date()
        for date_key in sorted(daily_forecasts.keys()):
            if date_key < today:
                continue

            forecast = daily_forecasts[date_key]
            temp_f = forecast['temp_c'] * 9/5 + 32
            # --- FIX: Corrected unit from meters to mm; convert mm to inches ---
            snow_in = forecast['snow_mm'] / 25.4
            
            if snow_in > 0.1 and temp_f <= 33.0: # 33F as a buffer
                logging.info(f"NWS Forecast Found: Snow on {date_key.strftime('%B %d')}")
                return date_key
                
        logging.info("No snow found in 7-day NWS forecast.")
        return None
        
    except Exception as e:
        logging.warning(f"Error fetching NWS forecast: {e}")
        return None

def bayesian_update(prior_mu, prior_sigma, likelihood_mu, likelihood_sigma=15.0):
    """Exact Bayesian update for two normals."""
    try:
        prior_var = prior_sigma ** 2
        like_var = likelihood_sigma ** 2
        post_var = 1 / (1 / prior_var + 1 / like_var)
        post_mu = (prior_mu / prior_var + likelihood_mu / like_var) * post_var
        return post_mu
    except:
        return (prior_mu + likelihood_mu) / 2

# --- 4. MAIN EXECUTION (Massively Updated) ---
if __name__ == "__main__":
    
    # --- STAGE 1: ONE-TIME SETUP ---
    try:
        location = input("Enter location (e.g., 'Lansing, MI' or lat,lon): ").strip()
        lat, lon = get_lat_lon(location)
        logging.info(f"Using coordinates: {lat:.4f}, {lon:.4f}")
        print("="*50)
        
        logging.info("--- STAGE 1: MODEL TRAINING (ONE-TIME) ---")
        
        model_1_data, model_2_data, historical_doys, _ = \
            get_historical_data_for_ml(lat, lon)
        
        if historical_doys is None or len(historical_doys) == 0:
            logging.critical("Fatal error: Could not retrieve any historical data. Exiting.")
            exit()
        
        # --- Train Model 1 (First Snow) ---
        (X_train_1, y_train_1) = model_1_data
        model_1, scalers_1 = train_first_snow_model(X_train_1, y_train_1)
        logging.info("Model 1 (First Snow) trained successfully.")
        
        # --- Train Model 2 (Seasonal Total) ---
        (X_train_2, y_train_2) = model_2_data
        if len(X_train_2) > 0 and len(y_train_2) > 0:
            model_2 = train_seasonal_snow_model(X_train_2, y_train_2)
            logging.info("Model 2 (Seasonal Total) trained successfully.")
        else:
            model_2 = None
            logging.warning("Not enough data to train Model 2 (Seasonal Total). Skipping.")
        
        # --- Normalize historical DOYs for cyclic nature ---
        cutoff = 200  # Assume no first snow before ~July
        adjusted_doys = [d if d < 365 else d - 365 for d in historical_doys]  # Normalize to 1-365
        prior_mu = np.median(adjusted_doys)
        prior_sigma = max(np.std(adjusted_doys) if len(adjusted_doys) > 1 else 15.0, 5.0)
        
        logging.info("ML models trained. Entering continuous prediction loop.")
        
    except Exception as e:
        logging.critical(f"Fatal error during setup: {e}", exc_info=True)
        exit()

    # --- STAGE 2: CONTINUOUS PREDICTION LOOP ---
    logging.info("--- STAGE 2: STARTING HOURLY PREDICTION LOOP ---")
    
    while True:
        try:
            current_date = datetime.now()
            logging.info(f"--- Running new prediction cycle ---")
            
            # --- FIX: Refetch current drivers each cycle to update ---
            _, current_oni = _fetch_historical_oni()
            _, current_ao = _fetch_historical_teleconnection(
                "https://psl.noaa.gov/data/correlation/ao.data", "AO")
            _, current_nao = _fetch_historical_teleconnection(
                "https://psl.noaa.gov/data/correlation/nao.data", "NAO")
            _, current_pna = _fetch_historical_teleconnection(
                "https://psl.noaa.gov/data/correlation/pna.data", "PNA")
            _, current_pdo = _fetch_historical_teleconnection(
                "https://psl.noaa.gov/data/correlation/pdo.data", "PDO")
            
            oni_status = "Neutral"
            if current_oni >= 0.5: oni_status = f"El Niño ({current_oni:.2f})"
            elif current_oni <= -0.5: oni_status = f"La Niña ({current_oni:.2f})"
            
            driver_status_string = (
                f"ENSO: {oni_status} | AO: {current_ao:.2f} | "
                f"NAO: {current_nao:.2f} | PNA: {current_pna:.2f} | PDO: {current_pdo:.2f}"
            )
            logging.info(f"Current Climate Drivers: {driver_status_string}")
            
            # 1. Get Real 7-Day Forecast (Highest Priority for First Snow)
            nws_snow_date = fetch_nws_forecast_with_snow(lat, lon)
            
            if nws_snow_date:
                days_until = (nws_snow_date - current_date.date()).days
                log_message = ("\n" + "="*50 +
                             "\n=== FORECAST 1: FIRST SNOWFALL (NWS) ===" +
                             f"\nFirst Snowfall: {nws_snow_date.strftime('%A, %B %d, %Y')}" +
                             f"\nDays from now: {days_until}" +
                             "\nConfidence: HIGH (Based on NWS forecast)" +
                             "\n" + "="*50)
                logging.info(log_message)
            
            else:
                # 2. No NWS forecast, run statistical "First Snow" model
                logging.info("No NWS snow. Moving to statistical Model 1...")
                
                # --- FIX: Removed soil and snow_depth ---
                current_soil, current_snow = 0.0, 0.0
                
                # Order MUST match training: oni, ao, nao, pna, pdo
                current_drivers_list = (
                    current_oni, current_ao, current_nao, current_pna, current_pdo
                )
                
                current_doy = current_date.timetuple().tm_yday
                current_year = current_date.year

                ml_unwrapped = predict_first_snow_with_ml(
                    model_1, scalers_1, current_doy, current_drivers_list, threshold=0.25
                )
                
                if ml_unwrapped is None:
                    logging.warning("Model 1 did not find a likely snow day in the next 180 days.")
                
                else:
                    # --- FIX: Keep unwrapped for fusion, adjust prior to future ---
                    prior_mu_adjusted = prior_mu if prior_mu > current_doy else prior_mu + 365
                    
                    logging.info("\n--- Statistical Model 1 Predictions ---" +
                                 f"\nHistorical Average (Prior):   DOY {prior_mu:.1f} ± {prior_sigma:.1f}" +
                                 f"\nML Model 1 (Full-Driver): DOY {ml_unwrapped:.1f}"
                                )
                    
                    final_fused_unwrapped = bayesian_update(prior_mu_adjusted, prior_sigma, ml_unwrapped)
                    
                    # --- Convert fused to date ---
                    fused = final_fused_unwrapped
                    predicted_year = current_year
                    while fused > 365:
                        fused -= 365
                        predicted_year += 1
                    predicted_doy = int(fused)
                    predicted_date = datetime(predicted_year, 1, 1) + timedelta(days=predicted_doy - 1)
                    days_until = (predicted_date - current_date).days
                    
                    ground_driver_string = (
                        f"Soil: {current_soil:.2f} m³/m³ | Snow: {current_snow:.1f} in"
                    )
                    
                    logging.info("\n" + "="*50 +
                                 "\n=== FORECAST 1: FIRST SNOWFALL (STATISTICAL) ===" +
                                 f"\nFirst Snowfall: {predicted_date.strftime('%B %d, %Y')}" +
                                 f"\nDays from now: {days_until}" +
                                 f"\nConfidence: MEDIUM (Based on {len(historical_doys)} years)" +
                                 f"\nClimate Drivers: {driver_status_string}" +
                                 f"\nGround Drivers:  {ground_driver_string}" +
                                 "\n" + "="*50
                                )
            
            # 3. Run Model 2 (Seasonal Total)
            if model_2 is not None:
                logging.info("Running Model 2 (Seasonal Snowfall Outlook)...")
                
                # Features: oni, ao, nao, pna, pdo
                seasonal_driver_features = np.array([[
                    current_oni, current_ao, current_nao, current_pna, current_pdo
                ]])
                
                predicted_total_snow = model_2.predict(seasonal_driver_features)[0]
                
                logging.info("\n" + "="*50 +
                             "\n=== FORECAST 2: SEASONAL SNOWFALL OUTLOOK ===" +
                             f"\nPredicted Total: {predicted_total_snow:.1f} inches" +
                             f"\nConfidence: MEDIUM (Based on {len(y_train_2)} seasons)" +
                             f"\nClimate Drivers: {driver_status_string}" +
                             "\n" + "="*50
                            )
            else:
                logging.warning("Skipping Model 2 (Seasonal Total) as it was not trained.")

            # 4. Sleep for 1 hour
            logging.info(f"Prediction cycle complete. Sleeping for 1 hour...")
            time.sleep(3600)
        
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Exiting continuous loop.")
            break 
            
        except Exception as e:
            logging.error(f"Error in prediction loop: {e}", exc_info=True)
            logging.info("Attempting to recover. Sleeping for 15 minutes...")
            time.sleep(900)

    logging.info("Script has shut down.")
