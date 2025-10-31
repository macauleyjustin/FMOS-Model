import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
import re
import logging
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pdb

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

        # Header line in ONI file: "YR  MON   TOTAL  ANOM"
        for i, line in enumerate(lines):
            if "YR" in line and "TOTAL" in line and "ANOM" in line:
                header_line = line.split()
                data_start_index = i + 1
                break

        if header_line is None:
            raise ValueError("Could not find header (YR, TOTAL, ANOM) in ONI data file.")

        yr_col = header_line.index("YR")
        seas_col = header_line.index("MON")  # months are coded numerically 1..12
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
                season = int(parts[seas_col])
                anomaly_str = parts[anom_col]

                if anomaly_str != '-99.9':
                    anomaly = float(anomaly_str)
                    if season == 10:  # SON -> Oct=10
                        historical_son_data[year] = anomaly

                    latest_oni_value = anomaly
                    latest_oni_year = year
                    latest_oni_period = f"Month {season}"
            except (ValueError, IndexError):
                continue

        if latest_oni_value is None:
            raise ValueError("Could not parse any valid ONI values.")

        logging.info(f"Successfully parsed ONI. Most recent: {latest_oni_period} {latest_oni_year} (ONI: {latest_oni_value:.2f})")
        return historical_son_data, latest_oni_value

    except Exception as e:
        logging.error(f"Failed to fetch or parse ONI data: {e}", exc_info=True)
        return {}, 0.0


# --- General Parser for AO, NAO, PNA ---
def _fetch_historical_teleconnection(url, name):
    """
    Generic parser for NOAA's monthly teleconnection files (AO, NAO, PNA).
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
                if year < 1900 or len(parts) < 13:  # Must have 12 months + year
                    continue

                for month_idx in range(1, 13):
                    value_str = parts[month_idx]
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue
                    if value > -99.9:
                        historical_data[year][month_idx] = value
                        latest_value = value
            except ValueError:
                continue  # Skip header lines or malformed data

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
    # --- FIX: Corrected PNA URL ---
    historical_pna_data, current_pna = _fetch_historical_teleconnection(
        "https://psl.noaa.gov/data/correlation/pna.data", "PNA")

    current_drivers = {
        "oni": current_oni, "ao": current_ao, "nao": current_nao, "pna": current_pna
    }

    # 2. Fetch daily Open-Meteo data
    logging.info(f"Fetching {years} years of daily climate data for ML model training...")
    current_year = datetime.now().year
    start_date = f"{current_year - years}-01-01"
    end_date = f"{current_year - 1}-12-31"

    base_url = "https://archive-api.open-meteo.com/v1/archive"

    # --- FIX: Removed soil_moisture_0_to_7cm (not available in archive API) ---
    # All units are default (Metric)
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': 'temperature_2m_min,snowfall_sum,snow_depth',
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
        temp_mins_c = daily_data.get('temperature_2m_min', [])  # In Celsius
        snow_sums_cm = daily_data.get('snowfall_sum', [])  # In CM
        snow_depth_m = daily_data.get('snow_depth', [])  # In meters

        # --- Fix: Ensure all lists have the same length ---
        min_len = min(len(times), len(temp_mins_c), len(snow_sums_cm), len(snow_depth_m))
        if len(times) != min_len:
            logging.warning(f"Data length mismatch in Open-Meteo response. Truncating to {min_len} days.")
            times = times[:min_len]
            temp_mins_c = temp_mins_c[:min_len]
            snow_sums_cm = snow_sums_cm[:min_len]
            snow_depth_m = snow_depth_m[:min_len]
        # --- End Fix ---

        X_train_first_snow, y_train_first_snow = [], []
        first_snow_doy_list = []
        seasonal_snow_totals = defaultdict(float)
        seasonal_driver_features = {}
        year_data = {}
        first_snow_years_found = set()

        for i, date_str in enumerate(times):
            # Check for missing data (None)
            if (temp_mins_c[i] is None or snow_sums_cm[i] is None or
                snow_depth_m[i] is None):
                continue

            # --- Convert units ---
            temp_f = temp_mins_c[i] * 9/5 + 32
            snow_in = snow_sums_cm[i] / 2.54  # cm to inches
            snow_depth_in = snow_depth_m[i] * 39.3701  # meters to inches

            date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
            doy = date.timetuple().tm_yday
            year = date.year
            month = date.month

            oni_val = historical_oni_data.get(year, 0.0)
            ao_val = historical_ao_data.get(year, {}).get(month, 0.0)
            nao_val = historical_nao_data.get(year, {}).get(month, 0.0)
            pna_val = historical_pna_data.get(year, {}).get(month, 0.0)

            if date.month >= 8 or date.month <= 5:  # "Snow Season"
                # --- FIX: Removed soil_moisture from features (index 2 becomes snow_depth) ---
                X_train_first_snow.append([
                    doy,           # 0: doy
                    oni_val,       # 1: oni
                    snow_depth_in, # 2: snow_depth (was soil)
                    ao_val,        # 3: ao
                    nao_val,       # 4: nao
                    pna_val        # 5: pna
                ])
                y_train_first_snow.append(1 if snow_in > 0.1 else 0)

            season_year = year if month >= 8 else year - 1
            if snow_in > 0:
                seasonal_snow_totals[season_year] += snow_in

            if month == 10:  # October
                if (season_year in historical_oni_data and
                    month in historical_ao_data.get(season_year, {}) and
                    month in historical_nao_data.get(season_year, {}) and
                    month in historical_pna_data.get(season_year, {})):
                    seasonal_driver_features[season_year] = [
                        historical_oni_data[season_year],
                        historical_ao_data[season_year][month],
                        historical_nao_data[season_year][month],
                        historical_pna_data[season_year][month]
                    ]

            if year not in year_data:
                year_data[year] = []
            year_data[year].append((date, temp_f, snow_in))

        # --- FIX: More robust first-snow-day finding logic ---
        for year in range(current_year - years, current_year):
            season_year = year
            if season_year in first_snow_years_found:
                continue

            # Check fall (Aug-Dec) of this year
            if season_year in year_data:
                for date, temp, snow in year_data[season_year]:
                    if date.month >= 8:
                        if temp <= 32.0 and snow > 0.1:
                            first_snow_doy_list.append(date.timetuple().tm_yday)
                            first_snow_years_found.add(season_year)
                            break

            # Check winter/spring (Jan-May) of *next* year
            if season_year + 1 in year_data and season_year not in first_snow_years_found:
                for date, temp, snow in year_data[season_year + 1]:
                    if date.month <= 5:
                        if temp <= 32.0 and snow > 0.1:
                            first_snow_doy_list.append(date.timetuple().tm_yday)
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
    logging.info("Fetching current ground state (Soil Moisture, Snow Depth)...")
    url = "https://api.open-meteo.com/v1/forecast"

    # --- FIX: Only use snow_depth (soil_moisture not available in forecast either) ---
    params = {
        'latitude': lat, 'longitude': lon,
        'daily': 'snow_depth',
        'forecast_days': 1, 'timezone': 'auto'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        daily_data = data.get('daily', {})
        snow_depth_m = daily_data.get('snow_depth', [0.0])[0]

        if snow_depth_m is None:
            snow_depth_m = 0.0

        # --- Convert units ---
        current_snow_depth_in = snow_depth_m * 39.3701  # meters to inches

        # --- FIX: Return snow_depth and a default soil moisture value ---
        logging.info(f"Current Ground State: Snow Depth={current_snow_depth_in:.2f} in")
        return 0.25, current_snow_depth_in  # Return default soil moisture + snow depth
    except Exception as e:
        logging.error(f"Failed to fetch current ground state: {e}", exc_info=True)
        return 0.25, 0.0


def train_first_snow_model(X_train, y_train):
    """
    Trains a Logistic Regression model for "First Snow"
    """
    # --- FIX: rename to avoid overshadowing the built-in 'model' ---
    clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear')

    # Features are:
    # 0: doy, 1: oni, 2: snow_depth, 3: ao, 4: nao, 5: pna
    doy_features = X_train[:, 0]

    doy_radians = (doy_features * (2 * np.pi / 365.25))
    X_cyclic_doy = np.column_stack((np.sin(doy_radians), np.cos(doy_radians)))

    # --- FIX: Updated scalers to match new features (removed soil) ---
    scalers = {}
    scaled_features_list = []

    feature_names = ['oni', 'snow_depth', 'ao', 'nao', 'pna']
    for i, name in enumerate(feature_names):
        feature_data = X_train[:, i + 1].reshape(-1, 1)
        scaler = StandardScaler()
        scaled_features_list.append(scaler.fit_transform(feature_data))
        scalers[name] = scaler

    # Concatenate cyclic DOY + 5 scaled drivers -> shape (n, 7)
    X_features_final = np.column_stack([X_cyclic_doy] + scaled_features_list)

    clf.fit(X_features_final, y_train)

    return clf, scalers


def train_seasonal_snow_model(X_train, y_train):
    """
    Trains a Linear Regression model for "Total Seasonal Snowfall".
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_first_snow_with_ml(first_snow_clf, scalers, current_doy, current_drivers, threshold=0.25):
    """
    Uses the trained ML model to find the first day after
    current_doy where snow probability exceeds the threshold.
    """
    # --- FIX: Updated to match new feature order (removed soil) ---
    (current_snow,
     current_ao, current_nao, current_pna) = current_drivers

    # current_drivers order: oni, snow, ao, nao, pna (soil removed)
    scaled_oni_val = scalers['oni'].transform([[current_drivers['oni']]])[0, 0]
    scaled_snow_val = scalers['snow_depth'].transform([[current_snow]])[0, 0]
    scaled_ao_val = scalers['ao'].transform([[current_ao]])[0, 0]
    scaled_nao_val = scalers['nao'].transform([[current_nao]])[0, 0]
    scaled_pna_val = scalers['pna'].transform([[current_pna]])[0, 0]

    scaled_drivers_list = [scaled_oni_val, scaled_snow_val,
                           scaled_ao_val, scaled_nao_val, scaled_pna_val]

    for day_offset in range(1, 180):
        doy = current_doy + day_offset
        if doy > 365:
            doy -= 365

        doy_rad = doy * (2 * np.pi / 365.25)
        X_cyclic_doy = [np.sin(doy_rad), np.cos(doy_rad)]

        X_test = np.array([[X_cyclic_doy[0], X_cyclic_doy[1]] + scaled_drivers_list])

        probability = first_snow_clf.predict_proba(X_test)[0, 1]

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

        # Get snowfall (in meters) and temp (in Celsius)
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
                        daily_forecasts[date_key] = {'temp_c': None, 'snow_m': 0.0}
                    # Only set temp if we don't have one yet
                    if daily_forecasts[date_key]['temp_c'] is None:
                        daily_forecasts[date_key]['temp_c'] = entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS temp data: {e}")

        for entry in snow_data:
            try:
                start_time = datetime.fromisoformat(entry['validTime'].split('/')[0].replace('Z', '+00:00'))
                date_key = start_time.date()
                if entry['value'] is not None:
                    if date_key not in daily_forecasts:
                        daily_forecasts[date_key] = {'temp_c': None, 'snow_m': 0.0}
                    daily_forecasts[date_key]['snow_m'] += entry['value']
            except Exception as e:
                logging.warning(f"Error parsing NWS snow data: {e}")

        today = datetime.now(timezone.utc).date()
        for date_key in sorted(daily_forecasts.keys()):
            if date_key < today:
                continue

            forecast = daily_forecasts[date_key]
            temp_c = forecast['temp_c']
            if temp_c is None:
                continue  # Skip days without temp
            temp_f = temp_c * 9/5 + 32
            snow_in = forecast['snow_m'] * 39.3701

            if snow_in > 0.1 and temp_f <= 33.0:  # 33F as a buffer
                logging.info(f"NWS Forecast Found: Snow on {date_key.strftime('%B %d')}")
                return date_key

        logging.info("No snow found in 7-day NWS forecast.")
        return None

    except Exception as e:
        logging.warning(f"Error fetching NWS forecast: {e}", exc_info=True)
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


# --- 4. MAIN EXECUTION (Massively Updated) ---
if __name__ == "__main__":

    # --- ADDED DEBUGGER ---
    # pdb.set_trace()
    # Uncomment the line above to step through the code

    # --- STAGE 1: ONE-TIME SETUP ---
    try:
        location = input("Enter location (e.g., 'Lansing, MI' or lat,lon): ").strip()
        lat, lon = get_lat_lon(location)
        logging.info(f"Using coordinates: {lat:.4f}, {lon:.4f}")
        print("=" * 50)

        logging.info("--- STAGE 1: MODEL TRAINING (ONE-TIME) ---")

        model_1_data, model_2_data, historical_doys, current_drivers_dict = \
            get_historical_data_for_ml(lat, lon)

        if historical_doys is None or len(historical_doys) == 0:
            logging.critical("Fatal error: Could not retrieve any historical data. Exiting.")
            exit()

        # --- Train Model 1 (First Snow) ---
        (X_train_1, y_train_1) = model_1_data
        first_snow_clf, scalers_1 = train_first_snow_model(X_train_1, y_train_1)
        logging.info("Model 1 (First Snow) trained successfully.")

        # --- Train Model 2 (Seasonal Total) ---
        (X_train_2, y_train_2) = model_2_data
        if len(X_train_2) > 0 and len(y_train_2) > 0:
            seasonal_snow_reg = train_seasonal_snow_model(X_train_2, y_train_2)
            logging.info("Model 2 (Seasonal Total) trained successfully.")
        else:
            seasonal_snow_reg = None
            logging.warning("Not enough data to train Model 2 (Seasonal Total). Skipping.")

        # --- Get current driver status ---
        current_oni = current_drivers_dict.get("oni", 0.0)
        current_ao = current_drivers_dict.get("ao", 0.0)
        current_nao = current_drivers_dict.get("nao", 0.0)
        current_pna = current_drivers_dict.get("pna", 0.0)

        oni_status = "Neutral"
        if current_oni >= 0.5:
            oni_status = f"El Niño ({current_oni:.2f})"
        elif current_oni <= -0.5:
            oni_status = f"La Niña ({current_oni:.2f})"

        driver_status_string = (
            f"ENSO: {oni_status} | AO: {current_ao:.2f} | "
            f"NAO: {current_nao:.2f} | PNA: {current_pna:.2f}"
        )
        logging.info(f"Current Climate Drivers: {driver_status_string}")

        logging.info("ML models trained. Entering continuous prediction loop.")

        prior_mu = np.median(historical_doys)
        prior_sigma = max(np.std(historical_doys) if len(historical_doys) > 1 else 15.0, 5.0)

    except Exception as e:
        logging.critical(f"Fatal error during setup: {e}", exc_info=True)
        exit()

    # --- STAGE 2: CONTINUOUS PREDICTION LOOP ---
    logging.info("--- STAGE 2: STARTING HOURLY PREDICTION LOOP ---")

    while True:
        try:
            current_date = datetime.now()
            logging.info(f"--- Running new prediction cycle ---")

            # 1. Get Real 7-Day Forecast (Highest Priority for First Snow)
            nws_snow_date = fetch_nws_forecast_with_snow(lat, lon)

            if nws_snow_date:
                days_until = (nws_snow_date - current_date.date()).days
                log_message = (
                    "\n" + "="*50 +
                    "\n=== FORECAST 1: FIRST SNOWFALL (NWS) ===" +
                    f"\nFirst Snowfall: {nws_snow_date.strftime('%A, %B %d, %Y')}" +
                    f"\nDays from now: {days_until}" +
                    "\nConfidence: HIGH (Based on NWS forecast)" +
                    "\n" + "="*50
                )
                logging.info(log_message)

            else:
                # 2. No NWS forecast, run statistical "First Snow" model
                logging.info("No NWS snow. Moving to statistical Model 1...")

                current_soil, current_snow = fetch_current_ground_state(lat, lon)

                # --- FIX: Updated current_drivers dict to match new feature order ---
                current_drivers_dict = {
                    "oni": current_oni,
                    "snow_depth": current_snow,  # was soil
                    "ao": current_ao,
                    "nao": current_nao,
                    "pna": current_pna
                }

                current_doy = current_date.timetuple().tm_yday
                current_year = current_date.year

                ml_onset_days = predict_first_snow_with_ml(
                    first_snow_clf, scalers_1, current_doy,
                    {"oni": current_oni, "snow_depth": current_snow,
                     "ao": current_ao, "nao": current_nao, "pna": current_pna},
                    threshold=0.25
                )

                if ml_onset_days is None:
                    logging.warning("Model 1 did not find a likely snow day in the next 180 days.")

                else:
                    ml_predicted_doy = (current_doy + ml_onset_days)
                    if ml_predicted_doy > 365:
                        ml_predicted_doy -= 365

                    logging.info("\n--- Statistical Model 1 Predictions ---" +
                                 f"\nHistorical Average (Prior):   DOY {prior_mu:.1f} ± {prior_sigma:.1f}" +
                                 f"\nML Model 1 (Full-Driver): DOY {ml_predicted_doy:.1f}"
                                )

                    final_fused_doy = variational_inference(prior_mu, prior_sigma, ml_predicted_doy)
                    predicted_doy = int(final_fused_doy)

                    predicted_year = current_year
                    if predicted_doy < current_doy:
                        predicted_year += 1

                    predicted_date = datetime(predicted_year, 1, 1) + timedelta(days=predicted_doy - 1)
                    days_until = (predicted_date - current_date).days

                    ground_driver_string = (
                        f"Snow: {current_snow:.1f} in"
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
            if seasonal_snow_reg is not None:
                logging.info("Running Model 2 (Seasonal Snowfall Outlook)...")

                # Features: oni, ao, nao, pna
                seasonal_driver_features = np.array([[
                    current_oni, current_ao, current_nao, current_pna
                ]])

                predicted_total_snow = seasonal_snow_reg.predict(seasonal_driver_features)[0]

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

