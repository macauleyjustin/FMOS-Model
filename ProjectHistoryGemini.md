Project History: From "Garbage" to GUI

This document outlines the development process of the FMOS Snow Predictor, tracing its evolution from a non-functional script to a multi-featured, operational weather service.

Phase 1: The Spark (The "Garbage" Code)

Our project began when you provided an initial Python script. This script was ambitious, using a NeuralODE (a complex deep learning model) with the goal of predicting the first snowfall.

However, in our critique, we identified a fundamental meteorological flaw:

The model only used temperature as a variable.

It completely ignored precipitation (moisture), which is a required ingredient for snow.

It was, in effect, a "First Freeze" predictor that was mislabeled as a "First Snow" predictor.

Its method of "forecasting" 90 days out was a fantasy, as it was just extrapolating from 14 days of data with no understanding of atmospheric physics.

We (somewhat jokingly) agreed this first version was "a batch of rubbish."

Phase 2: The Pivot (The Logical Rebuild)

You challenged me to build "Option B" — a true, ML-powered model that actually worked.

This meant throwing away the entire NeuralODE concept. A "fantasy" physics simulation was the wrong tool. The right tool was a statistical model that could learn from real climate history.

Phase 3: Building the Real Model

We started from scratch with a new, logical foundation:

New Model: We chose a LogisticRegression model from scikit-learn.

New Goal: Instead of simulating the weather, our model would learn the statistical probability of snow for any given day of the year.

New Data: We built a data pipeline to fetch 30 years of real climate data (both temperature and snowfall) from the Open-Meteo API.

The "Unique" Part: We kept one clever piece of the original code: the Bayesian variational_inference. We used it to intelligently blend the simple historical average (our "prior") with our new, smarter ML model's prediction (our "likelihood").

This gave us a working, logical forecaster.

Phase 4: Debugging and Making it Smarter

Our initial build had several bugs that we fixed together:

The NameError: A simple historical_Doy vs. historical_doys typo.

The Geocoder Bug: We discovered that US zip codes (like 73301 for Austin) were being geocoded to Estonia. This was because our weather API (NWS) is US-only, but our geocoder (Nominatim) is global. We fixed this by forcing the search to add ", USA" for any 5-digit number.

The "Smart" Upgrade (ENSO): I suggested the model was "blind" to what kind of year it was. You agreed we should add a long-range climate driver. We implemented a new data pipeline to fetch 70+ years of ENSO (El Niño/La Niña) data from NOAA. We then added this as a new feature to our ML model, allowing it to make different predictions for an El Niño year versus a La Niña year.

Phase 5: The Real Debugging

This upgrade was difficult. The NOAA data was in a legacy text format that was fragile and undocumented.

The Parser Fail: My parser failed repeatedly, first looking for a "YEAR" header that didn't exist.

The Debug Step: I added a debug logger to print the first 10 lines of the file. This showed us the real header: SEAS YR TOTAL ANOM.

The New Parser: With this info, I rewrote the _fetch_historical_oni function from scratch to correctly parse the file.

The strptime Fail: The model then crashed on a ValueError, which we found was due to invisible whitespace in the climate data's date strings. We fixed this with a simple .strip().

Phase 6: Becoming a "Full Fledge" Service

With the core model stable, you asked to make it a "full fledge" continuous model.

The Service: We converted the one-time script into an hourly service that runs in an infinite loop and logs its predictions.
