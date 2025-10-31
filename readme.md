MOS - First-Snow ML Operational Service

This project is a statistical weather model that provides a continuous, hourly forecast for the first measurable snowfall of the season at a given US location.

It is a hybrid forecaster that combines:

High-Priority Real-Time Data: It checks the official 7-day NWS forecast. If snow is predicted, it reports that immediately.

Long-Range Statistical Model: If no snow is in the 7-day forecast, it falls back to a custom machine learning model. This model is trained on 30 years of local climate data and is enhanced with the current ENSO (El Niño/La Niña) cycle to provide an accurate, long-range statistical prediction.

This project runs as a two-part service: a backend model and a frontend web dashboard.

How It Works

model.py (The Model Backend):

This is the main "engine." It runs as a continuous, hourly service in a terminal.

On first launch, it fetches 30 years of climate history (snowfall, temperature) and 70+ years of ENSO (El Niño/La Niña) data.

It trains a scikit-learn logistic regression model to learn the probability of snow based on the day of the year and the current ENSO status.

Every hour, it wakes up, fetches the latest 7-day NWS forecast, and generates a new prediction every hour. 
