Flight Delay Prediction

This project predicts if a U.S. flight will be delayed using flight data and weather data.

Dataset comes from Kaggle (2019 to 2023 flights) and weather data comes from Open-Meteo.

Dataset Link: https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023

**Built by:** Sarah Menezes, Ved Patel, and Aditya Velagapudi.

**What this does**

The goal is to predict if a flight will be delayed before it even takes off.

A delay means the flight leaves more than 15 minutes late or gets cancelled.

**The model uses things like:**

time of flight
airline
distance
weather at the airport

**How to run**

Download the dataset from Kaggle
Put the CSV file in the project folder
Install dependencies:

pip install -r requirements.txt

Clean the data:

python3 clean_data.py

Get weather data:

python3 fetch_weather.py

Merge weather with flights:

python3 merge_weather.py

Run models:

python3 model_logistic_regression.py
python3 model_random_forest.py
python3 model_gradient_boosting.py

**Data files:**

flights_sample_3m.csv
raw flight data

cleaned_flights.csv
cleaned version of the data

weather_cache.csv
weather data for each airport

flights_with_weather.csv
final dataset used for models

airports.csv
maps airport codes to location

**Models used**

Logistic Regression
simple and fast baseline

Random Forest
handles more complex patterns

Gradient Boosting
best performance, builds models step by step

**Important note**

Only data that is known before the flight is used.

Things like actual delay time or arrival time are removed so the model does not cheat.

**Output**

Each model prints results in the terminal and saves charts as images.
