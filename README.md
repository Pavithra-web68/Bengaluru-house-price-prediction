# Bengaluru House Price Prediction 🏠

## Overview

A Machine Learning project that predicts approximate residential
property prices in Bengaluru based on housing features.

## Features

- Data cleaning and preprocessing
- Location-based house price prediction
- Random Forest regression model
- Model evaluation using MAE and R²
- Interactive web application built with Streamlit

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

## Dataset

Bengaluru House Price Data:
https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Check the dataset licence and terms before redistributing the CSV.

## Installation

1. Clone or download this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Place the dataset at:

   data/Bengaluru_House_Data.csv

## Train the Model

Run:

python train_house_price_model.py

This creates house_price_model.joblib.

## Launch the Application

Run:

streamlit run app.py

Open the local URL shown in your terminal.

## Evaluation

The model reports Mean Absolute Error (MAE) and R² after training.
Actual performance will depend on data quality and the test split.

## Disclaimer

Predictions are estimates for educational purposes and are not
professional property valuations.