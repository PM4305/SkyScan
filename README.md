# SkyScan: Flight Price Prediction

## Overview
SkyScan is a machine learning project that predicts flight prices based on various factors such as airline, flight class, duration, source, destination, number of stops, and time to departure. The model helps users make informed travel decisions by providing accurate price predictions.

## Key Features
- Data preprocessing and cleaning of flight data
- Comprehensive Exploratory Data Analysis (EDA)
- Multiple machine learning models comparison
- Feature engineering and selection
- Pipeline implementation for scalable predictions
- Model evaluation and persistence

## Project Structure
```
skyscan/
│
├── data/
│   ├── economy.csv
│   └── business.csv
│
├── notebooks/
│   └── flight_price_prediction.ipynb
│
├── models/
│   └── best_flight_price_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
│
├── docs/
│   └── Project_Report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skyscan.git
cd skyscan
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Usage
1. Ensure your data files are in the `data/` directory
2. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/flight_price_prediction.ipynb
```

3. For predictions using the saved model:
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/best_flight_price_model.joblib')

# Prepare your data
new_data = pd.DataFrame({
    'airline': ['Air India'],
    'class': ['Economy'],
    'source_city': ['Delhi'],
    'destination_city': ['Bangalore'],
    'departure_time': ['Morning'],
    'stops': ['one'],
    'arrival_time': ['Evening'],
    'duration': [10.5],
    'days_left': [20]
})

# Make prediction
predicted_price = model.predict(new_data)
```

## Results
- Best performing model: Random Forest Regressor
- R-squared score: 0.922
- Mean Squared Error: 39,798,553.49

## License
This project is licensed under the MIT License - see the LICENSE file for details.