# MICROSOFT_STOCK_PRICE_PREDICTION
A deep learning–based time series forecasting project that uses an LSTM neural network to analyze historical Microsoft stock data and predict future closing prices.

## Table of Contents
- Project Overview
- Features
- Dataset
- Installation
- Usage
- Methodology
- Model Architecture
- Results
- Visualizations
- Future Improvements
- License

## Project Overview
The Microsoft Stock Price Prediction project is a deep learning-based time series forecasting application that predicts future closing prices of Microsoft (MSFT) stock. The project uses historical stock price data to train an LSTM (Long Short-Term Memory) model, a type of recurrent network (RNN) well-suited for sequential data. The main goal is to provide insights into potential stock price movements and help in financial decision-making.

## Features
- **Historical Data Analysis:** Explore and visualize MSFT stock prices from 2013–2018.
- **Data Preprocessing:** Clean, scale, and structure data for training.
- **LSTM Model:** Leverages deep learning for sequential pattern recognition in stock prices.
- **Sliding Window Approach:** Uses the past 60 days of prices to predict the next day’s closing price.
- **Prediction Evaluation:** Evaluates model performance using RMSE (Root Mean Squared Error).
- **Visualization:** Compare actual vs predicted prices with interactive plots.

## Dataset
The dataset used in this project is a CSV file containing Microsoft stock prices from 2013 to 2018.

**Columns include:**
- `date`: Trading date
- `open`: Opening stock price
- `high`: Highest price of the day
- `low`: Lowest price of the day
- `close`: Closing stock price
- `volume`: Number of shares traded
- `Name`: Stock ticker (MSFT)

You can find the dataset here or upload your own historical MSFT CSV.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/MICROSOFT_STOCK_PRICE_PREDICTION.git
   cd MICROSOFT_STOCK_PRICE_PREDICTION
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

**Requirements include:**
- Python >= 3.8
- TensorFlow
- pandas, numpy, matplotlib, seaborn, scikit-learn

## Usage

1. Place the dataset CSV (`MicrosoftStock.csv`) in the project folder.
2. Run the main Python script:
   ```bash
   python microsoft_stock_prediction.py
   ```

**The script will:**
- Load and visualize the data
- Preprocess the data
- Train the LSTM model
- Make predictions
- Generate plots comparing actual vs predicted stock prices

## Methodology

### 1. Data Cleaning and Exploration
- Remove non-numeric columns for correlation analysis.
- Handle missing values (none in this dataset).
- Visualize stock prices and trading volumes over time.

### 2. Feature Scaling
- Standardize data using `StandardScaler` from scikit-learn to improve LSTM performance.

### 3. Sliding Window Creation
- Use the last 60 days of closing prices as input features.
- Predict the next day’s closing price.

### 4. Model Training
- LSTM layers to capture sequential dependencies in stock prices.
- Dense and Dropout layers to reduce overfitting.

### 5. Evaluation
- Use RMSE to measure prediction accuracy.
- Compare predicted vs actual closing prices through visualizations.

## Model Architecture

| Layer | Type | Details |
| :--- | :--- | :--- |
| **Layer 1** | LSTM | 64 units, `return_sequences=True` |
| **Layer 2** | LSTM | 64 units, `return_sequences=False` |
| **Layer 3** | Dense | 128 units, ReLU activation |
| **Layer 4** | Dropout | 0.5 rate |
| **Output** | Dense | 1 unit |

- **Optimizer:** Adam
- **Loss Function:** Mean Absolute Error (MAE)
- **Metric:** Root Mean Squared Error (RMSE)

## Results
- **Training Performance:** Model trained for 20 epochs with batch size 32.
- **Prediction Accuracy:** Root Mean Squared Error (RMSE): **~1.77**
- **Interpretation:** The model successfully captures the overall trends in Microsoft stock prices, providing close predictions to actual values.

## Visualizations
- Open-Close Prices Over Time
- Trading Volumes
- Feature Correlation Heatmap
- Predicted vs Actual Closing Prices

## Future Improvements
- Include additional features such as market indices, sentiment analysis, or technical indicators.
- Experiment with other architectures: GRU, Transformer models, or hybrid models.
- Extend dataset to include recent stock prices for real-time predictions.
- Hyperparameter tuning for improved performance.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
