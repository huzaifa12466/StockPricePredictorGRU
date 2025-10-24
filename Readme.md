# Multi-Ticker Stock Price Prediction (GRU)

**📈 Predicting future stock prices for multiple tickers using a GRU-based deep learning model with interactive visualization in Streamlit.**

---

## Project Overview

This project implements a **multi-ticker stock price prediction system** using **GRU (Gated Recurrent Unit) neural networks**. The system is capable of:

* Predicting the next N days of stock prices for selected tickers.
* Comparing multiple tickers interactively.
* Visualizing recent and predicted stock prices using interactive plots.
* Providing predictive metrics for performance evaluation.

---

## Key Features

* **Multi-Ticker Prediction:** Trained on multiple stocks including AMZN, AAPL, MSFT, GOOGL, META.
* **Interactive Visualization:** Real-time interactive line plots using **Plotly** in Streamlit. Hover over the line to see exact prices.
* **Prediction Table:** Shows predicted prices in an easy-to-read table format.
* **Comparison:** Compare predictions and trends between two tickers.
* **Moving Average:** Overlay moving average for trend analysis.

---

## Folder Structure

```
project-root/
│
├── models/
│   ├── model.py         # GRU model definition
│   ├── predict.py       # Prediction functions
│   └── best_multi_ticker_model.pth  # Trained model weights
│
├── notebooks/
│   └── stock_prediction_notebook.ipynb  # EDA, data preprocessing, and model training notebook
│
├── streamlit_app.py     # Interactive Streamlit dashboard
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
```

---

## Tech Stack

* **Python 3.12+**
* **PyTorch:** For building and training GRU neural network.
* **scikit-learn:** Data preprocessing and evaluation metrics.
* **pandas / numpy:** Data manipulation.
* **yfinance:** Download historical stock data.
* **Matplotlib / Plotly / Seaborn:** Visualizations.
* **Streamlit:** Interactive web application.

---

## Data

* Historical stock prices from **Yahoo Finance** for:

  * Amazon (AMZN)
  * Apple (AAPL)
  * Microsoft (MSFT)
  * Alphabet/Google (GOOGL)
  * Meta/Facebook (META)
* Data period: **2015-01-01 to 2025-01-01**
* Feature used: **Closing price**

---

## Model Architecture

* **Type:** GRU (Gated Recurrent Unit)
* **Input:** Past 30 days closing prices (sequence length = 30)
* **Hidden Layers:** 2 layers with 128 hidden units
* **Dropout:** 0.2
* **Output:** Predicted next day closing price
* **Loss Function:** Huber Loss
* **Optimizer:** Adam
* **Early Stopping:** Patience = 10 epochs

---

## Training Results

| Ticker | RMSE  | MAE   | R²    |
| ------ | ----- | ----- | ----- |
| AMZN   | 6.056 | 3.889 | 0.998 |
| AAPL   | 4.600 | 2.932 | 0.999 |
| MSFT   | 4.548 | 2.874 | 0.999 |
| GOOGL  | 5.618 | 3.563 | 0.998 |
| META   | 5.894 | 3.548 | 0.998 |

**Interpretation:**
High R² scores indicate excellent fit; low RMSE and MAE indicate predictions are close to actual values.

---

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/huzaifa12466/StockPricePredictorGRU.git
```

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run Streamlit app:**

```bash
streamlit run streamlit_app.py
```

5. **Interact with the app:**

   * Select a ticker.
   * Optionally select a second ticker to compare.
   * Choose the number of days to predict.
   * Explore interactive plots and tables.

---

## Potential Improvements

* **Add more features:** Include volume, moving averages, RSI, MACD, or sentiment analysis to improve prediction accuracy.
* **Hyperparameter Tuning:** Experiment with GRU layers, hidden units, learning rates, and sequence length.
* **Multi-step Forecasting:** Predict multiple days simultaneously instead of recursively for better stability.
* **Portfolio Dashboard:** Allow users to create a watchlist and visualize correlations between multiple tickers.
* **Deployment:** Deploy as a full web app with authentication and real-time data updates.
* **Model Explainability:** Add SHAP or LIME to explain predictions.
