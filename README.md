# Commodity Prediction Feature Engineering

This project contains a Python class, CommodityAnalyzer, designed for comprehensive feature engineering on time-series data related to financial markets, with a focus on commodities, stocks, and foreign exchange (FX). The goal is to transform raw market data into a rich dataset suitable for training a machine learning model to predict future commodity prices.

The code is modular and organized into a single class that encapsulates the entire feature engineering pipeline.

## ⚙️ How It Works
The core of this project is the CommodityAnalyzer class, which takes raw data and generates a wide array of new features. The process is broken down into several key steps:

1. Data Ingestion: The class is designed to process pandas DataFrames, handling large datasets efficiently.

2. Feature Generation: It applies a series of methods to create new features from the raw data. These features fall into distinct categories:

     **Temporal Features**: Captures time-based patterns like day of the week, month, and quarter.

    **Price & Momentum Features**: Calculates various technical indicators like returns, moving averages (MA), exponential moving averages (EMA), and the Relative Strength Index (RSI).

    **Volatility Features**: Measures market fluctuation using rolling volatility and Average True Range (ATR).

    **Volume & Open Interest Features**: Analyzes trading volume and open interest to identify market trends and liquidity.

    **Cross-Asset Features**: Creates synthetic market indices (e.g., a "USD strength index" or a "JPX precious metals index") to capture inter-market relationships.

    **Lag & Target-Based Features**: Uses past values of the target variables (the values you are trying to predict) to create highly predictive features for the model.

    **Pipeline Execution**: The engineer_all_features method orchestrates this entire process, applying the transformations to both training and test datasets.

## 🗂️ Project Structure
The code is contained within a single class to promote reusability and maintainability.

    CommodityAnalyzer: The main class responsible for all feature engineering.

    _calculate_rsi: A helper method for calculating the Relative Strength Index.

    if __name__ == "__main__" block: A simple test case that demonstrates how to use the class with sample data.