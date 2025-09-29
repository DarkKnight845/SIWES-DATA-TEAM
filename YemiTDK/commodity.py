import numpy as np
import pandas as pd


class CommodityAnalyzer:
    def __init__(self, lookback_windows=[3, 5, 10, 20], volatility_window=[5, 10, 20]):
        self.lookback_windows = lookback_windows
        self.volatility_window = volatility_window
        self.scalers = {}

        self.lme_cols = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]

        self.jpx_price_cols = [
            "JPX_Gold_Mini_Futures_Open",
            "JPX_Gold_Rolling-Spot_Futures_Open",
            "JPX_Gold_Standard_Futures_Open",
            "JPX_Platinum_Mini_Futures_Open",
            "JPX_Platinum_Standard_Futures_Open",
            "JPX_RSS3_Rubber_Futures_Open",
            "JPX_Gold_Mini_Futures_High",
            "JPX_Gold_Rolling-Spot_Futures_High",
            "JPX_Gold_Standard_Futures_High",
            "JPX_Platinum_Mini_Futures_High",
            "JPX_Platinum_Standard_Futures_High",
            "JPX_RSS3_Rubber_Futures_High",
            "JPX_Gold_Mini_Futures_Low",
            "JPX_Gold_Rolling-Spot_Futures_Low",
            "JPX_Gold_Standard_Futures_Low",
            "JPX_Platinum_Mini_Futures_Low",
            "JPX_Platinum_Standard_Futures_Low",
            "JPX_RSS3_Rubber_Futures_Low",
            "JPX_Gold_Mini_Futures_Close",
            "JPX_Gold_Rolling-Spot_Futures_Close",
            "JPX_Gold_Standard_Futures_Close",
            "JPX_Platinum_Mini_Futures_Close",
            "JPX_Platinum_Standard_Futures_Close",
            "JPX_RSS3_Rubber_Futures_Close",
            "JPX_Gold_Mini_Futures_settlement_price",
            "JPX_Gold_Rolling-Spot_Futures_settlement_price",
            "JPX_Platinum_Mini_Futures_settlement_price",
            "JPX_RSS3_Rubber_Futures_settlement_price",
        ]

        self.jpx_volume_cols = [
            "JPX_Gold_Mini_Futures_Volume",
            "JPX_Gold_Rolling-Spot_Futures_Volume",
            "JPX_Gold_Standard_Futures_Volume",
            "JPX_Platinum_Mini_Futures_Volume",
            "JPX_Platinum_Standard_Futures_Volume",
            "JPX_RSS3_Rubber_Futures_Volume",
        ]

        self.jpx_oi_cols = [
            "JPX_Gold_Mini_Futures_open_interest",
            "JPX_Gold_Rolling-Spot_Futures_open_interest",
            "JPX_Gold_Standard_Futures_open_interest",
            "JPX_Platinum_Mini_Futures_open_interest",
            "JPX_Platinum_Standard_Futures_open_interest",
            "JPX_RSS3_Rubber_Futures_open_interest",
        ]

        self.fx_columns = [
            "FX_AUDJPY",
            "FX_AUDUSD",
            "FX_CADJPY",
            "FX_CHFJPY",
            "FX_EURAUD",
            "FX_EURGBP",
            "FX_EURJPY",
            "FX_EURUSD",
            "FX_GBPAUD",
            "FX_GBPJPY",
            "FX_GBPUSD",
            "FX_NZDJPY",
            "FX_NZDUSD",
            "FX_USDCHF",
            "FX_USDJPY",
            "FX_ZARJPY",
            "FX_ZARUSD",
            "FX_NOKUSD",
            "FX_NOKEUR",
            "FX_CADUSD",
            "FX_AUDNZD",
            "FX_EURCHF",
            "FX_EURCAD",
            "FX_AUDCAD",
            "FX_GBPCHF",
            "FX_EURNZD",
            "FX_AUDCHF",
            "FX_GBPNZD",
            "FX_GBPCAD",
            "FX_CADCHF",
            "FX_NZDCAD",
            "FX_NZDCHF",
            "FX_ZAREUR",
            "FX_NOKGBP",
            "FX_NOKCHF",
            "FX_ZARCHF",
            "FX_NOKJPY",
            "FX_ZARGBP",
        ]

    def identify_us_stock_columns(self, df):
        """Identify US Stock columns dynamically"""
        us_stock_open = [
            col
            for col in df.columns
            if col.startswith("US_Stock_") and col.endswith("_adj_open")
        ]
        us_stock_high = [
            col
            for col in df.columns
            if col.startswith("US_Stock_") and col.endswith("_adj_high")
        ]
        us_stock_low = [
            col
            for col in df.columns
            if col.startswith("US_Stock_") and col.endswith("_adj_low")
        ]
        us_stock_close = [
            col
            for col in df.columns
            if col.startswith("US_Stock_") and col.endswith("_adj_close")
        ]
        us_stock_volume = [
            col
            for col in df.columns
            if col.startswith("US_Stock_") and col.endswith("_adj_volume")
        ]

        return {
            "open": us_stock_open,
            "high": us_stock_high,
            "low": us_stock_low,
            "close": us_stock_close,
            "volume": us_stock_volume,
        }

    def create_ohlc_features(self, df, prefix_cols):
        """Create OHLC-based technical features for stocks/futures with OHLC data"""
        features = df.copy()

        # Get base names (remove _open, _high, _low, _close suffixes)
        base_names = set()
        for col in prefix_cols["close"]:
            base_name = col.replace("_adj_close", "").replace("_Close", "")
            base_names.add(base_name)

        for base_name in base_names:
            # Find corresponding OHLC columns
            open_col = None
            high_col = None
            low_col = None
            close_col = None

            for col in df.columns:
                if base_name in col:
                    if col.endswith("_adj_open") or col.endswith("_Open"):
                        open_col = col
                    elif col.endswith("_adj_high") or col.endswith("_High"):
                        high_col = col
                    elif col.endswith("_adj_low") or col.endswith("_Low"):
                        low_col = col
                    elif col.endswith("_adj_close") or col.endswith("_Close"):
                        close_col = col

            if all([open_col, high_col, low_col, close_col]) and all(
                [col in df.columns for col in [open_col, high_col, low_col, close_col]]
            ):
                # Price spreads and ratios
                features[f"{base_name}_hl_spread"] = df[high_col] - df[low_col]
                features[f"{base_name}_oc_spread"] = df[close_col] - df[open_col]
                features[f"{base_name}_hl_ratio"] = df[high_col] / df[low_col]

                # Position within day's range
                features[f"{base_name}_close_position"] = (
                    df[close_col] - df[low_col]
                ) / (df[high_col] - df[low_col] + 1e-8)

                # True Range (for volatility)
                prev_close = df[close_col].shift(1)
                tr1 = df[high_col] - df[low_col]
                tr2 = np.abs(df[high_col] - prev_close)
                tr3 = np.abs(df[low_col] - prev_close)
                features[f"{base_name}_true_range"] = np.maximum(
                    tr1, np.maximum(tr2, tr3)
                )

                # Average True Range
                for window in [5, 14, 20]:
                    features[f"{base_name}_atr_{window}"] = (
                        features[f"{base_name}_true_range"].rolling(window).mean()
                    )

        return features

    def create_price_momentum_features(self, df, price_columns):
        """Create momentum and trend features for price columns"""
        features = df.copy()

        for col in price_columns:
            if col not in df.columns:
                continue

            # Basic returns
            for window in self.lookback_windows:
                features[f"{col}_return_{window}d"] = df[col].pct_change(window)
                features[f"{col}_log_return_{window}d"] = np.log(
                    df[col] / df[col].shift(window)
                )

            # Moving averages and deviations
            for window in [5, 10, 20, 50]:
                ma = df[col].rolling(window=window, min_periods=1).mean()
                features[f"{col}_ma_{window}"] = ma
                features[f"{col}_ma_ratio_{window}"] = df[col] / ma

                # Exponential moving average
                ema = df[col].ewm(span=window).mean()
                features[f"{col}_ema_{window}"] = ema
                features[f"{col}_ema_ratio_{window}"] = df[col] / ema

            # Price momentum indicators
            features[f"{col}_roc_5d"] = (df[col] / df[col].shift(5) - 1) * 100
            features[f"{col}_roc_10d"] = (df[col] / df[col].shift(10) - 1) * 100

            # RSI
            features[f"{col}_rsi_14"] = self._calculate_rsi(df[col], 14)

            # Price percentile (where current price stands relative to recent history)
            for window in [20, 60]:
                rolling_min = df[col].rolling(window).min()
                rolling_max = df[col].rolling(window).max()
                features[f"{col}_percentile_{window}d"] = (df[col] - rolling_min) / (
                    rolling_max - rolling_min + 1e-8
                )

        return features

    def create_volatility_features(self, df, price_columns):
        """Create volatility-based features"""
        features = df.copy()

        for col in price_columns:
            if col not in df.columns:
                continue

            returns = df[col].pct_change()

            # Rolling volatility (multiple windows)
            for window in self.volatility_window:
                features[f"{col}_volatility_{window}d"] = returns.rolling(window).std()
                features[f"{col}_realized_vol_{window}d"] = np.sqrt(
                    (returns**2).rolling(window).sum()
                )

            # GARCH-like volatility
            squared_returns = returns**2
            for window in [5, 20]:
                features[f"{col}_vol_ma_{window}d"] = squared_returns.rolling(
                    window
                ).mean()

        return features

    def create_volume_features(self, df, volume_columns):
        """Create volume-based features"""
        features = df.copy()

        for col in volume_columns:
            if col not in df.columns or df[col].isna().all():
                continue

            # Volume transformations
            features[f"{col}_log"] = np.log(df[col] + 1)

            # Volume moving averages
            for window in [5, 10, 20]:
                vol_ma = df[col].rolling(window=window, min_periods=1).mean()
                features[f"{col}_ma_{window}"] = vol_ma
                features[f"{col}_ratio_{window}"] = df[col] / (vol_ma + 1e-8)

            # Volume rate of change
            for window in [1, 5, 10]:
                features[f"{col}_roc_{window}d"] = df[col] / df[col].shift(window) - 1

        return features

    def create_cross_asset_features(self, df):
        """Create cross-asset and market regime features"""
        features = df.copy()

        # Identify available columns
        us_stocks = self.identify_us_stock_columns(df)

        # Create market indices by asset class
        if self.lme_cols and any(col in df.columns for col in self.lme_cols):
            lme_data = df[[col for col in self.lme_cols if col in df.columns]]
            features["LME_index"] = lme_data.mean(axis=1, skipna=True)
            features["LME_volatility_10d"] = (
                features["LME_index"].pct_change().rolling(10).std()
            )

        # JPX precious metals index
        jpx_precious_cols = [
            col
            for col in self.jpx_price_cols
            if col in df.columns
            and ("Gold" in col or "Platinum" in col)
            and "Close" in col
        ]
        if jpx_precious_cols:
            features["JPX_precious_index"] = df[jpx_precious_cols].mean(
                axis=1, skipna=True
            )
            features["JPX_precious_vol_10d"] = (
                features["JPX_precious_index"].pct_change().rolling(10).std()
            )

        # US equity indices by sector
        if us_stocks["close"]:
            # Create broad US equity index
            us_close_data = df[[col for col in us_stocks["close"] if col in df.columns]]
            features["US_equity_index"] = us_close_data.mean(axis=1, skipna=True)
            features["US_equity_vol_10d"] = (
                features["US_equity_index"].pct_change().rolling(10).std()
            )

            # Sector-specific indices
            commodity_stocks = [
                col
                for col in us_stocks["close"]
                if any(
                    ticker in col for ticker in ["FCX", "NEM", "GOLD", "SCCO", "VALE"]
                )
            ]
            if commodity_stocks:
                features["US_commodity_index"] = df[commodity_stocks].mean(
                    axis=1, skipna=True
                )

        # FX indices
        if self.fx_columns and any(col in df.columns for col in self.fx_columns):
            # USD strength index (average of USD pairs)
            usd_pairs = [
                col for col in self.fx_columns if col in df.columns and "USD" in col
            ]
            if usd_pairs:
                # Normalize USD pairs (some are USD/XXX, some are XXX/USD)
                usd_strength_data = []
                for col in usd_pairs:
                    if col.startswith("FX_USD"):  # USD is base currency
                        usd_strength_data.append(df[col])
                    else:  # USD is quote currency, invert
                        usd_strength_data.append(1 / df[col])

                if usd_strength_data:
                    features["USD_strength_index"] = pd.concat(
                        usd_strength_data, axis=1
                    ).mean(axis=1, skipna=True)
                    features["USD_strength_vol_10d"] = (
                        features["USD_strength_index"].pct_change().rolling(10).std()
                    )

        return features

    def create_temporal_features(self, df, date_col="date_id"):
        """Create time-based features"""
        features = df.copy()

        # Create date column (assuming date_id is sequential days)
        if df[date_col].dtype in ["int64", "float64"]:
            # Convert to actual dates if needed
            features["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
                df[date_col], unit="D"
            )

        # Extract temporal features
        features["day_of_week"] = features["date"].dt.dayofweek
        features["day_of_month"] = features["date"].dt.day
        features["month"] = features["date"].dt.month
        features["quarter"] = features["date"].dt.quarter

        # Cyclical encoding
        features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        # Market structure indicators
        features["is_month_end"] = features["date"].dt.is_month_end.astype(int)
        features["is_quarter_end"] = features["date"].dt.is_quarter_end.astype(int)

        return features

    def create_target_lag_features(self, train_labels, target_pairs):
        """Create features based on target pairs and their lags"""
        features = train_labels.copy()

        # Group targets by lag
        lag_groups = target_pairs.groupby("lag")

        for lag, group in lag_groups:
            target_cols = group["target"].tolist()

            # Create lagged versions
            for target in target_cols:
                if target in train_labels.columns:
                    # Multiple lag features
                    for lag_val in [1, 2, 3, 5]:
                        features[f"{target}_lag_{lag_val}"] = train_labels[
                            target
                        ].shift(lag_val)

                    # Rolling statistics of targets
                    for window in [5, 10, 20]:
                        features[f"{target}_rolling_mean_{window}"] = (
                            train_labels[target].rolling(window).mean()
                        )
                        features[f"{target}_rolling_std_{window}"] = (
                            train_labels[target].rolling(window).std()
                        )

                        # Z-score relative to recent history
                        mean_val = train_labels[target].rolling(window).mean()
                        std_val = train_labels[target].rolling(window).std()
                        features[f"{target}_zscore_{window}"] = (
                            train_labels[target] - mean_val
                        ) / (std_val + 1e-8)

        return features

    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def engineer_all_features(
        self, train_df, test_df, train_labels=None, target_pairs=None
    ):
        """Apply comprehensive feature engineering"""
        print("Starting MITSUI-specific feature engineering...")

        results = {}

        for name, df in [("train", train_df), ("test", test_df)]:
            print(f"Processing {name} data...")

            # Start with original data
            features_df = df.copy()

            # Temporal features
            features_df = self.create_temporal_features(features_df)

            # Identify US stock columns
            us_stocks = self.identify_us_stock_columns(df)

            # OHLC features for US stocks (they have open/high/low/close)
            if us_stocks["close"]:
                features_df = self.create_ohlc_features(features_df, us_stocks)

            # Price momentum features
            all_price_cols = (
                self.lme_cols
                + self.jpx_price_cols
                + us_stocks["close"]
                + self.fx_columns
            )
            existing_price_cols = [col for col in all_price_cols if col in df.columns]
            if existing_price_cols:
                features_df = self.create_price_momentum_features(
                    features_df, existing_price_cols
                )
                features_df = self.create_volatility_features(
                    features_df, existing_price_cols
                )

            # Volume features
            all_volume_cols = self.jpx_volume_cols + us_stocks["volume"]
            existing_volume_cols = [col for col in all_volume_cols if col in df.columns]
            if existing_volume_cols:
                features_df = self.create_volume_features(
                    features_df, existing_volume_cols
                )

            # Cross-asset features
            features_df = self.create_cross_asset_features(features_df)

            results[name] = features_df

        if train_labels is not None and target_pairs is not None:
            print("Creating target-based features...")
            target_features = self.create_target_lag_features(
                train_labels, target_pairs
            )
            results["target_features"] = target_features

        print("Feature engineering completed!")
        for name, df in results.items():
            print(f"{name} shape: {df.shape}")

        return results

    def get_feature_importance_groups(self):
        """Return feature groups for analysis and selection"""
        return {
            "price_momentum": [
                "_return_",
                "_log_return_",
                "_ma_",
                "_ema_",
                "_roc_",
                "_rsi_",
                "_percentile_",
            ],
            "volatility": ["_volatility_", "_realized_vol_", "_vol_ma_", "_atr_"],
            "volume": ["_volume_", "_vol_ma_", "_vol_ratio_", "_vol_roc_"],
            "technical": ["_hl_spread", "_oc_spread", "_close_position", "_true_range"],
            "cross_asset": ["_index", "USD_strength", "LME_", "JPX_", "US_"],
            "temporal": [
                "day_of_week",
                "month",
                "quarter",
                "is_month_end",
                "is_quarter_end",
            ],
            "target_based": ["target_", "_lag_", "_rolling_", "_zscore_"],
        }


def process_mitsui_data(train_path, test_path, train_labels_path, target_pairs_path):
    """
    Complete pipeline for processing MITSUI competition data
    """
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(
        r"C:\Users\USER\Documents\Commodity_Prediction\mitsui-commodity-prediction-challenge\train.csv"
    )
    test_df = pd.read_csv(
        r"C:\Users\USER\Documents\Commodity_Prediction\mitsui-commodity-prediction-challenge\test.csv"
    )
    train_labels = pd.read_csv(
        r"C:\Users\USER\Documents\Commodity_Prediction\mitsui-commodity-prediction-challenge\train_labels.csv"
    )
    target_pairs = pd.read_csv(
        r"C:\Users\USER\Documents\Commodity_Prediction\mitsui-commodity-prediction-challenge\target_pairs.csv"
    )

    # Initialize feature engineer
    fe = CommodityAnalyzer()

    # Engineer features
    results = fe.engineer_all_features(train_df, test_df, train_labels, target_pairs)

    # Get feature groups for selection
    feature_groups = fe.get_feature_importance_groups()

    print("\nFeature groups created:")
    for group, patterns in feature_groups.items():
        matching_features = []
        for pattern in patterns:
            matching_features.extend(
                [col for col in results["train"].columns if pattern in col]
            )
        print(f"{group}: {len(set(matching_features))} features")

    return results, feature_groups


if __name__ == "__main__":

    np.random.seed(42)
    n_days = 1000

    sample_data = {
        "date_id": range(n_days),
        "LME_AH_Close": np.random.randn(n_days).cumsum() + 2600,
        "LME_CA_Close": np.random.randn(n_days).cumsum() + 9000,
        "LME_PB_Close": np.random.randn(n_days).cumsum() + 2000,
        "LME_ZS_Close": np.random.randn(n_days).cumsum() + 2900,
        "JPX_Gold_Mini_Futures_Close": np.random.randn(n_days).cumsum() + 13000,
        "JPX_Platinum_Mini_Futures_Close": np.random.randn(n_days).cumsum() + 4600,
        "US_Stock_VT_adj_close": np.random.randn(n_days).cumsum() + 100,
        "US_Stock_GLD_adj_close": np.random.randn(n_days).cumsum() + 180,
        "FX_EURUSD": np.random.randn(n_days).cumsum() + 1.1,
        "FX_USDJPY": np.random.randn(n_days).cumsum() + 110,
    }

    sample_df = pd.DataFrame(sample_data)
    fe = CommodityAnalyzer()

    # Test feature engineering
    results = fe.engineer_all_features(sample_df, sample_df.tail(100))

    print(f"Original features: {len(sample_df.columns)}")
    print(f"Engineered features: {len(results['train'].columns)}")
    print(
        f"New features added: {len(results['train'].columns) - len(sample_df.columns)}"
    )
