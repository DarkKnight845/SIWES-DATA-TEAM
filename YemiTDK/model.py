import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class MitsuiModelPipeline:
    """
    Comprehensive modeling pipeline for MITSUI Commodity Prediction Challenge
    Handles 424 targets with different lag structures
    """

    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.cv_scores = {}

    def prepare_data(self, features_df, labels_df, target_pairs_df):
        """
        Prepare data for modeling by aligning features with targets
        """
        print("Preparing data for modeling...")

        # Merge features with labels on date_id
        merged_data = features_df.merge(labels_df, on="date_id", how="inner")

        # Get target columns
        target_cols = [col for col in labels_df.columns if col.startswith("target_")]

        # Get feature columns (exclude date_id, is_scored, targets, and date)
        exclude_cols = target_cols + ["date_id", "is_scored", "date"]
        feature_cols = [
            col
            for col in merged_data.columns
            if col not in exclude_cols
            and merged_data[col].dtype in ["int64", "float64"]
        ]

        # Remove features with too many NaNs (>50%)
        valid_features = []
        for col in feature_cols:
            nan_ratio = merged_data[col].isna().sum() / len(merged_data)
            if nan_ratio < 0.5:
                valid_features.append(col)

        print(
            f"Total features: {len(valid_features)} (removed {len(feature_cols) - len(valid_features)} with >50% NaN)"
        )
        print(f"Total targets: {len(target_cols)}")

        return merged_data, valid_features, target_cols

    def create_time_series_splits(self, data, n_splits=5):
        """
        Create time series cross-validation splits
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []

        for train_idx, val_idx in tscv.split(data):
            splits.append((train_idx, val_idx))

        return splits

    def select_features(
        self, X_train, y_train, feature_names, n_features=200, method="lgb"
    ):
        """
        Select top features using tree-based feature importance
        """
        print(f"Selecting top {n_features} features using {method}...")

        if method == "lgb":
            # Train a quick LightGBM model for feature selection
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1,
            )
        elif method == "xgb":
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,
                verbosity=0,
            )

        model.fit(X_train, y_train)

        # Get feature importance
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Select top features
        top_features = importance.head(n_features)["feature"].tolist()

        return top_features, importance

    def train_lgb_model(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train a LightGBM model with given parameters
        """
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "max_depth": -1,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": self.random_state,
                "verbose": -1,
            }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )

        return model

    def train_xgb_model(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train an XGBoost model with given parameters
        """
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "verbosity": 0,
            }

        model = xgb.XGBRegressor(**params, n_estimators=1000)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )

        return model

    def train_single_target(
        self, merged_data, feature_cols, target_col, model_type="lgb", n_features=200
    ):
        """
        Train model for a single target with cross-validation
        """
        print(f"\nTraining model for {target_col}...")

        # Remove rows with missing target values
        valid_data = merged_data[merged_data[target_col].notna()].copy()

        if len(valid_data) < 100:
            print(f"Skipping {target_col} - insufficient data ({len(valid_data)} rows)")
            return None

        # Prepare features and target
        X = valid_data[feature_cols].fillna(0).values
        y = valid_data[target_col].values

        # Create time series splits
        splits = self.create_time_series_splits(valid_data, self.n_folds)

        # Store models and scores for each fold
        fold_models = []
        fold_scores = []
        fold_predictions = []

        # Feature selection on first fold
        train_idx, val_idx = splits[0]
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]

        selected_features, importance = self.select_features(
            X_train_fold, y_train_fold, feature_cols, n_features, method=model_type
        )

        # Store feature importance
        self.feature_importance[target_col] = importance

        # Get indices of selected features
        feature_indices = [
            i for i, col in enumerate(feature_cols) if col in selected_features
        ]

        # Cross-validation with selected features
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fold + 1}/{self.n_folds}...", end=" ")

            X_train_fold = X[train_idx][:, feature_indices]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx][:, feature_indices]
            y_val_fold = y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)

            # Train model
            if model_type == "lgb":
                model = self.train_lgb_model(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )
            elif model_type == "xgb":
                model = self.train_xgb_model(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )

            # Predict on validation set
            if model_type == "lgb":
                val_pred = model.predict(X_val_fold)
            else:
                val_pred = model.predict(X_val_fold)

            # Calculate score
            val_score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            fold_scores.append(val_score)
            fold_models.append(model)
            fold_predictions.append(val_pred)

            print(f"RMSE: {val_score:.6f}")

        # Calculate average CV score
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(f"  Average CV RMSE: {avg_score:.6f} (+/- {std_score:.6f})")

        # Store results
        result = {
            "models": fold_models,
            "selected_features": selected_features,
            "feature_indices": feature_indices,
            "cv_scores": fold_scores,
            "avg_cv_score": avg_score,
            "std_cv_score": std_score,
            "scaler": scaler,  # Store last fold scaler (could use ensemble of scalers)
        }

        return result

    def train_all_targets(
        self,
        merged_data,
        feature_cols,
        target_cols,
        model_type="lgb",
        n_features=200,
        max_targets=None,
    ):
        """
        Train models for all targets (or subset for testing)
        """
        print(f"\n{'=' * 60}")
        print(f"Training {model_type.upper()} models for all targets")
        print(f"{'=' * 60}")

        if max_targets:
            target_cols = target_cols[:max_targets]
            print(f"Training on first {max_targets} targets only (testing mode)")

        results = {}
        successful_targets = 0
        failed_targets = 0

        for i, target_col in enumerate(target_cols, 1):
            print(f"\n[{i}/{len(target_cols)}] Processing {target_col}...")

            try:
                result = self.train_single_target(
                    merged_data,
                    feature_cols,
                    target_col,
                    model_type=model_type,
                    n_features=n_features,
                )

                if result:
                    results[target_col] = result
                    successful_targets += 1
                    self.models[target_col] = result
                else:
                    failed_targets += 1

            except Exception as e:
                print(f"  Error training {target_col}: {str(e)}")
                failed_targets += 1
                continue

        print(f"\n{'=' * 60}")
        print("Training Summary:")
        print(f"  Successful: {successful_targets}")
        print(f"  Failed: {failed_targets}")
        print(f"  Total: {len(target_cols)}")
        print(f"{'=' * 60}")

        return results

    def predict_single_target(self, X, target_col, model_type="lgb"):
        """
        Make predictions for a single target using ensemble of CV models
        """
        if target_col not in self.models:
            return None

        model_info = self.models[target_col]
        feature_indices = model_info["feature_indices"]
        scaler = model_info["scaler"]
        models = model_info["models"]

        # Select and scale features
        X_selected = X[:, feature_indices]
        X_scaled = scaler.transform(X_selected)

        # Ensemble predictions from all folds
        predictions = []
        for model in models:
            if model_type == "lgb":
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_scaled)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def predict_all_targets(self, test_features, feature_cols, model_type="lgb"):
        """
        Make predictions for all targets on test data
        """
        print(f"\nGenerating predictions for {len(self.models)} targets...")

        # Ensure all feature columns exist in test data
        missing_cols = [col for col in feature_cols if col not in test_features.columns]
        if missing_cols:
            print(
                f"Warning: {len(missing_cols)} features missing in test data. Adding with zeros."
            )
            for col in missing_cols:
                test_features[col] = 0

        # Ensure column order matches
        X_test = test_features[feature_cols].fillna(0).values
        predictions = {}

        for target_col in self.models.keys():
            pred = self.predict_single_target(X_test, target_col, model_type)
            if pred is not None:
                predictions[target_col] = pred

        # Create predictions dataframe
        pred_df = pd.DataFrame(predictions)
        pred_df.insert(0, "date_id", test_features["date_id"].values)

        return pred_df

    def get_cv_summary(self):
        """
        Get summary of cross-validation scores
        """
        if not self.models:
            print("No models trained yet!")
            return None

        summary = []
        for target_col, model_info in self.models.items():
            summary.append(
                {
                    "target": target_col,
                    "avg_cv_rmse": model_info["avg_cv_score"],
                    "std_cv_rmse": model_info["std_cv_score"],
                    "n_features": len(model_info["selected_features"]),
                }
            )

        summary_df = pd.DataFrame(summary).sort_values("avg_cv_rmse")

        print("\nCross-Validation Summary:")
        print(f"{'=' * 70}")
        print(f"{'Target':<15} {'Avg RMSE':<12} {'Std RMSE':<12} {'N Features':<12}")
        print(f"{'=' * 70}")

        for _, row in summary_df.head(10).iterrows():
            print(
                f"{row['target']:<15} {row['avg_cv_rmse']:<12.6f} "
                f"{row['std_cv_rmse']:<12.6f} {row['n_features']:<12}"
            )

        print(f"{'=' * 70}")
        print(
            f"Average RMSE across all targets: {summary_df['avg_cv_rmse'].mean():.6f}"
        )
        print(
            f"Median RMSE across all targets: {summary_df['avg_cv_rmse'].median():.6f}"
        )

        return summary_df

    def save_models(self, filepath):
        """
        Save trained models to disk
        """
        import joblib

        joblib.dump(self.models, filepath)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath):
        """
        Load trained models from disk
        """
        import joblib

        self.models = joblib.load(filepath)
        print(f"Models loaded from {filepath}")


# Complete training pipeline
def run_complete_pipeline(
    features_train,
    features_test,
    labels_train,
    target_pairs,
    model_type="lgb",
    n_features=200,
    n_folds=5,
    max_targets=None,
    save_path=None,
):
    """
    Run complete modeling pipeline from feature engineering to predictions
    """
    print("\n" + "=" * 70)
    print("MITSUI COMMODITY PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Initialize pipeline
    pipeline = MitsuiModelPipeline(n_folds=n_folds, random_state=42)

    # Prepare data
    merged_data, feature_cols, target_cols = pipeline.prepare_data(
        features_train, labels_train, target_pairs
    )

    # Align test features with training features
    print("\nAligning train and test features...")
    train_feature_set = set(feature_cols)
    test_feature_set = set(features_test.columns)

    # Features in train but not in test
    missing_in_test = train_feature_set - test_feature_set
    if missing_in_test:
        print(
            f"Warning: {len(missing_in_test)} features in train missing from test. Adding with zeros."
        )
        for col in missing_in_test:
            features_test[col] = 0

    # Features in test but not in train (we'll ignore these)
    extra_in_test = test_feature_set - train_feature_set - {"date_id", "is_scored"}
    if extra_in_test:
        print(f"Info: {len(extra_in_test)} extra features in test will be ignored.")

    # Only use features that exist in both (or have been added to test)
    final_feature_cols = [col for col in feature_cols if col in features_test.columns]
    print(f"Final feature count: {len(final_feature_cols)}")

    # Train models
    results = pipeline.train_all_targets(
        merged_data,
        final_feature_cols,
        target_cols,
        model_type=model_type,
        n_features=n_features,
        max_targets=max_targets,
    )

    # Get CV summary
    cv_summary = pipeline.get_cv_summary()

    # Generate predictions on test set
    predictions = pipeline.predict_all_targets(
        features_test, final_feature_cols, model_type=model_type
    )

    # Save models if path provided
    if save_path:
        pipeline.save_models(save_path)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return pipeline, predictions, cv_summary


# Example usage
if __name__ == "__main__":
    import pandas as pd

    from YemiTDK.commodityv2 import MitsuiCommodityFeatureEngineer

    # Paths
    base_path = r"C:\Users\ayemi\OneDrive\Documents\Commodity_Prediction\mitsui-commodity-prediction-challenge"

    # Load raw data
    print("Loading data...")
    train_df = pd.read_csv(f"{base_path}\\train.csv")
    test_df = pd.read_csv(f"{base_path}\\test.csv")
    train_labels = pd.read_csv(f"{base_path}\\train_labels.csv")
    target_pairs = pd.read_csv(f"{base_path}\\target_pairs.csv")

    # Feature engineering
    print("\nFeature Engineering...")
    fe = MitsuiCommodityFeatureEngineer()
    results = fe.engineer_all_features(train_df, test_df, train_labels, target_pairs)

    # Model training
    print("\nModel Training...")
    pipeline, predictions, cv_summary = run_complete_pipeline(
        features_train=results["train"],
        features_test=results["test"],
        labels_train=train_labels,  # Use original labels, not target_features
        target_pairs=target_pairs,
        model_type="lgb",
        n_features=150,  # Reduced for faster training
        n_folds=3,  # Reduced for speed
        max_targets=20,  # Start with 20 targets for testing
        save_path="mitsui_models_v1.pkl",
    )

    # Save predictions
    predictions.to_csv("predictions_v1.csv", index=False)
    print("\nPredictions saved to predictions_v1.csv")
