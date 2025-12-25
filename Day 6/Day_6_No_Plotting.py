"""
Grocery Sales Forecasting System
Predict next day sales for a small store using simple features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GrocerySalesForecastSystem:
    """
    A sales forecasting system for grocery stores
    Predicts next day sales using historical data and simple features
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.df = None
        
    def generate_synthetic_data(self, days=365, start_date='2023-01-01'):
        """
        Generate synthetic grocery store sales data
        In practice, you would load your actual sales data
        """
        print("Generating synthetic grocery store sales data...")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        np.random.seed(42)
        
        # Base sales with trend
        base_sales = 1000 + np.linspace(0, 200, days)  # Growing trend
        
        # Day of week effect (higher on weekends)
        day_of_week_effect = np.array([
            100 if date.dayofweek >= 5 else 0  # Saturday, Sunday
            for date in dates
        ])
        
        # Monthly seasonality
        month_effect = np.array([
            50 * np.sin(2 * np.pi * date.month / 12) + 50
            for date in dates
        ])
        
        # Holiday effect (simplified)
        holiday_effect = np.array([
            200 if date.month == 12 and date.day >= 20  # Christmas season
            or date.month == 11 and date.day >= 20  # Thanksgiving
            or date.month == 7 and date.day == 4  # July 4th
            else 0
            for date in dates
        ])
        
        # Weather effect (simplified - rainy days reduce sales)
        weather_effect = np.random.choice([-50, 0, 0, 0], size=days)
        
        # Random noise
        noise = np.random.normal(0, 50, days)
        
        # Calculate total sales
        sales = (base_sales + day_of_week_effect + month_effect + 
                holiday_effect + weather_effect + noise)
        
        # Ensure non-negative sales
        sales = np.maximum(sales, 100)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'date': dates,
            'sales': sales
        })
        
        print(f"Generated {len(self.df)} days of sales data")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Average daily sales: ${self.df['sales'].mean():.2f}")
        print(f"Sales range: ${self.df['sales'].min():.2f} - ${self.df['sales'].max():.2f}")
        
        return self.df
    
    def engineer_features(self):
        """Create features from the date and historical sales data"""
        print("\nEngineering features...")
        
        df = self.df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is month start/end
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        df['season'] = ((df['month'] % 12 + 3) // 3) % 4
        
        # Holiday indicators (simplified)
        df['is_holiday_season'] = ((df['month'] == 12) & (df['day'] >= 20) |
                                   (df['month'] == 11) & (df['day'] >= 20) |
                                   (df['month'] == 7) & (df['day'] == 4)).astype(int)
        
        # Lag features (previous days' sales)
        for lag in [1, 2, 3, 7, 14]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window).std()
        
        # Drop rows with NaN values (from lag features)
        df = df.dropna()
        
        self.df = df
        
        print(f"Created {len(df.columns) - 2} features")  # -2 for date and sales
        print(f"Dataset shape after feature engineering: {df.shape}")
        
        return df
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for training"""
        print("\nPreparing data for training...")
        
        # Features to use (exclude date and target)
        feature_cols = [col for col in self.df.columns if col not in ['date', 'sales']]
        self.feature_names = feature_cols
        
        X = self.df[feature_cols]
        y = self.df['sales']
        
        # Split data (time series - no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} days")
        print(f"Test set: {X_test.shape[0]} days")
        print(f"Number of features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare"""
        print("\n" + "="*60)
        print("TRAINING FORECASTING MODELS")
        print("="*60)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': y_pred_test
            }
            
            print(f"  Train MAE: ${train_mae:.2f}")
            print(f"  Test MAE: ${test_mae:.2f}")
            print(f"  Test RMSE: ${test_rmse:.2f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        # Select best model based on test MAE
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        self.model = results[best_model_name]['model']
        
        print(f"\n✓ Best model: {best_model_name}")
        print(f"  Selected based on lowest test MAE: ${results[best_model_name]['test_mae']:.2f}")
        
        return results, best_model_name
    
    def forecast_next_day(self, date_str=None):
        """Forecast sales for the next day"""
        print("\n" + "="*60)
        print("NEXT DAY SALES FORECAST")
        print("="*60)
        
        # Get last date in dataset
        last_date = self.df['date'].max()
        next_date = last_date + timedelta(days=1)
        
        if date_str:
            next_date = pd.to_datetime(date_str)
        
        print(f"\nForecasting sales for: {next_date.strftime('%Y-%m-%d (%A)')}")
        
        # Create features for next day
        features = {
            'year': next_date.year,
            'month': next_date.month,
            'day': next_date.day,
            'day_of_week': next_date.dayofweek,
            'day_of_year': next_date.dayofyear,
            'week_of_year': next_date.isocalendar()[1],
            'quarter': next_date.quarter,
            'is_weekend': int(next_date.dayofweek >= 5),
            'is_month_start': int(next_date.is_month_start),
            'is_month_end': int(next_date.is_month_end),
            'season': ((next_date.month % 12 + 3) // 3) % 4,
            'is_holiday_season': int((next_date.month == 12 and next_date.day >= 20) or
                                    (next_date.month == 11 and next_date.day >= 20) or
                                    (next_date.month == 7 and next_date.day == 4))
        }
        
        # Get lag features from recent data
        recent_sales = self.df['sales'].tail(30).values
        for lag in [1, 2, 3, 7, 14]:
            if lag <= len(recent_sales):
                features[f'sales_lag_{lag}'] = recent_sales[-lag]
            else:
                features[f'sales_lag_{lag}'] = recent_sales[0]
        
        # Rolling statistics
        for window in [7, 14, 30]:
            if window <= len(recent_sales):
                features[f'sales_rolling_mean_{window}'] = recent_sales[-window:].mean()
                features[f'sales_rolling_std_{window}'] = recent_sales[-window:].std()
            else:
                features[f'sales_rolling_mean_{window}'] = recent_sales.mean()
                features[f'sales_rolling_std_{window}'] = recent_sales.std()
        
        # Create feature vector
        feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        # Display results
        print(f"\n📊 FORECAST RESULTS:")
        print(f"   Predicted Sales: ${prediction:.2f}")
        print(f"\n📅 Context:")
        print(f"   Day of Week: {next_date.strftime('%A')}")
        print(f"   Weekend: {'Yes' if features['is_weekend'] else 'No'}")
        print(f"   Holiday Season: {'Yes' if features['is_holiday_season'] else 'No'}")
        print(f"\n📈 Recent Performance:")
        print(f"   Yesterday's Sales: ${features['sales_lag_1']:.2f}")
        print(f"   7-Day Average: ${features['sales_rolling_mean_7']:.2f}")
        print(f"   30-Day Average: ${features['sales_rolling_mean_30']:.2f}")
        
        return prediction


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("GROCERY SALES FORECASTING SYSTEM")
    print("Predict Next Day Sales Using Machine Learning")
    print("="*60)
    
    # Initialize system
    forecaster = GrocerySalesForecastSystem()
    
    # Step 1: Generate/Load data
    df = forecaster.generate_synthetic_data(days=365)
    
    # Step 2: Engineer features
    df = forecaster.engineer_features()
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(test_size=0.2)
    
    # Step 4: Train models
    results, best_model = forecaster.train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Forecast next day
    forecaster.forecast_next_day()
    
    print("\n" + "="*60)
    print("FORECASTING SYSTEM COMPLETE!")
    print("="*60)

