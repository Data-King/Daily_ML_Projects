import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic house data for demonstration"""
        np.random.seed(42)

        # Generate features
        size_sqft = np.random.randint(800, 5000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age_years = np.random.randint(0, 50, n_samples)

        # Location categories
        locations = np.random.choice(["Downtown", "Suburb", "Rural", "Waterfront"], n_samples)

        # Generate prices based on features (with some noise)
        base_price = 50000
        price = (
            base_price
            + size_sqft * 150  # $150 per sqft
            + bedrooms * 20000  # $20k per bedroom
            + bathrooms * 15000  # $15k per bathroom
            + (50 - age_years) * 2000  # Newer homes worth more
        )

        # Location multipliers
        location_multipliers = {
            "Downtown": 1.4,
            "Waterfront": 1.5,
            "Suburb": 1.0,
            "Rural": 0.8,
        }

        price = price * [location_multipliers[loc] for loc in locations]

        # Add random noise
        price = price + np.random.normal(0, 50000, n_samples)
        price = np.maximum(price, 50000)  # Ensure positive prices

        # Create DataFrame
        df = pd.DataFrame(
            {
                "size_sqft": size_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age_years": age_years,
                "location": locations,
                "price": price,
            }
        )

        return df

    def prepare_data(self, df):
        """Prepare data for training"""
        # Encode categorical variables
        df_encoded = df.copy()

        if "location" in df_encoded.columns:
            if "location" not in self.label_encoders:
                self.label_encoders["location"] = LabelEncoder()
                df_encoded["location"] = self.label_encoders["location"].fit_transform(
                    df_encoded["location"]
                )
            else:
                df_encoded["location"] = self.label_encoders["location"].transform(
                    df_encoded["location"]
                )

        return df_encoded

    def train(self, df, model_type="random_forest"):
        """Train the model"""
        # Prepare data
        df_encoded = self.prepare_data(df)

        # Split features and target
        X = df_encoded.drop("price", axis=1)
        y = df_encoded["price"]

        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        metrics = {
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        }

        return metrics, X_test, y_test, test_pred

    def predict(self, size_sqft, bedrooms, bathrooms, age_years, location):
        """Predict price for a single house"""
        # Create DataFrame for prediction
        input_data = pd.DataFrame(
            {
                "size_sqft": [size_sqft],
                "bedrooms": [bedrooms],
                "bathrooms": [bathrooms],
                "age_years": [age_years],
                "location": [location],
            }
        )

        # Prepare and scale
        input_encoded = self.prepare_data(input_data)
        input_scaled = self.scaler.transform(input_encoded)

        # Predict
        prediction = self.model.predict(input_scaled)[0]
        return prediction


def main():
    print("=" * 60)
    print("HOUSE PRICE PREDICTOR")
    print("=" * 60)

    # Initialize predictor
    predictor = HousePricePredictor()

    # Generate sample data
    print("\n📊 Generating sample house data...")
    df = predictor.generate_sample_data(n_samples=1000)

    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\n📈 Dataset statistics:")
    print(df.describe())

    # Train model
    print("\n🤖 Training Random Forest model...")
    metrics, X_test, y_test, test_pred = predictor.train(df, model_type="random_forest")

    print("\n✅ Model Performance:")
    print(f"  Training R² Score: {metrics['train_r2']:.4f}")
    print(f"  Testing R² Score:  {metrics['test_r2']:.4f}")
    print(f"  Training MAE:      ${metrics['train_mae']:,.2f}")
    print(f"  Testing MAE:       ${metrics['test_mae']:,.2f}")
    print(f"  Training RMSE:     ${metrics['train_rmse']:,.2f}")
    print(f"  Testing RMSE:      ${metrics['test_rmse']:,.2f}")

    # Make sample predictions
    print("\n🏠 Sample Predictions:")
    print("-" * 60)

    test_houses = [
        (2500, 3, 2, 10, "Downtown"),
        (1800, 2, 2, 5, "Suburb"),
        (3500, 4, 3, 15, "Waterfront"),
        (1200, 2, 1, 30, "Rural"),
    ]

    for size, beds, baths, age, loc in test_houses:
        price = predictor.predict(size, beds, baths, age, loc)
        print(f"{size} sqft, {beds} bed, {baths} bath, {age} years old, {loc}")
        print(f"  Predicted Price: ${price:,.2f}\n")

    # Text-only summary (no visualizations)
    if hasattr(predictor.model, "feature_importances_") and predictor.feature_names:
        importances = predictor.model.feature_importances_
        fi = (
            pd.DataFrame({"feature": predictor.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        print("📌 Top feature importances (Random Forest):")
        print(fi.head(10).to_string(index=False))

    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()


