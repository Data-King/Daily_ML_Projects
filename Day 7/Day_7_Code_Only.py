"""
Technical Indicator Signal Classifier
Use RSI, MACD, SMA to classify "buy", "sell", "hold" signals
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicatorClassifier:
    """
    A trading signal classifier using technical indicators
    RSI, MACD, and SMA to predict buy/sell/hold signals
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.df = None
        
    def generate_synthetic_price_data(self, days=1000, start_price=100):
        """
        Generate synthetic stock price data
        In practice, you would load real stock data from yfinance or similar
        """
        print("Generating synthetic stock price data...")
        
        np.random.seed(42)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        prices = [start_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        prices = np.array(prices[1:])
        
        # Create date range
        dates = pd.date_range(start='2021-01-01', periods=days, freq='D')
        
        # Add volume
        volumes = np.random.randint(1000000, 10000000, days)
        
        self.df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        # Generate OHLC data
        self.df['open'] = self.df['close'] * (1 + np.random.uniform(-0.01, 0.01, days))
        self.df['high'] = self.df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, days))
        self.df['low'] = self.df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, days))
        
        print(f"Generated {len(self.df)} days of price data")
        print(f"Price range: ${self.df['close'].min():.2f} - ${self.df['close'].max():.2f}")
        print(f"Average price: ${self.df['close'].mean():.2f}")
        
        return self.df
    
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average (SMA)"""
        return data.rolling(window=period).mean()
    
    def calculate_technical_indicators(self):
        """Calculate all technical indicators"""
        print("\nCalculating technical indicators...")
        
        df = self.df.copy()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], period=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df['close'])
        
        # Simple Moving Averages
        df['sma_20'] = self.calculate_sma(df['close'], 20)
        df['sma_50'] = self.calculate_sma(df['close'], 50)
        df['sma_200'] = self.calculate_sma(df['close'], 200)
        
        # Price relative to SMAs
        df['price_to_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
        df['price_to_sma50'] = (df['close'] / df['sma_50'] - 1) * 100
        df['price_to_sma200'] = (df['close'] / df['sma_200'] - 1) * 100
        
        # SMA crossovers
        df['sma20_above_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma50_above_sma200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        print(f"Calculated {len([col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
        
        self.df = df
        return df
    
    def generate_trading_signals(self):
        """
        Generate trading signals based on technical indicators
        This creates our target labels: BUY (2), HOLD (1), SELL (0)
        """
        print("\nGenerating trading signals...")
        
        df = self.df.copy()
        
        # Initialize signal
        signals = []
        
        for i in range(len(df)):
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            if pd.notna(df['rsi'].iloc[i]):
                if df['rsi'].iloc[i] < 30:  # Oversold - buy signal
                    buy_score += 2
                elif df['rsi'].iloc[i] > 70:  # Overbought - sell signal
                    sell_score += 2
            
            # MACD signals
            if pd.notna(df['macd'].iloc[i]) and pd.notna(df['macd_signal'].iloc[i]):
                if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                    buy_score += 1
                else:
                    sell_score += 1
            
            # SMA signals
            if pd.notna(df['sma_20'].iloc[i]) and pd.notna(df['sma_50'].iloc[i]):
                if df['close'].iloc[i] > df['sma_20'].iloc[i] > df['sma_50'].iloc[i]:
                    buy_score += 1
                elif df['close'].iloc[i] < df['sma_20'].iloc[i] < df['sma_50'].iloc[i]:
                    sell_score += 1
            
            # Momentum signals
            if pd.notna(df['momentum_10'].iloc[i]):
                if df['momentum_10'].iloc[i] > 3:
                    buy_score += 1
                elif df['momentum_10'].iloc[i] < -3:
                    sell_score += 1
            
            # Bollinger Band signals
            if pd.notna(df['bb_position'].iloc[i]):
                if df['bb_position'].iloc[i] < 0.2:  # Near lower band
                    buy_score += 1
                elif df['bb_position'].iloc[i] > 0.8:  # Near upper band
                    sell_score += 1
            
            # Determine final signal
            if buy_score > sell_score and buy_score >= 3:
                signals.append(2)  # BUY
            elif sell_score > buy_score and sell_score >= 3:
                signals.append(0)  # SELL
            else:
                signals.append(1)  # HOLD
        
        df['signal'] = signals
        
        # Distribution of signals
        signal_counts = df['signal'].value_counts().sort_index()
        print(f"\nSignal distribution:")
        print(f"  SELL (0): {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  HOLD (1): {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  BUY (2): {signal_counts.get(2, 0)} ({signal_counts.get(2, 0)/len(df)*100:.1f}%)")
        
        self.df = df
        return df
    
    def prepare_data(self, test_size=0.2):
        """Prepare data for training"""
        print("\nPreparing data for training...")
        
        # Remove rows with NaN values
        df_clean = self.df.dropna()
        
        # Feature columns (all technical indicators)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'signal']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df_clean[feature_cols]
        y = df_clean['signal']
        
        # Split data (time series - no shuffle for realistic evaluation)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("TRAINING SIGNAL CLASSIFICATION MODELS")
        print("="*60)
        
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                    random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                            random_state=42)
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
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'predictions': y_pred_test
            }
            
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_acc'])
        self.model = results[best_model_name]['model']
        
        print(f"\n✓ Best model: {best_model_name}")
        print(f"  Test Accuracy: {results[best_model_name]['test_acc']:.4f}")
        
        return results, best_model_name
    
    def evaluate_model(self, y_test, predictions):
        """Evaluate model with detailed metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                   target_names=['SELL', 'HOLD', 'BUY'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("              SELL  HOLD   BUY")
        print(f"True SELL    {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}")
        print(f"True HOLD    {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}")
        print(f"True BUY     {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}")
    
    def get_feature_importance(self):
        """Get feature importance as text output"""
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE")
            print("="*60)
            
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]  # Sort descending
            
            print("\nTop 15 Most Important Technical Indicators:")
            print("-" * 60)
            for i in range(min(15, len(indices))):
                idx = indices[i]
                print(f"{i+1:2d}. {self.feature_names[idx]:25s} {importances[idx]:.4f}")
    
    def predict_signal(self, latest_data=None):
        """
        Predict trading signal for current market conditions
        
        Parameters:
        -----------
        latest_data : dict (optional)
            Dictionary with latest indicator values
        """
        print("\n" + "="*60)
        print("CURRENT TRADING SIGNAL PREDICTION")
        print("="*60)
        
        if latest_data is None:
            # Use last row from dataset
            latest_data = self.df[self.feature_names].iloc[-1:].values
        else:
            # Convert dict to array
            latest_data = np.array([latest_data[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale
        latest_scaled = self.scaler.transform(latest_data)
        
        # Predict
        prediction = self.model.predict(latest_scaled)[0]
        probabilities = self.model.predict_proba(latest_scaled)[0]
        
        signal_names = ['SELL', 'HOLD', 'BUY']
        
        # Display latest indicators
        print("\n📊 CURRENT MARKET INDICATORS:")
        latest_row = self.df.iloc[-1]
        print(f"  Price: ${latest_row['close']:.2f}")
        print(f"  RSI: {latest_row['rsi']:.2f}")
        print(f"  MACD: {latest_row['macd']:.4f}")
        print(f"  MACD Signal: {latest_row['macd_signal']:.4f}")
        print(f"  SMA 20: ${latest_row['sma_20']:.2f}")
        print(f"  SMA 50: ${latest_row['sma_50']:.2f}")
        
        print(f"\n🎯 PREDICTION: {signal_names[prediction]}")
        print(f"   Confidence: {probabilities[prediction]:.2%}")
        
        print(f"\n📈 PROBABILITY BREAKDOWN:")
        for i, name in enumerate(signal_names):
            print(f"   {name}: {probabilities[i]:.2%}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if prediction == 2:
            print("   ✓ Strong buy signal detected!")
            print("   Consider entering a long position.")
        elif prediction == 0:
            print("   ⚠ Strong sell signal detected!")
            print("   Consider exiting positions or shorting.")
        else:
            print("   → Hold current positions.")
            print("   Wait for clearer signals before acting.")
        
        return prediction, probabilities


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("TECHNICAL INDICATOR SIGNAL CLASSIFIER")
    print("Using RSI, MACD, SMA for Buy/Sell/Hold Signals")
    print("="*60)
    
    # Initialize system
    classifier = TechnicalIndicatorClassifier()
    
    # Step 1: Generate/Load price data
    df = classifier.generate_synthetic_price_data(days=1000)
    
    # Step 2: Calculate technical indicators
    df = classifier.calculate_technical_indicators()
    
    # Step 3: Generate trading signals (labels)
    df = classifier.generate_trading_signals()
    
    # Step 4: Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(test_size=0.2)
    
    # Step 5: Train models
    results, best_model = classifier.train_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Evaluate best model
    classifier.evaluate_model(y_test, results[best_model]['predictions'])
    
    # Step 7: Feature importance
    classifier.get_feature_importance()
    
    # Step 8: Predict current signal
    classifier.predict_signal()
    
    print("\n" + "="*60)
    print("CLASSIFICATION SYSTEM COMPLETE!")
    print("="*60)

