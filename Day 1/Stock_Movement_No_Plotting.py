"""
Daily Stock Movement Classifier (Up/Down)
Binary classification using OHLCV features
Step-by-step implementation with sample data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')




print("="*70)
print("DAILY STOCK MOVEMENT CLASSIFIER PROJECT")
print("Predict if stock will go UP or DOWN ")
print("="*70)

# ===============================================================
# STEP 1: GENERATE SAMPLE OHLCV DATA
# ===============================================================

print("\n[STEP 1] Generating Sample STock Market Data (OHLCV)...")

np.random.seed(42)
n_days = 1000

# Generate realistic stock price data

dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
initial_price = 100

# Simulate price movements with trend and volatility
returns = np.random.normal(0.0005, 0.02, n_days)
trend = np.linspace(0, 0.03, n_days)
returns = returns + trend / n_days