"""
Medical Diagnosis Explanation System
Uses XGBoost for prediction and SHAP for interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class MedicalDiagnosisSystem:
    """
    A medical diagnosis system that uses XGBoost for prediction
    and SHAP for explaining clinical decisions.
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the medical dataset (using breast cancer dataset as example)"""
        print("Loading medical dataset...")
        data = load_breast_cancer()
        
        # Create DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target
        
        self.feature_names = data.feature_names
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"\nTarget distribution:")
        print(df['diagnosis'].value_counts())
        
        return df
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\nPreparing data...")
        
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, params=None):
        """Train XGBoost model"""
        print("\nTraining XGBoost model...")
        
        if params is None:
            params = {
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': 42,
                'eval_metric': 'logloss'
            }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance with detailed metrics"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        y_pred = self.model.predict(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Malignant', 'Benign']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
    def initialize_explainer(self):
        """Initialize SHAP explainer for model interpretability"""
        print("\nInitializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("SHAP explainer ready!")
        
    def explain_global(self):
        """Generate global feature importance explanation"""
        print("\n" + "="*50)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*50)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(self.X_test)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, 
                         feature_names=self.feature_names,
                         show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        print("\nSHAP summary plot saved as 'shap_summary.png'")
        plt.show()
        
        # Bar plot of mean absolute SHAP values
        plt.figure()
        shap.summary_plot(shap_values, self.X_test,
                         feature_names=self.feature_names,
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved as 'shap_importance.png'")
        plt.show()
        
    def explain_prediction(self, patient_idx=0):
        """
        Explain a specific prediction for interpretability
        
        Parameters:
        -----------
        patient_idx : int
            Index of the patient in the test set
        """
        print("\n" + "="*50)
        print(f"INDIVIDUAL PREDICTION EXPLANATION (Patient {patient_idx})")
        print("="*50)
        
        # Get patient data
        patient_data = self.X_test.iloc[patient_idx:patient_idx+1]
        true_label = self.y_test.iloc[patient_idx]
        
        # Make prediction
        prediction = self.model.predict(patient_data)[0]
        probability = self.model.predict_proba(patient_data)[0]
        
        print(f"\nPatient Data Summary:")
        print(f"True Diagnosis: {'Benign' if true_label == 1 else 'Malignant'}")
        print(f"Predicted Diagnosis: {'Benign' if prediction == 1 else 'Malignant'}")
        print(f"Prediction Confidence: {max(probability):.2%}")
        print(f"  - Malignant probability: {probability[0]:.2%}")
        print(f"  - Benign probability: {probability[1]:.2%}")
        
        # Calculate SHAP values for this patient
        shap_values = self.explainer.shap_values(patient_data)
        
        # Waterfall plot
        print("\nGenerating explanation visualizations...")
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=patient_data.values[0],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_patient_{patient_idx}.png', dpi=300, bbox_inches='tight')
        print(f"Waterfall plot saved as 'shap_waterfall_patient_{patient_idx}.png'")
        plt.show()
        
        # Force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            patient_data.values[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'shap_force_patient_{patient_idx}.png', dpi=300, bbox_inches='tight')
        print(f"Force plot saved as 'shap_force_patient_{patient_idx}.png'")
        plt.show()
        
        # Top contributing features
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[-5:][::-1]
        
        print("\nTop 5 Contributing Features:")
        for i, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx]
            feature_value = patient_data.values[0][idx]
            shap_value = shap_values[0][idx]
            direction = "towards Benign" if shap_value > 0 else "towards Malignant"
            print(f"{i}. {feature_name}: {feature_value:.2f}")
            print(f"   Impact: {abs(shap_value):.4f} {direction}")
        
    def explain_multiple_predictions(self, n_patients=3):
        """Explain predictions for multiple patients"""
        print("\n" + "="*50)
        print(f"EXPLAINING {n_patients} PATIENT PREDICTIONS")
        print("="*50)
        
        for i in range(min(n_patients, len(self.X_test))):
            self.explain_prediction(patient_idx=i)
            print("\n" + "-"*50 + "\n")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("MEDICAL DIAGNOSIS EXPLANATION SYSTEM")
    print("Using XGBoost + SHAP for Interpretable Clinical Decisions")
    print("="*60)
    
    # Initialize system
    system = MedicalDiagnosisSystem()
    
    # Step 1: Load data
    df = system.load_data()
    
    # Step 2: Prepare data
    system.prepare_data(df)
    
    # Step 3: Train model
    system.train_model()
    
    # Step 4: Evaluate model
    system.evaluate_model()
    
    # Step 5: Initialize SHAP explainer
    system.initialize_explainer()
    
    # Step 6: Global explanations
    system.explain_global()
    
    # Step 7: Individual patient explanations
    system.explain_prediction(patient_idx=0)
    system.explain_prediction(patient_idx=5)
    
    print("\n" + "="*60)
    print("SYSTEM COMPLETE!")
    print("All visualizations have been saved.")
    print("="*60)