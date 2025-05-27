import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path

def preprocess_data(df):
    """
    Preprocess the construction safety data for model training.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: Processed data (X_train, X_test, y_train, y_test), preprocessor objects
    """
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Convert date to numerical features
    df_processed['month'] = df_processed['date'].dt.month
    df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
    df_processed.drop('date', axis=1, inplace=True)
    
    # Initialize label encoders for ordinal variables
    le_dict = {}
    ordinal_features = ['worker_experience_level', 'time_of_day']
    
    # Apply label encoding to ordinal features
    for feature in ordinal_features:
        le_dict[feature] = LabelEncoder()
        df_processed[feature] = le_dict[feature].fit_transform(df_processed[feature])
    
    # Apply one-hot encoding to nominal features
    nominal_features = ['weather_condition', 'task_type', 'equipment_used', 'site_location']
    df_processed = pd.get_dummies(df_processed, columns=nominal_features)
    
    # Convert binary features to numeric
    df_processed['supervisor_present'] = (df_processed['supervisor_present'] == 'Yes').astype(int)
    df_processed['safety_gear_compliance'] = (df_processed['safety_gear_compliance'] == 'Yes').astype(int)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['number_of_workers', 'previous_accidents_on_site', 'month', 'day_of_week']
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    # Split features and target
    X = df_processed.drop('accident_occurred', axis=1)
    y = df_processed['accident_occurred']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (X_train, X_test, y_train, y_test), (le_dict, scaler)

def train_models(X_train, y_train):
    """
    Train multiple models on the preprocessed data.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        dict: Trained models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models and print performance metrics.
    
    Args:
        models (dict): Trained models
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        dict: Model evaluation results
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': model
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"F1-score: {results[name]['f1']:.4f}")
        print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    return results

def save_models(models, preprocessor, best_model_name='Random Forest'):
    """
    Save the trained models and preprocessor objects.
    
    Args:
        models (dict): Trained models
        preprocessor (tuple): Preprocessor objects (le_dict, scaler)
        best_model_name (str): Name of the best performing model to save
    """
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Save the best model
    joblib.dump(models[best_model_name], 'models/random_forest_model.pkl')
    
    # Save preprocessor objects
    le_dict, scaler = preprocessor
    joblib.dump(le_dict, 'models/label_encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/construction_safety_data.csv', parse_dates=['date'])
    except FileNotFoundError:
        print("Please generate the dataset first using data_generation.py")
        exit(1)
    
    # Preprocess data
    (X_train, X_test, y_train, y_test), preprocessor = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Save the best model and preprocessor
    save_models(models, preprocessor) 