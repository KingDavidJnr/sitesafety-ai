import pandas as pd
import joblib
from pathlib import Path

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor objects.
    
    Returns:
        tuple: (model, label_encoders, scaler)
    """
    model_path = Path('models')
    
    try:
        model = joblib.load(model_path / 'random_forest_model.pkl')
        le_dict = joblib.load(model_path / 'label_encoders.pkl')
        scaler = joblib.load(model_path / 'scaler.pkl')
        return model, le_dict, scaler
    except FileNotFoundError:
        raise FileNotFoundError("Model files not found. Please train the model first.")

def preprocess_input(data, le_dict, scaler):
    """
    Preprocess input data for prediction.
    
    Args:
        data (pandas.DataFrame): Input data
        le_dict (dict): Label encoders
        scaler (StandardScaler): Feature scaler
        
    Returns:
        pandas.DataFrame: Preprocessed data
    """
    df = data.copy()
    
    # Convert date to numerical features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df.drop('date', axis=1, inplace=True)
    
    # Apply label encoding to ordinal features
    ordinal_features = ['worker_experience_level', 'time_of_day']
    for feature in ordinal_features:
        df[feature] = le_dict[feature].transform(df[feature])
    
    # Apply one-hot encoding to nominal features
    nominal_features = ['weather_condition', 'task_type', 'equipment_used', 'site_location']
    df = pd.get_dummies(df, columns=nominal_features)
    
    # Convert binary features to numeric
    df['supervisor_present'] = (df['supervisor_present'] == 'Yes').astype(int)
    df['safety_gear_compliance'] = (df['safety_gear_compliance'] == 'Yes').astype(int)
    
    # Scale numerical features
    numerical_features = ['number_of_workers', 'previous_accidents_on_site', 'month', 'day_of_week']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    return df

def predict_accident_probability(input_data):
    """
    Predict accident probability for new construction site data.
    
    Args:
        input_data (pandas.DataFrame): Input data with required features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    # Load model and preprocessor
    model, le_dict, scaler = load_model_and_preprocessor()
    
    # Preprocess input data
    processed_data = preprocess_input(input_data, le_dict, scaler)
    
    # Make predictions
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    return predictions, probabilities

if __name__ == '__main__':
    # Example usage
    sample_data = pd.DataFrame({
        'date': ['2024-02-15'],
        'time_of_day': ['Morning'],
        'weather_condition': ['Sunny'],
        'task_type': ['Welding'],
        'number_of_workers': [10],
        'supervisor_present': ['Yes'],
        'safety_gear_compliance': ['Yes'],
        'worker_experience_level': ['Expert'],
        'equipment_used': ['Scaffold'],
        'site_location': ['Lagos Mainland'],
        'previous_accidents_on_site': [0]
    })
    
    # Convert date to datetime
    sample_data['date'] = pd.to_datetime(sample_data['date'])
    
    try:
        predictions, probabilities = predict_accident_probability(sample_data)
        
        print("\nPrediction Results:")
        print(f"Accident Predicted: {'Yes' if predictions[0] == 1 else 'No'}")
        print(f"Accident Probability: {probabilities[0]:.2%}")
        
        # Print risk assessment
        if probabilities[0] < 0.2:
            risk_level = "Low"
        elif probabilities[0] < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        print(f"Risk Level: {risk_level}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}") 