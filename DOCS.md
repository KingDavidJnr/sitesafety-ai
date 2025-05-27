# Technical Documentation: Construction Site Safety Prediction Model

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Data Generation](#data-generation)
3. [Model Training](#model-training)
4. [Model Deployment](#model-deployment)
5. [Making Predictions](#making-predictions)
6. [Maintenance and Updates](#maintenance-and-updates)

## Model Architecture

### Overview
The project implements multiple machine learning models:
1. Logistic Regression (baseline model)
2. Random Forest Classifier (primary model)
3. XGBoost Classifier (optional advanced model)

### Feature Engineering
The model processes the following features:
- Temporal features (date, time_of_day)
- Environmental features (weather_condition)
- Operational features (task_type, number_of_workers)
- Safety features (supervisor_present, safety_gear_compliance)
- Worker-related features (worker_experience_level)
- Equipment features (equipment_used)
- Location features (site_location)
- Historical features (previous_accidents_on_site)

## Data Generation

### Synthetic Data Creation
The synthetic dataset is generated using realistic assumptions and domain knowledge:

1. **Date Generation**:
   - Random dates between 2022-2024
   - Weighted distribution for seasonal patterns

2. **Weather Conditions**:
   - Based on Lagos climate patterns
   - Correlation with accident probability

3. **Worker and Safety Parameters**:
   - Realistic worker counts (2-50)
   - Experience levels distribution
   - Safety compliance patterns

4. **Accident Probability Logic**:
```python
def calculate_accident_probability(row):
    base_probability = 0.05  # 5% base chance
    
    # Increase probability based on risk factors
    if row['weather_condition'] in ['Rainy', 'Windy']:
        base_probability += 0.1
    if row['supervisor_present'] == 'No':
        base_probability += 0.15
    if row['safety_gear_compliance'] == 'No':
        base_probability += 0.2
    if row['worker_experience_level'] == 'Beginner':
        base_probability += 0.1
        
    return min(base_probability, 0.95)  # Cap at 95%
```

## Model Training

### Data Preprocessing
1. **Categorical Encoding**:
   - One-hot encoding for nominal variables
   - Label encoding for ordinal variables

2. **Feature Scaling**:
   - StandardScaler for numerical features
   - MinMaxScaler for bounded features

3. **Data Split**:
   - 80% training data
   - 20% testing data
   - Stratified split based on target variable

### Training Process
```python
# Example training code
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(
    rf_model, 
    param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)

# Train model
grid_search.fit(X_train, y_train)
```

## Model Deployment

### Local Deployment
1. Clone the repository
2. Install dependencies
3. Load the trained model:
```python
import joblib

# Load the model
model = joblib.load('models/random_forest_model.pkl')

# Make predictions
predictions = model.predict(X_new)
```

### Production Deployment (Future Work)
- Docker containerization
- REST API implementation
- Cloud deployment options (AWS, GCP, Azure)

## Making Predictions

### Input Format
The model expects a pandas DataFrame with the following columns:
```python
required_columns = [
    'date', 'time_of_day', 'weather_condition',
    'task_type', 'number_of_workers', 'supervisor_present',
    'safety_gear_compliance', 'worker_experience_level',
    'equipment_used', 'site_location',
    'previous_accidents_on_site'
]
```

### Example Usage
```python
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load('models/random_forest_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Prepare input data
input_data = pd.DataFrame({
    'date': ['2024-02-15'],
    'time_of_day': ['Morning'],
    'weather_condition': ['Sunny'],
    # ... other features ...
})

# Preprocess input
X_processed = preprocessor.transform(input_data)

# Get prediction
prediction = model.predict(X_processed)
probability = model.predict_proba(X_processed)
```

## Maintenance and Updates

### Model Retraining
- Retrain the model quarterly when real data becomes available
- Update feature engineering based on new insights
- Validate model performance against baseline

### Performance Monitoring
- Track key metrics:
  - Model accuracy
  - False positive/negative rates
  - Feature importance stability
  - Prediction latency

### Version Control
- Model versions are tracked using semantic versioning
- Each release includes:
  - Trained model weights
  - Preprocessing pipeline
  - Performance metrics
  - Training data summary

## Contributing
Please refer to CONTRIBUTING.md for guidelines on:
- Code style
- Testing requirements
- Pull request process
- Documentation standards 