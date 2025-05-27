import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic construction site safety data.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define possible values for categorical features
    time_periods = ['Morning', 'Afternoon', 'Night']
    weather_conditions = ['Sunny', 'Rainy', 'Windy', 'Cloudy']
    task_types = ['Roofing', 'Welding', 'Excavation', 'Concrete Pouring', 'Electrical']
    experience_levels = ['Beginner', 'Intermediate', 'Expert']
    equipment_types = ['Scaffold', 'Crane', 'Ladder', 'Forklift']
    locations = ['Lagos Mainland', 'Lekki', 'Ikeja', 'Surulere', 'Epe']
    
    # Generate random dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) 
            for _ in range(n_samples)]
    
    # Generate other features
    data = {
        'date': dates,
        'time_of_day': [random.choice(time_periods) for _ in range(n_samples)],
        'weather_condition': [random.choice(weather_conditions) for _ in range(n_samples)],
        'task_type': [random.choice(task_types) for _ in range(n_samples)],
        'number_of_workers': [random.randint(2, 50) for _ in range(n_samples)],
        'supervisor_present': [random.choice(['Yes', 'No']) for _ in range(n_samples)],
        'safety_gear_compliance': [random.choice(['Yes', 'No']) for _ in range(n_samples)],
        'worker_experience_level': [random.choice(experience_levels) for _ in range(n_samples)],
        'equipment_used': [random.choice(equipment_types) for _ in range(n_samples)],
        'site_location': [random.choice(locations) for _ in range(n_samples)],
        'previous_accidents_on_site': [random.randint(0, 5) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate accident probability based on risk factors
    def calculate_accident_probability(row):
        base_probability = 0.05  # Base 5% chance of accident
        
        # Weather conditions
        if row['weather_condition'] in ['Rainy', 'Windy']:
            base_probability += 0.1
            
        # Supervision
        if row['supervisor_present'] == 'No':
            base_probability += 0.15
            
        # Safety compliance
        if row['safety_gear_compliance'] == 'No':
            base_probability += 0.2
            
        # Worker experience
        if row['worker_experience_level'] == 'Beginner':
            base_probability += 0.1
        elif row['worker_experience_level'] == 'Intermediate':
            base_probability += 0.05
            
        # Previous accidents increase risk
        base_probability += row['previous_accidents_on_site'] * 0.03
        
        # Time of day (night work is riskier)
        if row['time_of_day'] == 'Night':
            base_probability += 0.08
        
        # Equipment risk factors
        if row['equipment_used'] in ['Crane', 'Scaffold']:
            base_probability += 0.05
            
        # Task type risk factors
        if row['task_type'] in ['Roofing', 'Welding']:
            base_probability += 0.07
            
        # Cap probability at 95%
        return min(base_probability, 0.95)
    
    # Generate accident outcomes based on calculated probabilities
    probabilities = df.apply(calculate_accident_probability, axis=1)
    df['accident_occurred'] = [1 if random.random() < p else 0 for p in probabilities]
    
    return df

if __name__ == '__main__':
    # Generate sample dataset
    df = generate_synthetic_data(1000)
    
    # Save to CSV file
    df.to_csv('data/construction_safety_data.csv', index=False)
    print(f"Generated {len(df)} samples of synthetic data") 