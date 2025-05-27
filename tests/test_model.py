import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

class TestConstructionSafetyModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_input = pd.DataFrame({
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

    def test_data_generation(self):
        """Test if synthetic data generation works correctly"""
        try:
            from src.data_generation import generate_synthetic_data
            df = generate_synthetic_data(n_samples=100)
            
            # Check if all required columns are present
            required_columns = [
                'date', 'time_of_day', 'weather_condition', 'task_type',
                'number_of_workers', 'supervisor_present', 'safety_gear_compliance',
                'worker_experience_level', 'equipment_used', 'site_location',
                'previous_accidents_on_site', 'accident_occurred'
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns)
            
            # Check data types and value ranges
            self.assertTrue(all(isinstance(x, pd.Timestamp) for x in df['date']))
            self.assertTrue(all(df['number_of_workers'].between(2, 50)))
            self.assertTrue(all(df['previous_accidents_on_site'].between(0, 5)))
            self.assertTrue(all(df['accident_occurred'].isin([0, 1])))
            
        except ImportError:
            self.skipTest("Data generation module not implemented yet")

    def test_model_prediction(self):
        """Test if model can make predictions with correct format"""
        try:
            import joblib
            model = joblib.load('models/random_forest_model.pkl')
            preprocessor = joblib.load('models/preprocessor.pkl')
            
            # Test prediction
            processed_input = preprocessor.transform(self.sample_input)
            prediction = model.predict(processed_input)
            probabilities = model.predict_proba(processed_input)
            
            # Check prediction format
            self.assertIn(prediction[0], [0, 1])
            self.assertEqual(len(probabilities[0]), 2)
            self.assertTrue(all(0 <= p <= 1 for p in probabilities[0]))
            
        except FileNotFoundError:
            self.skipTest("Model files not found. Train the model first.")

if __name__ == '__main__':
    unittest.main() 