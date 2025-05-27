import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_generation import generate_synthetic_data
from src.model_training import preprocess_data, train_models, evaluate_models, save_models

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'visualizations']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def generate_visualizations(df, results):
    """Generate and save visualizations."""
    # Create visualizations directory if it doesn't exist
    vis_dir = Path('visualizations')
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numerical_cols = ['number_of_workers', 'previous_accidents_on_site']
    correlation = df[numerical_cols + ['accident_occurred']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(vis_dir / 'correlation_heatmap.png')
    plt.close()
    
    # 2. Accident distribution by location
    plt.figure(figsize=(12, 6))
    location_accidents = df.groupby('site_location')['accident_occurred'].mean()
    location_accidents.sort_values(ascending=False).plot(kind='bar')
    plt.title('Accident Rate by Location')
    plt.xlabel('Location')
    plt.ylabel('Accident Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / 'accident_rate_by_location.png')
    plt.close()
    
    # 3. Safety gear compliance impact
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='safety_gear_compliance', hue='accident_occurred')
    plt.title('Accident Occurrence by Safety Gear Compliance')
    plt.tight_layout()
    plt.savefig(vis_dir / 'safety_gear_impact.png')
    plt.close()
    
    # 4. Model comparison
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_comparison = pd.DataFrame({
        name: [results[name][metric] for metric in metrics]
        for name in results.keys()
    }, index=metrics)
    
    model_comparison.plot(kind='bar', rot=0)
    plt.title('Model Performance Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(vis_dir / 'model_comparison.png')
    plt.close()

def main():
    """Run the complete machine learning pipeline."""
    print("Starting the construction site safety prediction pipeline...\n")
    
    # Create necessary directories
    create_directories()
    
    # 1. Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=1000)
    df.to_csv('data/construction_safety_data.csv', index=False)
    print(f"Generated {len(df)} samples of synthetic data\n")
    
    # 2. Preprocess data
    print("Preprocessing data...")
    (X_train, X_test, y_train, y_test), preprocessor = preprocess_data(df)
    print("Data preprocessing completed\n")
    
    # 3. Train models
    print("Training models...")
    models = train_models(X_train, y_train)
    print("Model training completed\n")
    
    # 4. Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    print("Model evaluation completed\n")
    
    # 5. Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(df, results)
    print("Visualizations saved in 'visualizations' directory\n")
    
    # 6. Save the best model
    print("Saving models...")
    save_models(models, preprocessor)
    print("Models saved in 'models' directory\n")
    
    print("Pipeline completed successfully!")
    print("\nYou can now:")
    print("1. Check the visualizations in the 'visualizations' directory")
    print("2. Use src/predict.py to make predictions on new data")
    print("3. Review model performance metrics above")

if __name__ == '__main__':
    main() 