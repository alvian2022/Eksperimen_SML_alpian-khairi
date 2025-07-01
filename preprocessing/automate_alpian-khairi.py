"""
Automated Data Preprocessing Pipeline
Author: alpian_khairi_C1BO
Description: Automated preprocessing for Iris dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import logging
from typing import Tuple, Optional
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IrisDataPreprocessor:
    """
    Automated data preprocessing class for Iris dataset
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from file or sklearn
        
        Args:
            file_path: Path to CSV file (optional)
            
        Returns:
            DataFrame: Loaded dataset
        """
        try:
            if file_path and os.path.exists(file_path):
                logger.info(f"Loading data from {file_path}")
                df = pd.read_csv(file_path)
            else:
                logger.info("Loading data from sklearn datasets")
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['species'] = iris.target
                df['species_name'] = df['species'].map(self.target_mapping)
                
                # Save raw data
                df.to_csv('data/iris_raw.csv', index=False)
                logger.info("Raw data saved to data/iris_raw.csv")
            
            self.feature_names = [col for col in df.columns if col not in ['species', 'species_name', 'target']]
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame) -> dict:
        """
        Perform basic data exploration
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Exploration results
        """
        logger.info("Performing data exploration...")
        
        exploration_results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'target_distribution': df['species'].value_counts().to_dict() if 'species' in df.columns else {},
            'feature_stats': df[self.feature_names].describe().to_dict()
        }
        
        logger.info(f"Exploration completed:")
        logger.info(f"  - Shape: {exploration_results['shape']}")
        logger.info(f"  - Missing values: {sum(exploration_results['missing_values'].values())}")
        logger.info(f"  - Duplicates: {exploration_results['duplicates']}")
        
        return exploration_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned dataset
        """
        logger.info("Starting data cleaning...")
        
        original_shape = df.shape
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Found {missing_before} missing values")
            # For numerical columns, fill with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        # Remove duplicates
        duplicates_before = df.duplicated().sum()
        if duplicates_before > 0:
            logger.info(f"Found {duplicates_before} duplicate rows")
            df = df.drop_duplicates().reset_index(drop=True)
        
        # Check for outliers using IQR method
        outlier_info = self._detect_outliers(df)
        logger.info(f"Outlier detection completed: {outlier_info}")
        
        logger.info(f"Data cleaning completed. Shape: {original_shape} → {df.shape}")
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> dict:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Outlier information
        """
        outlier_info = {}
        
        for feature in self.feature_names:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                outlier_info[feature] = len(outliers)
        
        return outlier_info
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform feature engineering including scaling and encoding
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple: (Features DataFrame, Target Series)
        """
        logger.info("Starting feature engineering...")
        
        # Separate features and target
        X = df[self.feature_names].copy()
        
        # Handle target column
        if 'target' in df.columns:
            y = df['target']
        elif 'species' in df.columns:
            y = df['species']
        else:
            raise ValueError("No target column found (expected 'target' or 'species')")
        
        # Scale features
        logger.info("Scaling features using StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Encode target if needed
        if y.dtype == 'object' or y.dtype.name == 'category':
            logger.info("Encoding target labels...")
            y_encoded = self.label_encoder.fit_transform(y)
            y = pd.Series(y_encoded, name='target')
        
        # Add feature statistics
        feature_stats = {
            'original_ranges': {col: (X[col].min(), X[col].max()) for col in X.columns},
            'scaled_ranges': {col: (X_scaled_df[col].min(), X_scaled_df[col].max()) for col in X_scaled_df.columns}
        }
        
        logger.info("Feature engineering completed")
        logger.info(f"Feature ranges after scaling: {feature_stats['scaled_ranges']}")
        
        return X_scaled_df, y
    
    def validate_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Validate preprocessing quality with a quick model test
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            dict: Validation results
        """
        logger.info("Starting preprocessing validation...")
        
        try:
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train a simple model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report with proper target names
            # FIX: Convert numpy integers to strings for target_names
            target_names = [str(cls) for cls in self.label_encoder.classes_]
            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            validation_results = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'validation_successful': True
            }
            
            logger.info(f"Validation completed successfully. Accuracy: {accuracy:.4f}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'accuracy': 0.0,
                'classification_report': {},
                'validation_successful': False,
                'error': str(e)
            }
    
    def save_preprocessed_data(self, X: pd.DataFrame, y: pd.Series, output_path: str) -> None:
        """
        Save preprocessed data to file
        
        Args:
            X: Features DataFrame
            y: Target Series
            output_path: Output file path
        """
        try:
            # Combine features and target
            processed_df = X.copy()
            processed_df['target'] = y
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            processed_df.to_csv(output_path, index=False)
            
            # Save preprocessing objects for future use
            preprocessing_objects = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            
            objects_path = output_path.replace('.csv', '_objects.pkl')
            joblib.dump(preprocessing_objects, objects_path)
            
            logger.info(f"Preprocessed data saved to: {output_path}")
            logger.info(f"Preprocessing objects saved to: {objects_path}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise
    
    def preprocess_pipeline(self, 
                          input_path: Optional[str] = None, 
                          output_path: str = "preprocessing/iris_preprocessing.csv") -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            input_path: Input file path (optional, uses sklearn if None)
            output_path: Output file path
            
        Returns:
            DataFrame: Preprocessed dataset
        """
        logger.info("=== Starting Automated Preprocessing Pipeline ===")
        logger.info(f"Author: alpian_khairi_C1BO")
        
        try:
            # Step 1: Load data
            df = self.load_data(input_path)
            
            # Step 2: Explore data
            exploration_results = self.explore_data(df)
            
            # Step 3: Clean data
            df_clean = self.clean_data(df)
            
            # Step 4: Feature engineering
            X_processed, y_processed = self.engineer_features(df_clean)
            
            # Step 5: Validate preprocessing quality
            validation_results = self.validate_preprocessing(X_processed, y_processed)
            
            # Step 6: Save preprocessed data
            self.save_preprocessed_data(X_processed, y_processed, output_path)
            
            # Step 7: Create final dataset
            final_df = X_processed.copy()
            final_df['target'] = y_processed
            
            # Step 8: Generate summary report
            self._generate_summary_report(exploration_results, final_df, output_path, validation_results)
            
            logger.info("=== Preprocessing Pipeline Completed Successfully ===")
            return final_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _generate_summary_report(self, exploration_results: dict, final_df: pd.DataFrame, 
                               output_path: str, validation_results: dict) -> None:
        """
        Generate preprocessing summary report
        
        Args:
            exploration_results: Results from data exploration
            final_df: Final preprocessed DataFrame
            output_path: Output file path
            validation_results: Results from validation
        """
        report_path = output_path.replace('.csv', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("AUTOMATED PREPROCESSING REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Author: alpian_khairi_C1BO\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("ORIGINAL DATA:\n")
            f.write(f"  Shape: {exploration_results['shape']}\n")
            f.write(f"  Missing values: {sum(exploration_results['missing_values'].values())}\n")
            f.write(f"  Duplicates: {exploration_results['duplicates']}\n\n")
            
            f.write("PROCESSED DATA:\n")
            f.write(f"  Final shape: {final_df.shape}\n")
            f.write(f"  Features: {list(final_df.columns[:-1])}\n")
            f.write(f"  Target distribution: {final_df['target'].value_counts().to_dict()}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            if validation_results['validation_successful']:
                f.write(f"  Accuracy: {validation_results['accuracy']:.4f}\n")
                f.write(f"  Validation: PASSED\n\n")
            else:
                f.write(f"  Validation: FAILED\n")
                f.write(f"  Error: {validation_results.get('error', 'Unknown error')}\n\n")
            
            f.write("PREPROCESSING STEPS APPLIED:\n")
            f.write("  ✓ Data loading\n")
            f.write("  ✓ Missing value handling\n")
            f.write("  ✓ Duplicate removal\n")
            f.write("  ✓ Outlier detection\n")
            f.write("  ✓ Feature scaling (StandardScaler)\n")
            f.write("  ✓ Target encoding\n")
            f.write("  ✓ Data validation\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the preprocessing pipeline"""
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('preprocessing', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = IrisDataPreprocessor()
    
    # Run preprocessing pipeline
    try:
        # Check if raw data exists, otherwise use sklearn
        input_file = "data/iris_raw.csv" if os.path.exists("data/iris_raw.csv") else None
        
        processed_data = preprocessor.preprocess_pipeline(
            input_path=input_file,
            output_path="preprocessing/iris_preprocessing.csv"
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final dataset shape: {processed_data.shape}")
        print(f"Features: {list(processed_data.columns[:-1])}")
        print(f"Target classes: {sorted(processed_data['target'].unique())}")
        print(f"Output saved to: preprocessing/iris_preprocessing.csv")
        print("Files generated:")
        print("  - preprocessing/iris_preprocessing.csv")
        print("  - preprocessing/iris_preprocessing_objects.pkl")
        print("  - preprocessing/iris_preprocessing_report.txt")
        print("  - preprocessing.log")
        
    except Exception as e:
        print(f"ERROR: Preprocessing failed - {str(e)}")
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()