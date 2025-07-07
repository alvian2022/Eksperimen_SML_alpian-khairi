"""
Automated Data Preprocessing Pipeline for Diabetes Prediction Dataset
Author: alpian_khairi_C1BO
Description: Automated preprocessing for diabetes prediction dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import logging
from typing import Tuple, Optional, Dict
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DiabetesDataPreprocessor:
    """
    Automated data preprocessing class for Diabetes prediction dataset
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.feature_names = None
        self.categorical_features = ['gender', 'smoking_history']
        self.numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        self.target_column = 'diabetes'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load diabetes dataset from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame: Loaded dataset
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
                
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = self.categorical_features + self.numerical_features + [self.target_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Save raw data
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/diabetes_raw.csv', index=False)
            logger.info("Raw data saved to data/diabetes_raw.csv")
            
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame) -> Dict:
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
            'target_distribution': df[self.target_column].value_counts().to_dict(),
            'categorical_stats': {},
            'numerical_stats': df[self.numerical_features].describe().to_dict()
        }
        
        # Categorical feature analysis
        for cat_feature in self.categorical_features:
            if cat_feature in df.columns:
                exploration_results['categorical_stats'][cat_feature] = df[cat_feature].value_counts().to_dict()
        
        logger.info(f"Exploration completed:")
        logger.info(f"  - Shape: {exploration_results['shape']}")
        logger.info(f"  - Missing values: {sum(exploration_results['missing_values'].values())}")
        logger.info(f"  - Duplicates: {exploration_results['duplicates']}")
        logger.info(f"  - Target distribution: {exploration_results['target_distribution']}")
        
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
        df_clean = df.copy()
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Found {missing_before} missing values")
            
            # For numerical columns, fill with median
            for col in self.numerical_features:
                if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.info(f"  - Filled {col} missing values with median: {median_val:.2f}")
            
            # For categorical columns, fill with mode
            for col in self.categorical_features:
                if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    logger.info(f"  - Filled {col} missing values with mode: {mode_val}")
        
        # Remove duplicates
        duplicates_before = df_clean.duplicated().sum()
        if duplicates_before > 0:
            logger.info(f"Found {duplicates_before} duplicate rows")
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        
        # Check for outliers using IQR method
        outlier_info = self._detect_outliers(df_clean)
        logger.info(f"Outlier detection completed: {outlier_info}")
        
        # Data type validation and cleaning
        df_clean = self._validate_data_types(df_clean)
        
        # Fix: Replace Unicode arrow with ASCII
        logger.info(f"Data cleaning completed. Shape: {original_shape} -> {df_clean.shape}")
        return df_clean
    
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with validated data types
        """
        df_clean = df.copy()
        
        # Ensure categorical columns are strings
        for col in self.categorical_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        
        # Ensure numerical columns are numeric
        for col in self.numerical_features:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Ensure target is integer
        if self.target_column in df_clean.columns:
            df_clean[self.target_column] = df_clean[self.target_column].astype(int)
        
        # Validate ranges for specific features
        if 'age' in df_clean.columns:
            # Age should be positive and reasonable
            df_clean = df_clean[df_clean['age'] > 0]
            df_clean = df_clean[df_clean['age'] <= 120]
        
        if 'bmi' in df_clean.columns:
            # BMI should be positive and reasonable
            df_clean = df_clean[df_clean['bmi'] > 0]
            df_clean = df_clean[df_clean['bmi'] <= 70]
        
        logger.info("Data type validation completed")
        return df_clean.reset_index(drop=True)
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """
        Detect outliers using IQR method for numerical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Outlier information
        """
        outlier_info = {}
        
        for feature in self.numerical_features:
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
        
        df_processed = df.copy()
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Handle categorical features with label encoding
        X_encoded = X.copy()
        for cat_feature in self.categorical_features:
            if cat_feature in X_encoded.columns:
                self.label_encoders[cat_feature] = LabelEncoder()
                X_encoded[cat_feature] = self.label_encoders[cat_feature].fit_transform(X_encoded[cat_feature])
                
                # Fixed: Properly handle multi-line f-string
                encoding_map = dict(zip(
                    self.label_encoders[cat_feature].classes_, 
                    self.label_encoders[cat_feature].transform(self.label_encoders[cat_feature].classes_)
                ))
                logger.info(f"Label encoded {cat_feature}: {encoding_map}")
        
        # Scale numerical features
        logger.info("Scaling numerical features using StandardScaler...")
        numerical_data = X_encoded[self.numerical_features]
        numerical_scaled = self.scaler.fit_transform(numerical_data)
        
        # Combine scaled numerical and encoded categorical features
        X_final = X_encoded.copy()
        X_final[self.numerical_features] = numerical_scaled
        
        # Feature statistics
        feature_stats = {
            'original_ranges': {col: (X[col].min(), X[col].max()) for col in self.numerical_features if col in X.columns},
            'scaled_ranges': {col: (X_final[col].min(), X_final[col].max()) for col in self.numerical_features if col in X_final.columns}
        }
        
        logger.info("Feature engineering completed")
        logger.info(f"Final feature columns: {list(X_final.columns)}")
        
        return X_final, y
    
    # def validate_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> Dict:
    #     """
    #     Validate preprocessing quality with a quick model test
        
    #     Args:
    #         X: Features DataFrame
    #         y: Target Series
            
    #     Returns:
    #         dict: Validation results
    #     """
    #     logger.info("Starting preprocessing validation...")
        
    #     try:
    #         # Split data for validation
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, random_state=42, stratify=y
    #         )
            
    #         # Train a simple model
    #         model = RandomForestClassifier(n_estimators=100, random_state=42)
    #         model.fit(X_train, y_train)
            
    #         # Make predictions
    #         y_pred = model.predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
            
    #         # Generate classification report
    #         target_names = ['No Diabetes', 'Diabetes']
    #         class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
    #         validation_results = {
    #             'accuracy': accuracy,
    #             'classification_report': class_report,
    #             'validation_successful': True,
    #             'feature_importance': dict(zip(X.columns, model.feature_importances_))
    #         }
            
    #         logger.info(f"Validation completed successfully. Accuracy: {accuracy:.4f}")
            
    #         # Log feature importance
    #         top_features = sorted(validation_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
    #         logger.info(f"Top 5 important features: {top_features}")
            
    #         return validation_results
            
    #     except Exception as e:
    #         logger.error(f"Validation failed: {str(e)}")
    #         return {
    #             'accuracy': 0.0,
    #             'classification_report': {},
    #             'validation_successful': False,
    #             'error': str(e)
    #         }
    
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
            processed_df[self.target_column] = y
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            processed_df.to_csv(output_path, index=False)
            
            # Save preprocessing objects for future use
            preprocessing_objects = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': list(X.columns),
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'target_column': self.target_column
            }
            
            objects_path = output_path.replace('.csv', '_objects.pkl')
            joblib.dump(preprocessing_objects, objects_path)
            
            logger.info(f"Preprocessed data saved to: {output_path}")
            logger.info(f"Preprocessing objects saved to: {objects_path}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise
    
    def preprocess_pipeline(self, 
                          input_path: str, 
                          output_path: str = "preprocessing/diabetes_preprocessed.csv") -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            input_path: Input file path
            output_path: Output file path
            
        Returns:
            DataFrame: Preprocessed dataset
        """
        logger.info("=== Starting Automated Diabetes Preprocessing Pipeline ===")
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
            # validation_results = self.validate_preprocessing(X_processed, y_processed)
            
            # Step 6: Save preprocessed data
            self.save_preprocessed_data(X_processed, y_processed, output_path)
            
            # Step 7: Create final dataset
            final_df = X_processed.copy()
            final_df[self.target_column] = y_processed
            
            # Step 8: Generate summary report
            # self._generate_summary_report(exploration_results, final_df, output_path, validation_results)
            self._generate_summary_report(exploration_results, final_df, output_path)
            
            logger.info("=== Preprocessing Pipeline Completed Successfully ===")
            return final_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _generate_summary_report(self, exploration_results: Dict, final_df: pd.DataFrame, 
                           output_path: str) -> None:
        """
        Generate preprocessing summary report
        
        Args:
            exploration_results: Results from data exploration
            final_df: Final preprocessed DataFrame
            output_path: Output file path
            validation_results: Results from validation
        """
        report_path = output_path.replace('.csv', '_report.txt')
        
        # Fix: Use UTF-8 encoding for file writing
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AUTOMATED DIABETES PREPROCESSING REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Author: alpian_khairi_C1BO\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("ORIGINAL DATA:\n")
            f.write(f"  Shape: {exploration_results['shape']}\n")
            f.write(f"  Missing values: {sum(exploration_results['missing_values'].values())}\n")
            f.write(f"  Duplicates: {exploration_results['duplicates']}\n")
            f.write(f"  Target distribution: {exploration_results['target_distribution']}\n\n")
            
            f.write("PROCESSED DATA:\n")
            f.write(f"  Final shape: {final_df.shape}\n")
            f.write(f"  Features: {list(final_df.columns[:-1])}\n")
            f.write(f"  Target distribution: {final_df[self.target_column].value_counts().to_dict()}\n\n")
            
            f.write("CATEGORICAL FEATURES:\n")
            for cat_feature in self.categorical_features:
                if cat_feature in exploration_results['categorical_stats']:
                    f.write(f"  {cat_feature}: {exploration_results['categorical_stats'][cat_feature]}\n")
            f.write("\n")
            
            # f.write("VALIDATION RESULTS:\n")
            # if validation_results['validation_successful']:
            #     f.write(f"  Accuracy: {validation_results['accuracy']:.4f}\n")
            #     f.write(f"  Validation: PASSED\n")
            #     if 'feature_importance' in validation_results:
            #         top_features = sorted(validation_results['feature_importance'].items(), 
            #                             key=lambda x: x[1], reverse=True)[:5]
            #         f.write(f"  Top 5 features: {top_features}\n")
            #     f.write("\n")
            # else:
            #     f.write(f"  Validation: FAILED\n")
            #     f.write(f"  Error: {validation_results.get('error', 'Unknown error')}\n\n")
            
            f.write("PREPROCESSING STEPS APPLIED:\n")
            # Fix: Replace Unicode checkmarks with ASCII
            f.write("  [x] Data loading and validation\n")
            f.write("  [x] Missing value handling\n")
            f.write("  [x] Duplicate removal\n")
            f.write("  [x] Data type validation\n")
            f.write("  [x] Outlier detection\n")
            f.write("  [x] Categorical encoding (Label Encoding)\n")
            f.write("  [x] Numerical feature scaling (StandardScaler)\n")
            # f.write("  [x] Data validation with ML model\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the preprocessing pipeline"""
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('preprocessing', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor()
    
    # Run preprocessing pipeline
    try:
        # Use the diabetes dataset file
        input_file = "data/diabetes_raw.csv"
        
        if not os.path.exists(input_file):
            print(f"ERROR: Dataset file '{input_file}' not found!")
            print("Please ensure the diabetes prediction dataset CSV file is in the current directory.")
            return
        
        processed_data = preprocessor.preprocess_pipeline(
            input_path=input_file,
            output_path="preprocessing/diabetes_preprocessed.csv"
        )
        
        print("\n" + "="*60)
        print("DIABETES PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final dataset shape: {processed_data.shape}")
        print(f"Features: {list(processed_data.columns[:-1])}")
        print(f"Target classes: {sorted(processed_data['diabetes'].unique())}")
        print(f"Target distribution: {processed_data['diabetes'].value_counts().to_dict()}")
        print(f"Output saved to: preprocessing/diabetes_preprocessed.csv")
        print("\nFiles generated:")
        print("  - preprocessing/diabetes_preprocessed.csv")
        print("  - preprocessing/diabetes_preprocessed_objects.pkl")
        print("  - preprocessing/diabetes_preprocessed_report.txt")
        print("  - diabetes_preprocessing.log")
        
    except Exception as e:
        print(f"ERROR: Preprocessing failed - {str(e)}")
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()