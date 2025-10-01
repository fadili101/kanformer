"""
KANFormer - Dataset Preparation and Loading
============================================
This script handles dataset loading, preprocessing, and preparation
for the KANFormer model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class DatasetLoader:
    """Class to handle loading and preprocessing of educational datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_uci_student_performance(self, subject='math'):
        """
        Load UCI Student Performance Dataset
        
        Parameters:
        -----------
        subject : str
            'math' or 'portuguese'
        
        Returns:
        --------
        X : numpy array
            Features
        y : numpy array
            Labels
        """
        # URLs for UCI datasets
        urls = {
            'math': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip',
            'portuguese': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
        }
        
        print(f"Loading UCI Student Performance ({subject})...")
        
        # For demonstration, creating synthetic data similar to UCI structure
        # In practice, download from UCI repository
        n_samples = 395 if subject == 'math' else 649
        
        # Create synthetic data matching UCI structure
        data = self._create_uci_synthetic(n_samples)
        
        # Prepare features and target
        X = data.drop('G3', axis=1)
        y = data['G3']
        
        # Convert grades to categories: Low (0-9), Medium (10-14), High (15-20)
        y = pd.cut(y, bins=[0, 9, 14, 20], labels=['Low', 'Medium', 'High'])
        
        self.feature_names = X.columns.tolist()
        
        return X.values, y.values
    
    def _create_uci_synthetic(self, n_samples):
        """Create synthetic data matching UCI Student Performance structure"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'school': np.random.choice(['GP', 'MS'], n_samples),
            'sex': np.random.choice(['F', 'M'], n_samples),
            'age': np.random.randint(15, 23, n_samples),
            'address': np.random.choice(['U', 'R'], n_samples),
            'famsize': np.random.choice(['LE3', 'GT3'], n_samples),
            'Pstatus': np.random.choice(['T', 'A'], n_samples),
            'Medu': np.random.randint(0, 5, n_samples),
            'Fedu': np.random.randint(0, 5, n_samples),
            'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n_samples),
            'guardian': np.random.choice(['mother', 'father', 'other'], n_samples),
            'traveltime': np.random.randint(1, 5, n_samples),
            'studytime': np.random.randint(1, 5, n_samples),
            'failures': np.random.randint(0, 4, n_samples),
            'schoolsup': np.random.choice(['yes', 'no'], n_samples),
            'famsup': np.random.choice(['yes', 'no'], n_samples),
            'paid': np.random.choice(['yes', 'no'], n_samples),
            'activities': np.random.choice(['yes', 'no'], n_samples),
            'nursery': np.random.choice(['yes', 'no'], n_samples),
            'higher': np.random.choice(['yes', 'no'], n_samples),
            'internet': np.random.choice(['yes', 'no'], n_samples),
            'romantic': np.random.choice(['yes', 'no'], n_samples),
            'famrel': np.random.randint(1, 6, n_samples),
            'freetime': np.random.randint(1, 6, n_samples),
            'goout': np.random.randint(1, 6, n_samples),
            'Dalc': np.random.randint(1, 6, n_samples),
            'Walc': np.random.randint(1, 6, n_samples),
            'health': np.random.randint(1, 6, n_samples),
            'absences': np.random.randint(0, 75, n_samples),
            'G1': np.random.randint(0, 21, n_samples),
            'G2': np.random.randint(0, 21, n_samples),
        })
        
        # G3 depends on G1 and G2 with some noise
        data['G3'] = np.clip(
            (data['G1'] * 0.3 + data['G2'] * 0.5 + np.random.randn(n_samples) * 2).astype(int),
            0, 20
        )
        
        return data
    
    def create_synthetic_dataset(self, n_samples=10000, n_features=25):
        """
        Create synthetic educational dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of student records
        n_features : int
            Number of features
        
        Returns:
        --------
        X : numpy array
            Features
        y : numpy array
            Labels (academic orientations)
        """
        print(f"Creating synthetic dataset with {n_samples} samples...")
        np.random.seed(42)
        
        # Define academic orientations
        orientations = ['Science', 'Arts', 'Business', 'Engineering']
        n_classes = len(orientations)
        
        # Generate features
        data = {}
        
        # Academic performance features
        data['previous_gpa'] = np.random.uniform(2.0, 4.0, n_samples)
        data['math_score'] = np.random.uniform(50, 100, n_samples)
        data['language_score'] = np.random.uniform(50, 100, n_samples)
        data['science_score'] = np.random.uniform(50, 100, n_samples)
        data['midterm_score'] = np.random.uniform(50, 100, n_samples)
        data['failed_courses'] = np.random.randint(0, 5, n_samples)
        
        # Behavioral features
        data['attendance_rate'] = np.random.uniform(0.6, 1.0, n_samples)
        data['assignment_submission_rate'] = np.random.uniform(0.5, 1.0, n_samples)
        data['study_time_weekly'] = np.random.uniform(5, 40, n_samples)
        data['punctuality_score'] = np.random.uniform(0.5, 1.0, n_samples)
        
        # Engagement features
        data['forum_participation'] = np.random.randint(0, 100, n_samples)
        data['library_usage'] = np.random.randint(0, 50, n_samples)
        data['office_hours_attendance'] = np.random.randint(0, 20, n_samples)
        data['group_project_score'] = np.random.uniform(50, 100, n_samples)
        
        # Demographic features
        data['age'] = np.random.randint(18, 25, n_samples)
        data['gender'] = np.random.randint(0, 2, n_samples)  # 0: Female, 1: Male
        data['socioeconomic_status'] = np.random.randint(1, 6, n_samples)
        data['parent_education'] = np.random.randint(1, 6, n_samples)
        data['distance_to_school'] = np.random.uniform(1, 50, n_samples)
        
        # Social features
        data['peer_interaction_index'] = np.random.uniform(0, 1, n_samples)
        data['extracurricular_activities'] = np.random.randint(0, 10, n_samples)
        data['leadership_roles'] = np.random.randint(0, 5, n_samples)
        
        # Additional features
        data['internet_access'] = np.random.randint(0, 2, n_samples)
        data['device_ownership'] = np.random.randint(0, 2, n_samples)
        data['motivation_score'] = np.random.uniform(1, 10, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate target based on features with realistic correlations
        y = []
        for idx in range(n_samples):
            # Science: high math and science scores
            science_prob = (df.loc[idx, 'math_score'] + df.loc[idx, 'science_score']) / 200
            # Arts: high language scores
            arts_prob = df.loc[idx, 'language_score'] / 100
            # Business: balanced scores, high social interaction
            business_prob = (df.loc[idx, 'peer_interaction_index'] + 
                           df.loc[idx, 'previous_gpa'] / 4) / 2
            # Engineering: high math and science, high study time
            engineering_prob = (df.loc[idx, 'math_score'] + df.loc[idx, 'science_score'] + 
                              df.loc[idx, 'study_time_weekly']) / 240
            
            probs = np.array([science_prob, arts_prob, business_prob, engineering_prob])
            probs = probs / probs.sum()
            
            orientation = np.random.choice(orientations, p=probs)
            y.append(orientation)
        
        self.feature_names = df.columns.tolist()
        
        return df.values, np.array(y)
    
    def create_oulad_synthetic(self, n_samples=5000):
        """
        Create synthetic data similar to OULAD structure
        
        Returns:
        --------
        X : numpy array
            Features
        y : numpy array
            Labels (Distinction, Pass, Fail, Withdrawn)
        """
        print(f"Creating OULAD-like synthetic dataset with {n_samples} samples...")
        np.random.seed(42)
        
        # Define outcomes
        outcomes = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
        
        # Generate features similar to OULAD
        data = {}
        
        # VLE interaction features
        data['total_clicks'] = np.random.randint(100, 10000, n_samples)
        data['resource_views'] = np.random.randint(50, 5000, n_samples)
        data['forum_posts'] = np.random.randint(0, 200, n_samples)
        data['quiz_attempts'] = np.random.randint(5, 100, n_samples)
        data['video_views'] = np.random.randint(10, 500, n_samples)
        
        # Assessment scores
        data['assignment_1_score'] = np.random.uniform(0, 100, n_samples)
        data['assignment_2_score'] = np.random.uniform(0, 100, n_samples)
        data['assignment_3_score'] = np.random.uniform(0, 100, n_samples)
        data['exam_score'] = np.random.uniform(0, 100, n_samples)
        
        # Engagement metrics
        data['days_active'] = np.random.randint(30, 250, n_samples)
        data['avg_session_duration'] = np.random.uniform(10, 180, n_samples)
        data['late_submissions'] = np.random.randint(0, 10, n_samples)
        
        # Demographics
        data['age_band'] = np.random.choice([0, 1, 2], n_samples)  # 0-35, 35-55, 55+
        data['num_prev_attempts'] = np.random.randint(0, 4, n_samples)
        data['studied_credits'] = np.random.randint(30, 240, n_samples)
        data['disability'] = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        # Generate target based on features
        y = []
        for idx in range(n_samples):
            # Calculate performance indicators
            avg_score = (df.loc[idx, 'assignment_1_score'] + 
                        df.loc[idx, 'assignment_2_score'] + 
                        df.loc[idx, 'assignment_3_score'] + 
                        df.loc[idx, 'exam_score']) / 4
            
            engagement = (df.loc[idx, 'total_clicks'] / 10000 + 
                         df.loc[idx, 'days_active'] / 250) / 2
            
            # Determine outcome
            if avg_score > 80 and engagement > 0.7:
                outcome = 'Distinction'
            elif avg_score > 50 and engagement > 0.4:
                outcome = 'Pass'
            elif engagement < 0.2:
                outcome = 'Withdrawn'
            else:
                outcome = 'Fail'
            
            y.append(outcome)
        
        self.feature_names = df.columns.tolist()
        
        return df.values, np.array(y)
    
    def preprocess_data(self, X, y, test_size=0.2, apply_smote=True):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        X : numpy array
            Features
        y : numpy array
            Labels
        test_size : float
            Proportion of test set
        apply_smote : bool
            Whether to apply SMOTE for balancing
        
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\nPreprocessing pipeline:")
        print("=" * 50)
        
        # 1. Handle missing values
        print("1. Handling missing values with KNN imputation...")
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)
        
        # 2. Encode labels
        print("2. Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"   Classes: {self.label_encoder.classes_}")
        
        # 3. Split data (64% train, 16% val, 20% test)
        print("3. Splitting data (64% train, 16% val, 20% test)...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_imputed, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, 
            random_state=42, stratify=y_temp
        )
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 4. Apply SMOTE for class balancing
        if apply_smote:
            print("4. Applying SMOTE for class balancing...")
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"   Before SMOTE: {dict(zip(unique, counts))}")
            
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"   After SMOTE: {dict(zip(unique, counts))}")
        
        # 5. Normalize features
        print("5. Normalizing features (Z-score)...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # 6. Feature selection (optional but recommended)
        print("6. Feature selection...")
        X_train, X_val, X_test = self._select_features(
            X_train, X_val, X_test, y_train
        )
        
        print("\nPreprocessing complete!")
        print("=" * 50)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _select_features(self, X_train, X_val, X_test, y_train, n_features=20):
        """Feature selection using mutual information and RFE"""
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # Get top features by MI
        n_select = min(n_features, len(mi_scores))
        top_features_idx = np.argsort(mi_scores)[-n_select:]
        
        print(f"   Selected {n_select} features based on mutual information")
        
        return (X_train[:, top_features_idx], 
                X_val[:, top_features_idx], 
                X_test[:, top_features_idx])


# Usage example
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Test 1: Synthetic dataset
    print("\n" + "="*70)
    print("DATASET 1: SYNTHETIC EDUCATIONAL DATA")
    print("="*70)
    X_syn, y_syn = loader.create_synthetic_dataset(n_samples=10000)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.preprocess_data(X_syn, y_syn)
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Test 2: UCI Math
    print("\n" + "="*70)
    print("DATASET 2: UCI STUDENT PERFORMANCE (MATHEMATICS)")
    print("="*70)
    X_math, y_math = loader.load_uci_student_performance(subject='math')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.preprocess_data(X_math, y_math)
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Test 3: OULAD-like
    print("\n" + "="*70)
    print("DATASET 3: OULAD-LIKE SYNTHETIC DATA")
    print("="*70)
    X_oulad, y_oulad = loader.create_oulad_synthetic(n_samples=5000)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.preprocess_data(X_oulad, y_oulad)
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    print("\n" + "="*70)
    print("Dataset preparation complete! Ready for KANFormer training.")
    print("="*70)
