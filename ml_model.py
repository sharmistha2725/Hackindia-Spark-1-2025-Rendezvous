import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class SeverityScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = RandomForestClassifier()
        self.severity_mapping = {
            'mild': 1,
            'severe': 2,
            'critical': 3
        }

    def train(self, dataset_path):
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Prepare features (combine relevant text columns)
        texts = df['Symptoms & Clinical Features'].fillna('') + ' ' + \
                df['Diagnosis Methods'].fillna('') + ' ' + \
                df['Treatment Options'].fillna('')
        
        # Prepare labels
        labels = df['Severity Level'].str.lower()
        labels = labels[labels.isin(['mild', 'severe', 'critical'])]
        label_nums = labels.map(self.severity_mapping)
        
        # Create TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Train model
        self.model.fit(X, label_nums)
        
        # Save model
        with open('severity_model.pkl', 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)
        
        return True

    def predict_severity(self, symptoms, diagnosis, treatment):
        # Load model if exists
        if os.path.exists('severity_model.pkl'):
            with open('severity_model.pkl', 'rb') as f:
                self.vectorizer, self.model = pickle.load(f)
        
        # Combine text
        text = f"{symptoms} {diagnosis} {treatment}"
        
        # Create features
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        # Convert to 1-10 scale
        severity_score = int((prediction / 3) * 10)
        
        return {
            'score': severity_score,
            'confidence': confidence,
            'prediction': prediction
        }

# Initialize model
severity_scorer = SeverityScorer()

# Train model if dataset exists
if os.path.exists('patient_severity_dataset.csv'):
    severity_scorer.train('patient_severity_dataset.csv')
