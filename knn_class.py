import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from parser import read_pdf
from parser import get_details
import sys
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeClassifier:
    """
    A K-Nearest Neighbors classifier for categorizing resumes into job categories.
    Uses TF-IDF vectorization for text feature extraction.
    """
    
    def __init__(self, dataset_path: str = 'UpdatedResumeDataSet.csv', n_neighbors: int = 5):
        """
        Initialize the resume classifier.
        
        Args:
            dataset_path (str): Path to the CSV dataset
            n_neighbors (int): Number of neighbors for KNN classifier
        """
        try:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
                
            self.data = pd.read_csv(dataset_path, encoding='latin')
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.le = LabelEncoder()
            self.is_fitted = False
            
            logger.info(f"Initialized classifier with {len(self.data)} samples")
            self.fit_model()
            
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise

    def clean_data(self) -> None:
        """
        Clean and preprocess the dataset.
        
        This method:
        - Encodes categorical labels to numeric values
        - Removes duplicate entries
        - Handles missing values
        """
        try:
            # Remove any missing values
            self.data = self.data.dropna()
            
            # Encode category labels to numeric values
            self.data['Numeric Category'] = self.le.fit_transform(self.data['Category'])
            
            # Remove duplicate resumes
            self.data.drop_duplicates(subset=['Resume'], keep='first', inplace=True)
            
            logger.info(f"Data cleaned: {len(self.data)} samples remaining")
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
    
    def fit_model(self) -> None:
        """
        Train the KNN model on the cleaned dataset.
        
        This method:
        - Cleans the data
        - Vectorizes the resume text using TF-IDF
        - Splits data into train/test sets
        - Trains the KNN classifier
        - Evaluates model performance
        """
        try:
            self.clean_data()

            X = self.data["Resume"]
            y = self.data["Numeric Category"]
            
            # Fit vectorizer and transform text to features
            self.vectorizer.fit(X)
            features = self.vectorizer.transform(X)

            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                y,
                test_size=0.2,
                shuffle=True,
                random_state=2024,
                stratify=y  # Ensure balanced splits
            )

            # Train the model
            self.model.fit(X_train, y_train)
            self.is_fitted = True

            # Evaluate model performance
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}")
            
            # Print detailed classification report
            logger.info("Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred, zero_division=1)}")
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise

    def predict_pdf(self, pdf_path: str) -> str:
        """
        Predict the job category for a resume PDF.
        
        Args:
            pdf_path (str): Path to the PDF resume file
            
        Returns:
            str: Predicted job category
            
        Raises:
            ValueError: If model is not fitted or file doesn't exist
            Exception: If PDF cannot be processed
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Extract text from PDF
            pdf_data = read_pdf(pdf_path)
            
            if not pdf_data.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return "Unknown"
            
            # Transform text to features using fitted vectorizer
            features = self.vectorizer.transform([pdf_data])
            
            # Make prediction
            pred_numeric = self.model.predict(features)
            
            # Convert back to category label
            pred_category = self.le.inverse_transform(pred_numeric)
            
            logger.info(f"Predicted category for {pdf_path}: {pred_category[0]}")
            return pred_category[0]
            
        except Exception as e:
            logger.error(f"Error predicting PDF {pdf_path}: {e}")
            raise
    
    def predict_text(self, text: str) -> str:
        """
        Predict the job category for raw resume text.
        
        Args:
            text (str): Resume text content
            
        Returns:
            str: Predicted job category
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            features = self.vectorizer.transform([text])
            pred_numeric = self.model.predict(features)
            pred_category = self.le.inverse_transform(pred_numeric)
            
            return pred_category[0]
            
        except Exception as e:
            logger.error(f"Error predicting text: {e}")
            raise
    
    def get_categories(self) -> list:
        """
        Get all available job categories.
        
        Returns:
            list: List of all job categories in the dataset
        """
        return list(self.le.classes_)
    
    def get_prediction_probabilities(self, pdf_path: str) -> dict:
        """
        Get prediction probabilities for all categories.
        
        Args:
            pdf_path (str): Path to the PDF resume file
            
        Returns:
            dict: Category names mapped to prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            pdf_data = read_pdf(pdf_path)
            features = self.vectorizer.transform([pdf_data])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            categories = self.le.classes_
            
            # Create category-probability mapping
            prob_dict = dict(zip(categories, probabilities))
            
            # Sort by probability descending
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_probs
            
        except Exception as e:
            logger.error(f"Error getting probabilities for {pdf_path}: {e}")
            raise


# Legacy class name for backward compatibility
class knn_pred(ResumeClassifier):
    """Legacy class name - use ResumeClassifier instead."""
    pass

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("give path")
        exit(1)
    name = sys.argv[1]

    model_class = knn_pred()
    model_class.predict_pdf(name)
