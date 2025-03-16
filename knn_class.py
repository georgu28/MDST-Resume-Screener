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
from sklearn.metrics import classification_report
from parser import read_pdf
from parser import get_details
import sys

class knn_pred:
    data = pd.read_csv('UpdatedResumeDataSet.csv', encoding='latin')
    model = KNeighborsClassifier()
    tfidf = TfidfVectorizer()
    vectorizer = TfidfVectorizer(stop_words="english")
    le = LabelEncoder()

    def __init__(self):
        self.fit_model()

    def clean_data(self):
        self.data['Numeric Category'] = self.le.fit_transform(self.data['Category'])
        self.data.drop_duplicates(keep='first', inplace=True)
    
    def fit_model(self):
        self.clean_data()

        X = self.data["Resume"]
        y = self.data["Numeric Category"]
        self.vectorizer.fit(X)
        features = self.vectorizer.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=0.2,
            shuffle=True,
            random_state=2024,
        )

        # choose highest freq label in neighbors
        self.model.fit(X_train, y_train)

        # predicting
        # pred = self.model.predict(X_test)
        # print(classification_report(y_test, pred, zero_division=True))

    def predict_pdf(self,pdf_path):
        pdf_data = read_pdf(pdf_path)

        features = self.vectorizer.transform([pdf_data])
        
        pred = self.model.predict(features)
        pred = self.le.inverse_transform(pred)
        print(pred[0])
        return pred[0]

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("give path")
        exit(1)
    name = sys.argv[1]

    model_class = knn_pred()
    model_class.predict_pdf(name)
