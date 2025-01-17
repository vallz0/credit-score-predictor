import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple

class CreditScorePredictor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }

    def encode_column(self, column_name: str) -> None:
        encoder = LabelEncoder()
        self.data[column_name] = encoder.fit_transform(self.data[column_name])
        self.label_encoders[column_name] = encoder

    def preprocess_data(self) -> None:
        columns_to_encode = ["profissao", "mix_credito", "comportamento_pagamento"]
        for column in columns_to_encode:
            self.encode_column(column)

        y = self.data["score_credito"]
        x = self.data.drop(columns="score_credito")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    def train_models(self) -> None:
        for model_name, model in self.models.items():
            model.fit(self.x_train, self.y_train)

    def evaluate_models(self) -> None:
        for model_name, model in self.models.items():
            predictions = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions) * 100
            print(f"Accuracy for {model_name}: {accuracy:.2f}%")

    def analyze_data(self) -> None:
        plt.figure(figsize=(10, 6))
        self.data["score_credito"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Distribution of Credit Scores")
        plt.xlabel("Credit Score")
        plt.ylabel("Count")
        plt.show()
