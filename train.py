import os

print("MLFLOW_TRACKING_URI before setting:", os.getenv("MLFLOW_TRACKING_URI"))

# Set a clean, absolute Linux-friendly path
os.environ["MLFLOW_TRACKING_URI"] = os.path.abspath("./mlruns")

print("MLFLOW_TRACKING_URI after setting:", os.getenv("MLFLOW_TRACKING_URI"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow

mlflow.set_experiment("digits-experiment")

def train():
    digits = load_digits()
    X = pd.DataFrame(digits.data)
    y = pd.Series(digits.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=10000)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 10000)
        mlflow.log_metric("accuracy", acc)

        joblib.dump(model, "best_model.pkl")
        mlflow.log_artifact("best_model.pkl")

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

if __name__ == "__main__":
    train()