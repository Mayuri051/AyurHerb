"""
Practical Experiment 11: Evaluation of Classification Algorithms
Syllabus Unit: Unit V - Advanced ML Classification Techniques (LO5)
Description: Evaluates classification performance metrics:
             Confusion Matrix, Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def main():
    print("=" * 70)
    print("EXPERIMENT 11: EVALUATION OF CLASSIFICATION ALGORITHMS (UNIT V)")
    print("=" * 70)

    # 1. Binary Classification Benchmark
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # 2. Train Classifier
    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 3. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\n--- 1. Confusion Matrix Analysis ---")
    print(f"True Positives (TP):  {tp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp} (Type I Error)")
    print(f"False Negatives (FN): {fn} (Type II Error)")

    print("\n--- 2. Performance Metric Computations ---")
    print(f"Accuracy  = (TP + TN) / Total        = {acc * 100:.2f}%")
    print(f"Precision = TP / (TP + FP)           = {prec * 100:.2f}%")
    print(f"Recall    = TP / (TP + FN) (Sensitivity) = {rec * 100:.2f}%")
    print(f"F1-Score  = 2 * (P * R) / (P + R)   = {f1 * 100:.2f}%")
    print(f"ROC-AUC Score                        = {roc_auc:.4f}")

    print("\n--- 3. Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("=" * 70)
    print("Classification Evaluation Experiment Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

