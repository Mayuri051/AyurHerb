"""
Practical Experiment 10: Implementation of Supervised Learning Algorithms (AdaBoost & Random Forest)
Syllabus Unit: Unit V - Advanced ML Classification Techniques (LO4, LO5)
Description: Implements and compares ensemble supervised learning algorithms:
             a. Ada-Boosting (Adaptive Boosting)
             b. Random Forests (Bootstrap Aggregation + Feature Subspace Sampling)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 70)
    print("EXPERIMENT 10: ADABOOST & RANDOM FOREST ENSEMBLE CLASSIFICATION (UNIT V)")
    print("=" * 70)

    # 1. Load standard benchmark dataset
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(target_names)} classes.")

    # 2. Train-Test Split (75/25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Random Forest Implementation (Bagging Ensemble)
    print("\n--- 1. Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # Feature Importances from Random Forest
    print("Top 5 Most Important Features in Random Forest:")
    importances = rf.feature_importances_
    for idx in np.argsort(importances)[::-1][:5]:
        print(f"  - {feature_names[idx]}: {importances[idx]*100:.2f}%")

    # 4. AdaBoost Implementation (Boosting Ensemble)
    print("\n--- 2. AdaBoost Classifier ---")
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=50,
        learning_rate=0.8,
        random_state=42
    )
    ada.fit(X_train, y_train)
    ada_preds = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, ada_preds)
    print(f"AdaBoost Test Accuracy: {ada_acc * 100:.2f}%")

    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY:")
    print(f"  Random Forest Accuracy: {rf_acc * 100:.2f}%")
    print(f"  AdaBoost Accuracy:      {ada_acc * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

