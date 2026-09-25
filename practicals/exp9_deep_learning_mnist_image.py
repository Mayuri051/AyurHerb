"""
Practical Experiment 9: Deep Learning Image Classification System (MNIST / Digit Recognition)
Syllabus Unit: Unit IV - Introduction to Deep Learning (LO4)
Description: Builds, trains, and evaluates a Deep Neural Network / Multi-Layer Perceptron (MLP)
             for handwritten digit recognition using standard datasets (MNIST / Digits).
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 70)
    print("EXPERIMENT 9: DEEP LEARNING IMAGE CLASSIFICATION (UNIT IV)")
    print("=" * 70)

    # 1. Load standard 8x8 handwritten digit dataset
    print("Loading image dataset (1797 samples of 8x8 grayscale pixel images)...")
    digits = load_digits()
    X = digits.data / 16.0  # Normalize pixel values to [0, 1]
    y = digits.target

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} images | Test set: {X_test.shape[0]} images")
    print(f"Feature Dimension: {X_train.shape[1]} input nodes (8x8 pixels)")

    # 3. Build Deep Neural Network Architecture
    # 64 Input Nodes -> Hidden Layer 1 (128 ReLU) -> Hidden Layer 2 (64 ReLU) -> 10 Output Softmax
    print("\nInitializing Deep Neural Network Architecture:")
    print("  - Input Layer: 64 nodes")
    print("  - Hidden Layer 1: 128 neurons (ReLU activation)")
    print("  - Hidden Layer 2: 64 neurons (ReLU activation)")
    print("  - Output Layer: 10 classes (Softmax probability)")
    print("  - Optimizer: Adam (Adaptive Moment Estimation), Early Stopping = True")

    dnn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True
    )

    # 4. Train Model
    print("\nTraining Deep Neural Network...")
    dnn_model.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = dnn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[+] Deep Learning Model Training Completed in {dnn_model.n_iter_} iterations.")
    print(f"    - Test Accuracy: {acc * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n" + "=" * 70)
    print("Deep Learning Image Classification Experiment Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

