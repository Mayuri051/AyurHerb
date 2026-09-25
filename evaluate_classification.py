"""
AyurHerb - Machine Learning Evaluation Suite (evaluate_classification.py)
Author: Data Science Team
Description: Benchmarks and evaluates multiple supervised classification algorithms:
             1. Random Forest Classifier
             2. AdaBoost Classifier
             3. Decision Tree Classifier
             4. Multinomial Naive Bayes
             Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix, and saves plots.
             Directly fulfills Syllabus Unit V (Experiments 10a, 10b, 11).
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)


def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run_benchmark(data_path: str, output_dir: str) -> dict:
    """Execute full model benchmark and generate analytical visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Filter conditions that have enough representation
    value_counts = df["Disease"].value_counts()
    frequent_diseases = value_counts[value_counts >= 3].index
    df_filtered = df[df["Disease"].isin(frequent_diseases)].copy()

    if len(df_filtered) < 20:
        df_filtered = df.copy()  # Fallback to full dataset

    df_filtered["Clean_Symptoms"] = df_filtered["Symptoms"].fillna("").apply(clean_text)
    df_filtered["Features"] = (
        df_filtered["Clean_Symptoms"] + " " +
        df_filtered["Doshas"].fillna("").apply(clean_text) + " " +
        df_filtered["Symptom Severity"].fillna("").apply(clean_text)
    )

    X_text = df_filtered["Features"]
    y = df_filtered["Disease"].astype(str).str.strip()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=1500)
    X = vectorizer.fit_transform(X_text)

    # Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=60,
            learning_rate=0.8,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.5)
    }

    results = {}
    
    print("=" * 80)
    print("AYURHERB - ADVANCED ML CLASSIFICATION BENCHMARK & EVALUATION")
    print("=" * 80)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} TF-IDF features, {len(np.unique(y))} unique disease classes.")
    print("-" * 80)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        
        results[name] = {
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        }
        print(f"[{name:15s}] -> Accuracy: {acc*100:6.2f}% | Precision: {prec*100:6.2f}% | Recall: {rec*100:6.2f}% | F1: {f1*100:6.2f}%")

    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_df = pd.DataFrame(results).T
    ax = metrics_df.plot(kind="bar", figsize=(11, 6), colormap="viridis", edgecolor="black")
    plt.title("Evaluation of Supervised Classification Algorithms (Unit V Benchmark)", fontsize=14, weight="bold", pad=15)
    plt.ylabel("Score (%)", fontsize=12)
    plt.xlabel("Machine Learning Algorithm", fontsize=12)
    plt.xticks(rotation=0, ha="center", fontsize=11)
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=8, rotation=45)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "ml_model_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"\n[+] Generated: {chart1_path}")

    # Plot 2: Random Forest Feature Importance
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_features = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    top_scores = importances[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores * 100, y=top_features, palette="mako")
    plt.title("Top 15 Most Important Symptom Features (Random Forest Gini Importance)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Feature Importance Percentage (%)", fontsize=11)
    plt.ylabel("Symptom Feature Token", fontsize=11)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "ml_feature_importance.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[+] Generated: {chart2_path}")

    # Plot 3: Confusion Matrix for Top 8 Frequent Classes
    top_8_classes = list(pd.Series(y_test).value_counts().head(8).index)
    if len(top_8_classes) >= 3:
        mask = y_test.isin(top_8_classes)
        y_test_sub = y_test[mask]
        preds_sub = rf_model.predict(X_test[mask.values])
        
        cm = confusion_matrix(y_test_sub, preds_sub, labels=top_8_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_8_classes, yticklabels=top_8_classes, cbar=True)
        plt.title("Confusion Matrix for Top Classes (Random Forest Classifier)", fontsize=13, weight="bold", pad=12)
        plt.xlabel("Predicted Condition", fontsize=11)
        plt.ylabel("Actual Documented Condition", fontsize=11)
        plt.xticks(rotation=40, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        chart3_path = os.path.join(output_dir, "ml_confusion_matrix.png")
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        print(f"[+] Generated: {chart3_path}")

    print("=" * 80)
    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
    plots_dir = os.path.join(base_dir, "static", "plots")
    run_benchmark(data_path, plots_dir)


if __name__ == "__main__":
    main()

