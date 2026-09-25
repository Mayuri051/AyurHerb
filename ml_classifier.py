"""
AyurHerb - Machine Learning Classification Module (ml_classifier.py)
Author: Data Science Team
Description: Implements supervised Machine Learning models (Random Forest and AdaBoost)
             for automated Ayurvedic disease and dosha classification based on symptom inputs.
             Addresses Unit V (Supervised Learning & Ensemble Methods) and Unit VI of syllabus.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping punctuation."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class AyurvedicMLClassifier:
    """
    Supervised Machine Learning Ensemble Classifier for Ayurvedic Diagnosis.
    Trained with Random Forest and AdaBoost on symptom feature matrices.
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
        
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=2500
        )
        self.rf_model = None
        self.adaboost_model = None
        self.classes_ = []
        self.metrics_ = {}
        self.is_trained = False
        
        self._train_models()

    def _train_models(self) -> None:
        """Load dataset, extract features, and train Random Forest & AdaBoost."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Filter classes with at least 1 record
        df["Symptoms_Clean"] = df["Symptoms"].fillna("").apply(clean_text)
        df["Unified_Features"] = (
            df["Symptoms_Clean"] + " " +
            df["Doshas"].fillna("").apply(clean_text) + " " +
            df["Symptom Severity"].fillna("").apply(clean_text)
        )
        
        X_raw = df["Unified_Features"]
        y_raw = df["Disease"].astype(str).str.strip()

        # Vectorize text features
        X = self.vectorizer.fit_transform(X_raw)
        self.classes_ = sorted(list(y_raw.unique()))
        
        # Train Random Forest Classifier
        self.rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            random_state=42,
            class_weight="balanced"
        )
        self.rf_model.fit(X, y_raw)

        # Train AdaBoost Classifier (with Decision Tree base estimator)
        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.adaboost_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=80,
            learning_rate=0.8,
            random_state=42
        )
        self.adaboost_model.fit(X, y_raw)

        # Evaluate models on the dataset
        rf_preds = self.rf_model.predict(X)
        ada_preds = self.adaboost_model.predict(X)

        self.metrics_ = {
            "rf_accuracy": float(accuracy_score(y_raw, rf_preds)),
            "ada_accuracy": float(accuracy_score(y_raw, ada_preds)),
            "n_samples": int(len(df)),
            "n_classes": int(len(self.classes_)),
            "n_features": int(X.shape[1])
        }
        self.is_trained = True
        print(f"[+] Ayurvedic ML Classifier Trained successfully:")
        print(f"    - Random Forest Accuracy: {self.metrics_['rf_accuracy'] * 100:.2f}%")
        print(f"    - AdaBoost Accuracy: {self.metrics_['ada_accuracy'] * 100:.2f}%")
        print(f"    - Classes: {self.metrics_['n_classes']}, Features: {self.metrics_['n_features']}")

    def predict(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Classify a user symptom query using Random Forest and AdaBoost.
        Returns predicted condition, confidence score, and top-K ranked alternatives.
        """
        if not query or not query.strip() or not self.is_trained:
            return {
                "status": "invalid_input",
                "rf_prediction": None,
                "rf_confidence": 0.0,
                "ada_prediction": None,
                "ada_confidence": 0.0,
                "top_predictions": []
            }

        cleaned = clean_text(query)
        vec = self.vectorizer.transform([cleaned])

        # Random Forest Probabilities
        rf_probs = self.rf_model.predict_proba(vec)[0]
        rf_top_idx = np.argsort(rf_probs)[::-1][:top_k]

        top_candidates = []
        for idx in rf_top_idx:
            prob = float(rf_probs[idx])
            top_candidates.append({
                "disease": self.rf_model.classes_[idx],
                "confidence_score": round(prob, 4),
                "confidence_percentage": round(prob * 100, 1)
            })

        # AdaBoost Probabilities
        ada_probs = self.adaboost_model.predict_proba(vec)[0]
        ada_top_idx = np.argmax(ada_probs)
        ada_pred = self.adaboost_model.classes_[ada_top_idx]
        ada_conf = float(ada_probs[ada_top_idx])

        return {
            "status": "success",
            "rf_prediction": top_candidates[0]["disease"] if top_candidates else "Unknown",
            "rf_confidence": top_candidates[0]["confidence_percentage"] if top_candidates else 0.0,
            "ada_prediction": ada_pred,
            "ada_confidence": round(ada_conf * 100, 1),
            "top_predictions": top_candidates,
            "algorithm_used": "Random Forest Ensemble + AdaBoost Classifier"
        }

    def get_feature_importance(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """Return the most important symptom features learned by the Random Forest."""
        if not self.is_trained or self.rf_model is None:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        results = []
        for idx in indices:
            results.append({
                "feature": feature_names[idx],
                "importance": round(float(importances[idx]), 4),
                "importance_percentage": round(float(importances[idx]) * 100, 2)
            })
        return results


if __name__ == "__main__":
    classifier = AyurvedicMLClassifier()
    test_queries = [
        "cough and fever with sore throat",
        "severe headache migraine and nausea",
        "indigestion acidity and stomach bloating",
        "knee joint pain stiffness"
    ]
    print("\n--- Model Classification Test ---")
    for q in test_queries:
        res = classifier.predict(q)
        print(f"Query: '{q}'")
        print(f"  -> Random Forest: {res['rf_prediction']} ({res['rf_confidence']}%)")
        print(f"  -> AdaBoost: {res['ada_prediction']} ({res['ada_confidence']}%)")

