"""
Practical Experiment 12: Mini-Project Pipeline (AyurHerb Data Science System)
Syllabus Unit: Unit VI - Mini-project on trends and applications in Data Science (LO4, LO5)
Description: Demonstrates the end-to-end multi-paradigm Data Science pipeline of AyurHerb:
             1. NLP Preprocessing & Tokenization
             2. Random Forest & AdaBoost Supervised ML Classification
             3. Fuzzy Logic Dosage Controller
             4. Bayesian Network Probabilistic Inference
             5. Clinical Safety & Triage Guardrails
"""

import os
import sys

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ml_classifier import AyurvedicMLClassifier
from fuzzy_engine import AyurvedicFuzzyInferenceSystem
from bayesian_engine import AyurvedicBayesianNetwork
from safety import evaluate_safety


def run_full_pipeline(symptom_query: str, severity_score: float = 6.0, duration_days: int = 5):
    print("=" * 75)
    print(f"AYURHERB MINI-PROJECT PIPELINE EXECUTION")
    print(f"Input Query: '{symptom_query}' | Severity: {severity_score}/10 | Duration: {duration_days} days")
    print("=" * 75)

    # 1. Safety Triage
    safety = evaluate_safety(symptom_query)
    if safety["is_emergency"]:
        print(f"[STAGE 1: EMERGENCY SAFETY ALERT TRIGGERED]\n  Alert: {safety['emergency_message']}")
        return

    print("[STAGE 1: SAFETY TRIAGE] -> PASSED (Non-emergency)")

    # 2. Supervised ML Classification (Random Forest & AdaBoost)
    ml = AyurvedicMLClassifier()
    ml_res = ml.predict(symptom_query)
    print("\n[STAGE 2: SUPERVISED ML CLASSIFICATION (Unit V)]")
    print(f"  * Random Forest Prediction: {ml_res['rf_prediction']} (Confidence: {ml_res['rf_confidence']}%)")
    print(f"  * AdaBoost Prediction:      {ml_res['ada_prediction']} (Confidence: {ml_res['ada_confidence']}%)")

    # 3. Bayesian Network Inference (Unit I)
    bn = AyurvedicBayesianNetwork()
    bn_res = bn.infer(symptom_query)
    print("\n[STAGE 3: BAYESIAN INFERENCE UNDER UNCERTAINTY (Unit I)]")
    print(f"  * Posterior Top Condition: {bn_res['top_condition']} ({bn_res['top_probability']}%)")

    # 4. Fuzzy Logic Dosage Control (Unit III)
    fis = AyurvedicFuzzyInferenceSystem()
    fuzzy_res = fis.infer(severity_score=severity_score, duration_days=duration_days, agni_score=6.0)
    print("\n[STAGE 4: FUZZY LOGIC DOSAGE CONTROLLER (Unit III)]")
    print(f"  * Calculated Potency Index: {fuzzy_res['potency_index']}/100")
    print(f"  * Dosage Regimen:           {fuzzy_res['regimen']}")
    print(f"  * Recommended Anupana:      {fuzzy_res['recommended_anupana']}")

    print("\n" + "=" * 75)
    print("Mini-Project Multi-Paradigm Pipeline Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_full_pipeline("I have cough and sore throat with mild fever", severity_score=5.5, duration_days=4)

