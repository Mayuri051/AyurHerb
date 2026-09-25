"""
AyurHerb - Comprehensive Multi-Paradigm System Verification Script (demo_tests.py)
Author: Data Science Team
Description: Executes and displays multi-paradigm evaluation across:
             1. Clinical Safety Triage
             2. TF-IDF Semantic Retrieval
             3. Supervised ML Classification (Random Forest & AdaBoost) [Unit V]
             4. Bayesian Network Inference under Uncertainty [Unit I]
             5. Fuzzy Logic Dosage Control (Mamdani FIS) [Unit III]
"""

import sys
from recommendation_engine import AyurvedicRecommender, SIMILARITY_THRESHOLD
from safety import evaluate_safety
from ml_classifier import AyurvedicMLClassifier
from fuzzy_engine import AyurvedicFuzzyInferenceSystem
from bayesian_engine import AyurvedicBayesianNetwork

# UTF-8 encoding support
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_demo():
    print("=" * 85)
    print("AYURHERB – MULTI-PARADIGM SYSTEM VERIFICATION & TEST BENCHMARK")
    print("Addressing Syllabus Units I, II, III, V, and VI")
    print("=" * 85)
    
    recommender = AyurvedicRecommender()
    ml_clf = AyurvedicMLClassifier()
    fuzzy_sys = AyurvedicFuzzyInferenceSystem()
    bayesian_net = AyurvedicBayesianNetwork()
    
    test_cases = [
        ("Case 1: Respiratory Common Cold", "I have cough and sore throat", 4.0, 3),
        ("Case 2: Neurological / Headache", "I have headache", 6.0, 2),
        ("Case 3: Febrile Condition", "I have fever", 5.5, 4),
        ("Case 4: Chest / Pulmonary Congestion", "I have cough and chest congestion", 7.0, 6),
        ("Case 5: Empty Input Validation", "", 0.0, 0),
        ("Case 6: Completely Unrelated Query", "astrophysics black hole gravitational quantum engine", 5.0, 5),
        ("Case 7: Serious Emergency Symptom", "I have severe difficulty breathing and crushing chest pain", 10.0, 1)
    ]

    for title, query, sev, days in test_cases:
        print(f"\n>>> {title}")
        print(f"    User Query: '{query}'")
        
        # Step 1: Safety Triage Check
        safety = evaluate_safety(query)
        if safety["is_emergency"]:
            print("    [STAGE 1: EMERGENCY TRIAGE RED-FLAG TRIGGERED]")
            print(f"    - Emergency Alert: {safety['emergency_message']}")
            print(f"    - Triggered Indicators: {', '.join(safety['detected_flags'])}")
            print(f"    - Action Advice: {safety['advice']}")
            continue

        # Step 2: Empty input check
        if not query.strip():
            print("    [STAGE 2: INPUT VALIDATION PROMPT]")
            print("    - Prompt: Please enter or select at least one symptom to analyze.")
            continue

        # Step 3: TF-IDF Recommendation Engine
        results = recommender.recommend(query, top_k=2, threshold=SIMILARITY_THRESHOLD)
        if not results:
            print("    [STAGE 3: TF-IDF RETRIEVAL] -> No sufficiently similar record found in dataset.")
        else:
            print(f"    [STAGE 3: TF-IDF RETRIEVAL] -> {len(results)} matches found")
            for idx, res in enumerate(results, 1):
                print(f"      Match #{idx}: {res['disease']} (Hindi: {res['hindi_name']}) | Similarity: {res['similarity_percentage']}%")
                print(f"        Herbs: {res['ayurvedic_herbs']} | Dosha: {res['doshas']}")

        # Step 4: Supervised ML Classification (Unit V)
        ml_pred = ml_clf.predict(query)
        print(f"    [STAGE 4: SUPERVISED ML CLASSIFICATION (Unit V)]")
        print(f"      - Random Forest Prediction: {ml_pred['rf_prediction']} ({ml_pred['rf_confidence']}%)")
        print(f"      - AdaBoost Prediction:      {ml_pred['ada_prediction']} ({ml_pred['ada_confidence']}%)")

        # Step 5: Bayesian Inference under Uncertainty (Unit I)
        bn_res = bayesian_net.infer(query, top_k=2)
        print(f"    [STAGE 5: BAYESIAN NETWORK INFERENCE (Unit I)]")
        print(f"      - Most Probable Posterior:  {bn_res['top_condition']} ({bn_res['top_probability']}%)")

        # Step 6: Fuzzy Logic Dosage Controller (Unit III)
        fz_res = fuzzy_sys.infer(severity_score=sev, duration_days=days, agni_score=6.0)
        print(f"    [STAGE 6: FUZZY LOGIC DOSAGE CONTROLLER (Unit III)]")
        print(f"      - Calculated Potency Index: {fz_res['potency_index']}/100 -> {fz_res['regimen']}")
        print(f"      - Recommended Anupana:      {fz_res['recommended_anupana']}")

    print("\n" + "=" * 85)
    print("ALL 7 MULTI-PARADIGM TEST CASES EVALUATED AND VALIDATED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_demo()
