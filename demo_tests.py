"""
AyurHerb - Demonstration & Verification Script (demo_tests.py)
Author: Data Science Team
Description: Executes and displays the 7 required test cases from the specification.
"""

import sys
from recommendation_engine import AyurvedicRecommender, SIMILARITY_THRESHOLD
from safety import evaluate_safety

# UTF-8 encoding support
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_demo():
    print("=" * 80)
    print("AYURHERB – FINAL SYSTEM VERIFICATION & TEST BENCHMARK")
    print("=" * 80)
    
    recommender = AyurvedicRecommender()
    
    test_cases = [
        ("Case 1: Respiratory Common Cold", "I have cough and sore throat"),
        ("Case 2: Neurological / Headache", "I have headache"),
        ("Case 3: Febrile Condition", "I have fever"),
        ("Case 4: Chest / Pulmonary Congestion", "I have cough and chest congestion"),
        ("Case 5: Empty Input Validation", ""),
        ("Case 6: Completely Unrelated Query", "astrophysics black hole gravitational quantum engine"),
        ("Case 7: Serious Emergency Symptom", "I have severe difficulty breathing and crushing chest pain")
    ]

    for title, query in test_cases:
        print(f"\n>>> {title}")
        print(f"    User Query: '{query}'")
        
        # Step 1: Safety Triage Check
        safety = evaluate_safety(query)
        if safety["is_emergency"]:
            print("    [STATUS: EMERGENCY TRIAGE RED-FLAG TRIGGERED]")
            print(f"    - Emergency Alert: {safety['emergency_message']}")
            print(f"    - Triggered Indicators: {', '.join(safety['detected_flags'])}")
            print(f"    - Action Advice: {safety['advice']}")
            continue

        # Step 2: Empty input check
        if not query.strip():
            print("    [STATUS: INPUT VALIDATION PROMPT]")
            print("    - Prompt: Please enter or select at least one symptom to analyze.")
            continue

        # Step 3: Recommendation Engine
        results = recommender.recommend(query, top_k=3, threshold=SIMILARITY_THRESHOLD)
        if not results:
            print("    [STATUS: NO SUFFICIENTLY SIMILAR RECORD]")
            print("    - Result: No sufficiently similar record was found in the current dataset.")
        else:
            print(f"    [STATUS: {len(results)} MATCHING AYURVEDIC RECOMMENDATIONS FOUND]")
            for idx, res in enumerate(results, 1):
                print(f"    Match #{idx}: {res['disease']} (Hindi: {res['hindi_name']})")
                print(f"      * Symptom Similarity Score: {res['similarity_percentage']}% ({res['similarity_score']})")
                print(f"      * Documented Symptoms: {res['symptoms']}")
                print(f"      * Ayurvedic Herbs: {res['ayurvedic_herbs']}")
                print(f"      * Formulation: {res['formulation']}")
                print(f"      * Doshas: {res['doshas']}")
                print(f"      * Diet & Lifestyle: {res['diet_and_lifestyle']}")

    print("\n" + "=" * 80)
    print("ALL 7 DEMONSTRATION CASES EVALUATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
