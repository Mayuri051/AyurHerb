"""
Practical Experiment 2: Building a Cognitive Healthcare Application
Syllabus Unit: Unit II - Cognitive Computing (LO2)
Description: Demonstrates a cognitive computing pipeline for clinical symptom
             comprehension, multi-modal reasoning, and Ayurvedic decision support.
"""

import re
from typing import Dict, Any, List


class CognitiveHealthcareSystem:
    """
    Cognitive Computing architecture for Healthcare Decision Support:
    1. Perception & Natural Language Understanding (Symptom extraction)
    2. Knowledge Representation & Ontology (Dosha & Herb associations)
    3. Hypothesis Generation & Reasoning (Semantic Matching & Confidence Scoring)
    4. Safety Guardrails & Triage (Emergency Red-Flags)
    """

    def __init__(self):
        self.emergency_keywords = ["chest pain", "breathing difficulty", "severe bleeding", "unconscious", "stroke"]
        self.knowledge_base = {
            "respiratory_imbalance": {
                "symptoms": ["cough", "cold", "sore throat", "congestion", "sneezing"],
                "dosha": "Kapha-Vata",
                "herbs": ["Tulsi", "Adhatoda Vasica (Vasa)", "Ginger", "Licorice (Mulethi)"],
                "lifestyle": "Steam inhalation, warm fluids, avoid dairy during acute phase."
            },
            "digestive_imbalance": {
                "symptoms": ["acidity", "indigestion", "bloating", "heartburn", "constipation"],
                "dosha": "Pitta-Vata",
                "herbs": ["Amla", "Triphala", "Fennel", "Cumin"],
                "lifestyle": "Regular meal timing, warm cooked meals, avoid excessively pungent foods."
            },
            "neurological_stress": {
                "symptoms": ["headache", "migraine", "insomnia", "stress", "anxiety"],
                "dosha": "Vata-Pitta",
                "herbs": ["Brahmi", "Ashwagandha", "Shankhpushpi"],
                "lifestyle": "Shirodhara, Pranayama (Anulom Vilom), regular circadian sleep."
            }
        }

    def perceive_and_triage(self, query: str) -> Dict[str, Any]:
        """Check for emergency triggers."""
        q_lower = query.lower()
        for red_flag in self.emergency_keywords:
            if red_flag in q_lower:
                return {
                    "is_emergency": True,
                    "alert": f"CRITICAL: '{red_flag}' detected. Immediate emergency medical intervention required!"
                }
        return {"is_emergency": False, "alert": "Triage passed: Non-critical symptoms."}

    def cognitive_reason(self, query: str) -> Dict[str, Any]:
        """Cognitive hypothesis generation and reasoning."""
        triage = self.perceive_and_triage(query)
        if triage["is_emergency"]:
            return triage

        q_tokens = set(re.findall(r"\w+", query.lower()))
        hypotheses = []

        for category, info in self.knowledge_base.items():
            sym_set = set(info["symptoms"])
            matched = q_tokens.intersection(sym_set)
            if matched:
                confidence = len(matched) / len(sym_set)
                hypotheses.append({
                    "category": category,
                    "confidence_score": round(confidence, 2),
                    "confidence_pct": round(confidence * 100, 1),
                    "matched_symptoms": list(matched),
                    "dosha": info["dosha"],
                    "recommended_herbs": info["herbs"],
                    "lifestyle_guidelines": info["lifestyle"]
                })

        hypotheses.sort(key=lambda x: x["confidence_score"], reverse=True)
        return {
            "is_emergency": False,
            "query": query,
            "hypotheses_count": len(hypotheses),
            "best_hypothesis": hypotheses[0] if hypotheses else None,
            "all_hypotheses": hypotheses
        }


def main():
    print("=" * 70)
    print("EXPERIMENT 2: COGNITIVE HEALTHCARE APPLICATION (UNIT II)")
    print("=" * 70)

    cognitive_app = CognitiveHealthcareSystem()

    queries = [
        "I have severe cough, cold, and sore throat since yesterday",
        "Experiencing stomach acidity, indigestion and heartburn after eating",
        "Patient has sudden severe chest pain and breathing difficulty"
    ]

    for q in queries:
        print(f"\nUser Query: '{q}'")
        res = cognitive_app.cognitive_reason(q)
        if res.get("is_emergency"):
            print(f"  [STATUS: EMERGENCY RED-FLAG] -> {res['alert']}")
        else:
            best = res["best_hypothesis"]
            if best:
                print(f"  [Cognitive Match]: {best['category'].upper()} (Confidence: {best['confidence_pct']}%)")
                print(f"  - Matched Symptoms: {', '.join(best['matched_symptoms'])}")
                print(f"  - Dosha Imbalance: {best['dosha']}")
                print(f"  - Ayurvedic Herbs: {', '.join(best['recommended_herbs'])}")
                print(f"  - Lifestyle Advice: {best['lifestyle_guidelines']}")
            else:
                print("  No direct cognitive match found.")

    print("\n" + "=" * 70)
    print("Cognitive Healthcare Experiment Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

