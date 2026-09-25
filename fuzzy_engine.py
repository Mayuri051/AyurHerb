"""
AyurHerb - Fuzzy Logic & Inference Engine (fuzzy_engine.py)
Author: Data Science Team
Description: Implements Fuzzy Sets, Membership Functions (Triangular/Trapezoidal),
             and a Mamdani Fuzzy Rule-Based Inference System for determining
             Ayurvedic Herbal Treatment Intensity and Dosage Potency.
             Addresses Unit III (Fuzzy Logic & Its Applications) of the syllabus.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """Triangular fuzzy membership function: trimf(x; a, b, c)."""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0


def trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal fuzzy membership function: trapmf(x; a, b, c, d)."""
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d > c else 1.0


class AyurvedicFuzzyInferenceSystem:
    """
    Fuzzy Control System for Ayurvedic Dosage & Formulation Potency Evaluation.
    
    Fuzzy Inputs:
      1. Symptom Severity (0 to 10): Mild, Moderate, Severe
      2. Condition Chronicity / Duration (1 to 60 days): Acute, Subacute, Chronic
      3. Patient Agni / Digestive Strength (0 to 10): Manda (Weak), Sama (Balanced), Tikshna (Intense)
    
    Fuzzy Output:
      - Herbal Treatment Potency Index (0 to 100): Mild (Mridu), Moderate (Madhyama), High (Tikshna)
      - Recommended Dosage Regimen & Formulation Strategy
    """

    def __init__(self):
        # Universe of Discourse for Output (Potency Index 0 to 100)
        self.output_universe = np.linspace(0, 100, 201)

    def fuzzify_severity(self, severity_score: float) -> Dict[str, float]:
        """Compute membership grades for Symptom Severity (0-10)."""
        return {
            "mild": trapezoidal_mf(severity_score, 0, 0, 2.5, 5.0),
            "moderate": triangular_mf(severity_score, 3.0, 5.5, 8.0),
            "severe": trapezoidal_mf(severity_score, 6.0, 8.0, 10.0, 10.0)
        }

    def fuzzify_chronicity(self, duration_days: float) -> Dict[str, float]:
        """Compute membership grades for Condition Chronicity (1-60 days)."""
        return {
            "acute": trapezoidal_mf(duration_days, 0, 1, 3, 7),
            "subacute": triangular_mf(duration_days, 5, 14, 25),
            "chronic": trapezoidal_mf(duration_days, 18, 30, 60, 100)
        }

    def fuzzify_agni(self, agni_score: float) -> Dict[str, float]:
        """Compute membership grades for Digestive Capacity / Agni (0-10)."""
        return {
            "manda": trapezoidal_mf(agni_score, 0, 0, 2.5, 5.0),     # Weak digestion
            "sama": triangular_mf(agni_score, 3.5, 5.5, 7.5),        # Balanced digestion
            "tikshna": trapezoidal_mf(agni_score, 6.5, 8.5, 10, 10)  # Strong / Hyperactive
        }

    def evaluate_rules(
        self,
        sev: Dict[str, float],
        chrn: Dict[str, float],
        agni: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate Mamdani Fuzzy Rules (using MIN for T-norm AND, MAX for S-norm OR).
        
        Rules:
        R1: IF Severity is Mild AND Chronicity is Acute THEN Potency is Mild
        R2: IF Severity is Mild AND Chronicity is Chronic THEN Potency is Moderate
        R3: IF Severity is Moderate AND Agni is Sama THEN Potency is Moderate
        R4: IF Severity is Moderate AND Agni is Manda THEN Potency is Mild
        R5: IF Severity is Severe AND Chronicity is Acute THEN Potency is High
        R6: IF Severity is Severe AND Chronicity is Chronic THEN Potency is High
        R7: IF Chronicity is Chronic AND Agni is Manda THEN Potency is Moderate
        R8: IF Severity is Severe AND Agni is Tikshna THEN Potency is High
        """
        r1 = min(sev["mild"], chrn["acute"])
        r2 = min(sev["mild"], chrn["chronic"])
        r3 = min(sev["moderate"], agni["sama"])
        r4 = min(sev["moderate"], agni["manda"])
        r5 = min(sev["severe"], chrn["acute"])
        r6 = min(sev["severe"], chrn["chronic"])
        r7 = min(chrn["chronic"], agni["manda"])
        r8 = min(sev["severe"], agni["tikshna"])

        # Aggregate Rule Consequents via MAX (Fuzzy Union)
        output_mild = max(r1, r4)
        output_moderate = max(r2, r3, r7)
        output_high = max(r5, r6, r8)

        return {
            "mild": float(output_mild),
            "moderate": float(output_moderate),
            "high": float(output_high)
        }

    def defuzzify(self, aggregated_levels: Dict[str, float]) -> float:
        """
        Defuzzification using Centroid (Center of Gravity) Method.
        $$\text{Centroid} = \frac{\sum \mu(x) \cdot x}{\sum \mu(x)}$$
        """
        numerator = 0.0
        denominator = 0.0

        for x in self.output_universe:
            # Output Membership values at point x
            mf_mild = trapezoidal_mf(x, 0, 0, 20, 45)
            mf_mod = triangular_mf(x, 25, 50, 75)
            mf_high = trapezoidal_mf(x, 55, 80, 100, 100)

            # Implication by clipping (Mamdani min)
            mu_mild = min(aggregated_levels["mild"], mf_mild)
            mu_mod = min(aggregated_levels["moderate"], mf_mod)
            mu_high = min(aggregated_levels["high"], mf_high)

            # Overall membership at x
            mu_x = max(mu_mild, mu_mod, mu_high)

            numerator += mu_x * x
            denominator += mu_x

        if denominator == 0:
            return 50.0  # Default neutral midpoint
        return float(numerator / denominator)

    def infer(
        self,
        severity_score: float = 5.0,
        duration_days: float = 7.0,
        agni_score: float = 5.5
    ) -> Dict[str, Any]:
        """
        Execute full Fuzzy Inference pipeline from crisp inputs to crisp output & clinical advice.
        """
        # 1. Fuzzification
        f_sev = self.fuzzify_severity(severity_score)
        f_chrn = self.fuzzify_chronicity(duration_days)
        f_agni = self.fuzzify_agni(agni_score)

        # 2. Rule Evaluation
        rule_outputs = self.evaluate_rules(f_sev, f_chrn, f_agni)

        # 3. Defuzzification (Centroid method)
        potency_index = round(self.defuzzify(rule_outputs), 2)

        # 4. Clinical Interpretation & Ayurvedic Mapping
        if potency_index < 35:
            regimen = "Mridu (Mild/Gentle Care)"
            dosage = "Low to standard dose (1-2g Churna or 250mg Vati twice daily with warm water)"
            formulation_type = "Swasa/Digestive Herbal Teas, Warm Infusions (Phant/Hima)"
            anupana = "Lukewarm Water (Ushnodaka)"
        elif potency_index < 68:
            regimen = "Madhyama (Standard Therapeutic Potency)"
            dosage = "Standard therapeutic dose (3-5g Churna or 500mg Vati twice daily before meals)"
            formulation_type = "Decoctions (Kwath), Compound Churna, Fermented Asava/Arishta"
            anupana = "Warm Water with Honey or Ginger Juice"
        else:
            regimen = "Tikshna (Intensive Clinical Support)"
            dosage = "Intensive managed regimen under clinical supervision (Taila/Ghritha/Rasayana)"
            formulation_type = "Concentrated Kwath, Medicated Ghee (Ghrita), Classical Rasayana"
            anupana = "Warm Milk / Medicated Cow's Ghee (Godugdha / Goghrita)"

        return {
            "potency_index": potency_index,
            "regimen": regimen,
            "dosage_advice": dosage,
            "recommended_formulation_type": formulation_type,
            "recommended_anupana": anupana,
            "fuzzy_memberships": {
                "severity": {k: round(v, 3) for k, v in f_sev.items()},
                "chronicity": {k: round(v, 3) for k, v in f_chrn.items()},
                "agni": {k: round(v, 3) for k, v in f_agni.items()}
            },
            "rule_activations": {k: round(v, 3) for k, v in rule_outputs.items()}
        }


if __name__ == "__main__":
    fis = AyurvedicFuzzyInferenceSystem()
    print("=" * 70)
    print("AYURHERB - FUZZY LOGIC DOSAGE INFERENCE SYSTEM TEST")
    print("=" * 70)
    
    test_scenarios = [
        {"name": "Mild Acute Cold", "sev": 2.5, "days": 3, "agni": 6.0},
        {"name": "Moderate Chronic Joint Pain", "sev": 6.0, "days": 30, "agni": 4.0},
        {"name": "Severe Acute Flare-up with High Agni", "sev": 8.5, "days": 2, "agni": 8.0}
    ]

    for sc in test_scenarios:
        res = fis.infer(sc["sev"], sc["days"], sc["agni"])
        print(f"\nScenario: {sc['name']}")
        print(f"  Inputs -> Severity: {sc['sev']}, Duration: {sc['days']} days, Agni: {sc['agni']}")
        print(f"  Output -> Fuzzy Potency Index: {res['potency_index']}/100")
        print(f"  Regimen: {res['regimen']}")
        print(f"  Dosage: {res['dosage_advice']}")
        print(f"  Carrier (Anupana): {res['recommended_anupana']}")

