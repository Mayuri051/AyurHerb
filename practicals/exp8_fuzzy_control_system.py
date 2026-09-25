"""
Practical Experiment 8: Design of a Fuzzy Control System
Syllabus Unit: Unit III - Fuzzy Logic & Its Applications (LO3)
Description: Designs a complete Mamdani Fuzzy Logic Controller with Fuzzification,
             Inference Engine (Rule Base), and Centroid Defuzzification for automated
             therapeutic dosage / herbal potency control.
"""

import numpy as np


class FuzzyControlSystem:
    """
    Mamdani Fuzzy Logic Controller for Herbal Formulation Potency Control.
    
    Inputs:
      - Symptom Severity S in [0, 10]
      - Patient Age A in [0, 100]
    Output:
      - Dosage Potency P in [0, 100]%
    """

    def __init__(self):
        self.u_output = np.linspace(0, 100, 200)

    # Membership functions
    def trimf(self, x, a, b, c):
        return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

    def trapmf(self, x, a, b, c, d):
        return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))

    def fuzzify(self, severity, age):
        """Fuzzify input values."""
        # Severity: Low, Medium, High
        sev_low = self.trapmf(severity, 0, 0, 2, 4)
        sev_med = self.trimf(severity, 3, 5, 7)
        sev_high = self.trapmf(severity, 6, 8, 10, 10)

        # Age: Child/Elderly (Sensitive), Adult (Robust)
        age_sensitive = max(self.trapmf(age, 0, 0, 12, 18), self.trapmf(age, 60, 70, 100, 100))
        age_adult = self.trimf(age, 16, 38, 65)

        return {
            "sev": {"low": sev_low, "med": sev_med, "high": sev_high},
            "age": {"sensitive": age_sensitive, "adult": age_adult}
        }

    def infer(self, severity, age):
        """Evaluate Fuzzy Rules and compute Defuzzified Centroid Output."""
        f = self.fuzzify(severity, age)

        # Rule Base:
        # R1: IF Severity is Low THEN Potency is Low
        # R2: IF Severity is Med AND Age is Adult THEN Potency is Medium
        # R3: IF Severity is Med AND Age is Sensitive THEN Potency is Low
        # R4: IF Severity is High AND Age is Adult THEN Potency is High
        # R5: IF Severity is High AND Age is Sensitive THEN Potency is Medium
        r1 = f["sev"]["low"]
        r2 = min(f["sev"]["med"], f["age"]["adult"])
        r3 = min(f["sev"]["med"], f["age"]["sensitive"])
        r4 = min(f["sev"]["high"], f["age"]["adult"])
        r5 = min(f["sev"]["high"], f["age"]["sensitive"])

        out_low = max(r1, r3)
        out_med = max(r2, r5)
        out_high = r4

        # Centroid Defuzzification:
        num = 0.0
        den = 0.0
        for x in self.u_output:
            mf_l = self.trapmf(x, 0, 0, 20, 40)
            mf_m = self.trimf(x, 30, 50, 70)
            mf_h = self.trapmf(x, 60, 80, 100, 100)

            clip_l = min(out_low, mf_l)
            clip_m = min(out_med, mf_m)
            clip_h = min(out_high, mf_h)

            mu = max(clip_l, clip_m, clip_h)
            num += mu * x
            den += mu

        potency = (num / den) if den > 0 else 50.0
        return round(float(potency), 2), {"low": out_low, "med": out_med, "high": out_high}


def main():
    print("=" * 70)
    print("EXPERIMENT 8: FUZZY CONTROL SYSTEM (UNIT III)")
    print("=" * 70)

    fcs = FuzzyControlSystem()

    test_patients = [
        {"desc": "Child (Age 8) with Mild Fever (Severity 3.0)", "sev": 3.0, "age": 8},
        {"desc": "Adult (Age 32) with Moderate Pain (Severity 5.5)", "sev": 5.5, "age": 32},
        {"desc": "Elderly (Age 75) with Severe Cough (Severity 8.5)", "sev": 8.5, "age": 75},
        {"desc": "Adult (Age 28) with High Severity (Severity 9.0)", "sev": 9.0, "age": 28},
    ]

    for p in test_patients:
        potency, activations = fcs.infer(p["sev"], p["age"])
        print(f"\n{p['desc']}:")
        print(f"  -> Calculated Dosage Potency: {potency}%")
        print(f"  -> Rule Activation: Low={activations['low']:.2f}, Med={activations['med']:.2f}, High={activations['high']:.2f}")

    print("\n" + "=" * 70)
    print("Fuzzy Control System Simulation Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

