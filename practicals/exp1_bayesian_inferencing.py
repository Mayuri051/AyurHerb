"""
Practical Experiment 1: Inferencing with Bayesian Network in Python
Syllabus Unit: Unit I - Uncertainty in AI (LO1)
Description: Demonstrates Bayesian belief networks and probabilistic inference
             calculating disease and symptom probabilities under uncertainty.
"""

import numpy as np


class BayesianNetworkExperiment:
    """
    Demonstrates Bayesian Network Inference for Healthcare Diagnosis.
    Variables:
      - Fever (F) in {0, 1}
      - Cough (C) in {0, 1}
      - Fatigue (FT) in {0, 1}
      - Disease (D) in {'Flu', 'Cold', 'Healthy'}
    """

    def __init__(self):
        # 1. Prior Probabilities P(Disease)
        self.p_disease = {
            "Flu": 0.10,
            "Cold": 0.30,
            "Healthy": 0.60
        }

        # 2. Conditional Probability Tables P(Symptom = 1 | Disease)
        self.cpt_fever = {"Flu": 0.90, "Cold": 0.30, "Healthy": 0.02}
        self.cpt_cough = {"Flu": 0.80, "Cold": 0.85, "Healthy": 0.05}
        self.cpt_fatigue = {"Flu": 0.85, "Cold": 0.40, "Healthy": 0.10}

    def infer_disease_probability(self, has_fever=True, has_cough=True, has_fatigue=False):
        """
        Compute P(Disease | Evidence) using Bayes' Theorem:
        P(D | F, C, FT) = alpha * P(D) * P(F|D) * P(C|D) * P(FT|D)
        """
        posteriors = {}
        for disease, p_d in self.p_disease.items():
            # Likelihood computation
            p_f = self.cpt_fever[disease] if has_fever else (1 - self.cpt_fever[disease])
            p_c = self.cpt_cough[disease] if has_cough else (1 - self.cpt_cough[disease])
            p_ft = self.cpt_fatigue[disease] if has_fatigue else (1 - self.cpt_fatigue[disease])

            # Joint Probability P(Disease, Evidence)
            joint_prob = p_d * p_f * p_c * p_ft
            posteriors[disease] = joint_prob

        # Normalization (alpha)
        total = sum(posteriors.values())
        normalized = {k: round(v / total, 4) for k, v in posteriors.items()}
        return normalized


def main():
    print("=" * 70)
    print("EXPERIMENT 1: INFERENCING WITH BAYESIAN NETWORK (UNIT I)")
    print("=" * 70)

    bn = BayesianNetworkExperiment()

    evidence_cases = [
        {"desc": "Case A: Patient has Fever and Cough (no severe Fatigue)", "f": True, "c": True, "ft": False},
        {"desc": "Case B: Patient has Fever, Cough, AND severe Fatigue", "f": True, "c": True, "ft": True},
        {"desc": "Case C: Patient has No Fever, No Cough, but mild Fatigue", "f": False, "c": False, "ft": True},
    ]

    for case in evidence_cases:
        print(f"\n{case['desc']}")
        posteriors = bn.infer_disease_probability(case["f"], case["c"], case["ft"])
        for disease, prob in posteriors.items():
            print(f"  P(Disease = {disease:7s} | Evidence) = {prob * 100:6.2f}%")
        
        most_likely = max(posteriors, key=posteriors.get)
        print(f"  --> Most Probable Diagnosis: {most_likely} ({posteriors[most_likely]*100:.2f}%)")

    print("\n" + "=" * 70)
    print("Bayesian Network Inference Experiment Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

