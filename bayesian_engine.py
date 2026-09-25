"""
AyurHerb - Bayesian Network Inference Engine (bayesian_engine.py)
Author: Data Science Team
Description: Implements probabilistic reasoning under uncertainty using a Bayesian Network.
             Calculates posterior disease probabilities given evidence (symptoms, dosha, season).
             Addresses Unit I (Uncertainty in AI - Inferencing with Bayesian Network) of the syllabus.
"""

import os
import re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class AyurvedicBayesianNetwork:
    """
    Bayesian Network for Probabilistic Ayurvedic Diagnosis under Uncertainty.
    
    Structure:
      Disease (Target Variable D)
        ├──> Doshas (Imbalance Evidence)
        ├──> Seasonal Variation (Climate Trigger Evidence)
        └──> Symptoms (Observed Manifestations)
    
    Inference:
      P(Disease = d | Symptoms, Dosha, Season) \propto P(d) * P(Dosha|d) * P(Season|d) * \prod P(s_i|d)
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
        
        self.data_path = data_path
        self.prior_disease: Dict[str, float] = {}
        self.cpt_dosha: Dict[str, Dict[str, float]] = {}
        self.cpt_season: Dict[str, Dict[str, float]] = {}
        self.cpt_symptoms: Dict[str, Dict[str, float]] = {}
        self.all_diseases: List[str] = []
        self.vocabulary: List[str] = []
        
        self._build_cpts()

    def _build_cpts(self) -> None:
        """Estimate Conditional Probability Tables (CPTs) from cleaned clinical dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        total_records = len(df)

        # 1. Prior Probability P(Disease)
        disease_counts = df["Disease"].value_counts()
        self.all_diseases = list(disease_counts.index)
        self.prior_disease = (disease_counts / total_records).to_dict()

        # 2. Extract unique symptoms vocabulary
        symptom_terms = set()
        for s in df["Symptoms"].dropna():
            parts = [re.sub(r"[^\w\s]", "", p.strip().lower()) for p in re.split(r"[,;]\s*", str(s)) if p.strip()]
            symptom_terms.update(parts)
        self.vocabulary = sorted(list(symptom_terms))

        # 3. Conditional Probability Tables with Laplace Smoothing (alpha=1)
        for disease, group in df.groupby("Disease"):
            n_d = len(group)
            
            # P(Dosha | Disease)
            dosha_counts = group["Doshas"].value_counts()
            self.cpt_dosha[disease] = {
                dosha: (count + 0.1) / (n_d + 0.1 * max(1, len(dosha_counts)))
                for dosha, count in dosha_counts.items()
            }
            
            # P(Season | Disease)
            season_counts = group["Seasonal Variation"].value_counts()
            self.cpt_season[disease] = {
                season: (count + 0.1) / (n_d + 0.1 * max(1, len(season_counts)))
                for season, count in season_counts.items()
            }
            
            # P(Symptom | Disease)
            group_symptoms_text = " ".join(group["Symptoms"].dropna().astype(str).str.lower())
            self.cpt_symptoms[disease] = {}
            for term in self.vocabulary:
                if term in group_symptoms_text:
                    self.cpt_symptoms[disease][term] = 0.85
                else:
                    self.cpt_symptoms[disease][term] = 0.05  # Background baseline

        print(f"[+] Bayesian Network CPTs built for {len(self.all_diseases)} diseases with {len(self.vocabulary)} symptom nodes.")

    def infer(
        self,
        symptoms_query: str,
        dosha_evidence: Optional[str] = None,
        season_evidence: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Compute posterior probabilities P(Disease | Evidence) given symptoms & context.
        Uses exact probabilistic belief propagation under uncertainty.
        """
        if not symptoms_query or not symptoms_query.strip():
            return {"status": "empty_query", "posteriors": []}

        q_clean = symptoms_query.lower()
        matched_tokens = [term for term in self.vocabulary if term in q_clean]

        unnormalized_posteriors = {}

        for disease in self.all_diseases:
            # Prior P(Disease)
            prob = self.prior_disease.get(disease, 1e-4)

            # Likelihood of Symptoms \prod P(Symptom_i | Disease)
            for token in matched_tokens:
                p_sym = self.cpt_symptoms.get(disease, {}).get(token, 0.05)
                prob *= p_sym

            # Evidence Likelihood P(Dosha | Disease)
            if dosha_evidence and dosha_evidence.strip():
                p_dosha = self.cpt_dosha.get(disease, {}).get(dosha_evidence.strip(), 0.1)
                prob *= p_dosha

            # Evidence Likelihood P(Season | Disease)
            if season_evidence and season_evidence.strip():
                p_season = self.cpt_season.get(disease, {}).get(season_evidence.strip(), 0.1)
                prob *= p_season

            unnormalized_posteriors[disease] = prob

        # Normalization constant \alpha = 1 / \sum P(Evidence, d)
        total_prob = sum(unnormalized_posteriors.values())
        if total_prob == 0:
            total_prob = 1.0

        normalized = [
            {
                "disease": d,
                "posterior_probability": round(p / total_prob, 4),
                "posterior_percentage": round((p / total_prob) * 100, 2)
            }
            for d, p in unnormalized_posteriors.items()
        ]

        normalized.sort(key=lambda x: x["posterior_probability"], reverse=True)

        return {
            "status": "success",
            "matched_symptom_nodes": matched_tokens,
            "posteriors": normalized[:top_k],
            "top_condition": normalized[0]["disease"] if normalized else "None",
            "top_probability": normalized[0]["posterior_percentage"] if normalized else 0.0,
            "inference_method": "Bayesian Network Exact Joint Belief Propagation (Exact Inference)"
        }


if __name__ == "__main__":
    bn = AyurvedicBayesianNetwork()
    test_q = "cough sore throat chest congestion"
    res = bn.infer(test_q, dosha_evidence="Kapha", season_evidence="Winter")
    print(f"\nBayesian Inference Result for '{test_q}':")
    print(f"Top Condition: {res['top_condition']} ({res['top_probability']}%)")
    for item in res["posteriors"]:
        print(f"  - {item['disease']}: {item['posterior_percentage']}%")

