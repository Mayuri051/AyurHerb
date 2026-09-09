"""
AyurHerb - Recommendation Engine (recommendation_engine.py)
Author: Data Science Team
Description: Implements TF-IDF vectorization and Cosine Similarity to recommend
             Ayurvedic wellness remedies, herbs, and lifestyle guidelines based on user symptoms.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure console supports UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Configurable similarity threshold
SIMILARITY_THRESHOLD = 0.12  # Threshold for considering a match relevant


def clean_query_text(text: str) -> str:
    """
    Clean and normalize user query or symptom string.
    """
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower().strip()
    # Remove common punctuation but preserve spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AyurvedicRecommender:
    """
    Data-driven Ayurvedic Recommendation Engine using TF-IDF & Cosine Similarity.
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
        
        self.data_path = data_path
        self.df = self._load_data()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _load_data(self) -> pd.DataFrame:
        """Load the cleaned dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Cleaned dataset not found at {self.data_path}. Please run preprocess.py first."
            )
        df = pd.read_csv(self.data_path)
        # Ensure string types on text columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("")
        return df

    def _build_index(self) -> None:
        """
        Build TF-IDF Vectorizer and Matrix on symptoms & unified context.
        Uses sublinear TF scaling and word (1, 2) n-grams for capturing composite symptoms (e.g. 'sore throat').
        """
        # Primary feature: Symptoms + Disease + Doshas for balanced semantic matching
        corpus = self.df["Symptoms"].astype(str) + " " + self.df["Disease"].astype(str) + " " + self.df["Doshas"].astype(str)
        corpus_cleaned = [clean_query_text(t) for t in corpus]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_cleaned)
        print(f"Recommender index initialized with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} TF-IDF features.")

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Analyze user-entered symptoms and return ranked Ayurvedic recommendations.
        """
        cleaned_query = clean_query_text(query)
        if not cleaned_query:
            return []

        # Vectorize user input
        query_vec = self.vectorizer.transform([cleaned_query])

        # Compute cosine similarity
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get sorted indices in descending order
        ranked_indices = np.argsort(cosine_scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(cosine_scores[idx])
            if score < threshold:
                break  # Reached below relevance threshold

            row = self.df.iloc[idx]
            
            # Find matching symptom tokens for user feedback
            query_words = set(cleaned_query.split())
            dataset_symptoms = clean_query_text(str(row["Symptoms"])).split()
            matched_terms = list(query_words.intersection(set(dataset_symptoms)))

            result_item = {
                "disease": row.get("Disease", "General Condition"),
                "hindi_name": row.get("Hindi Name", "N/A"),
                "marathi_name": row.get("Marathi Name", "N/A"),
                "similarity_score": round(score, 4),
                "similarity_percentage": round(score * 100, 1),
                "symptoms": row.get("Symptoms", "Not specified"),
                "matched_terms": matched_terms,
                "symptom_severity": row.get("Symptom Severity", "Moderate"),
                "duration_of_treatment": row.get("Duration of Treatment", "As advised"),
                "ayurvedic_herbs": row.get("Ayurvedic Herbs", "Herbal consultation recommended"),
                "herbal_remedies": row.get("Herbal/Alternative Remedies", "Lifestyle modification"),
                "formulation": row.get("Formulation", "Traditional preparation"),
                "doshas": row.get("Doshas", "Tridosha"),
                "prakriti": row.get("Constitution/Prakriti", "General"),
                "diet_and_lifestyle": row.get("Diet and Lifestyle Recommendations", "Healthy balanced diet"),
                "yoga_and_therapy": row.get("Yoga & Physical Therapy", "Gentle breathing exercises"),
                "prevention": row.get("Prevention", "Daily wellness routine (Dinacharya)"),
                "medical_intervention": row.get("Medical Intervention", "Consult a physician if needed"),
                "complications": row.get("Complications", "None noted if managed"),
                "allergies": row.get("Allergies (Food/Env)", "None reported"),
                "patient_recommendations": row.get("Patient Recommendations", "Rest and hydration"),
                "seasonal_variation": row.get("Seasonal Variation", "All seasons"),
                "age_group": row.get("Age Group", "All ages"),
                "source": "AyurGenixAI Clinical Knowledge Dataset"
            }
            results.append(result_item)
            if len(results) >= top_k:
                break

        return results

    def get_all_herbs(self) -> List[Dict[str, Any]]:
        """
        Extract unique list of Ayurvedic herbs with aggregated conditions and details.
        """
        herb_map = {}
        for _, row in self.df.iterrows():
            herbs_raw = str(row.get("Ayurvedic Herbs", ""))
            items = [h.strip().title() for h in re.split(r"[,;/]\s*", herbs_raw) if h.strip() and "consult" not in h.lower()]
            for herb in items:
                if herb not in herb_map:
                    herb_map[herb] = {
                        "name": herb,
                        "conditions": set(),
                        "doshas": set(),
                        "formulations": set(),
                        "remedies": set(),
                        "count": 0
                    }
                herb_map[herb]["conditions"].add(row.get("Disease", ""))
                herb_map[herb]["doshas"].add(row.get("Doshas", ""))
                herb_map[herb]["formulations"].add(row.get("Formulation", ""))
                herb_map[herb]["remedies"].add(row.get("Herbal/Alternative Remedies", ""))
                herb_map[herb]["count"] += 1

        herb_list = []
        for name, data in herb_map.items():
            herb_list.append({
                "name": name,
                "conditions": list(data["conditions"])[:5],
                "doshas": ", ".join(list(data["doshas"])[:3]),
                "formulations": ", ".join(list(data["formulations"])[:3]),
                "remedies": ", ".join(list(data["remedies"])[:2]),
                "count": data["count"]
            })
        herb_list.sort(key=lambda x: x["count"], reverse=True)
        return herb_list

    def get_herb_detail(self, herb_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information and matching records for a specific herb.
        """
        target = herb_name.lower().strip()
        matching_rows = []
        for _, row in self.df.iterrows():
            herbs_str = str(row.get("Ayurvedic Herbs", "")).lower()
            if target in herbs_str:
                matching_rows.append(row)

        if not matching_rows:
            return None

        # Aggregate information
        conditions = list(set([r["Disease"] for r in matching_rows]))
        doshas = list(set([r["Doshas"] for r in matching_rows if r["Doshas"]]))
        formulations = list(set([r["Formulation"] for r in matching_rows if r["Formulation"]]))
        remedies = list(set([r["Herbal/Alternative Remedies"] for r in matching_rows if r["Herbal/Alternative Remedies"]]))
        diets = list(set([r["Diet and Lifestyle Recommendations"] for r in matching_rows if r["Diet and Lifestyle Recommendations"]]))
        yogas = list(set([r["Yoga & Physical Therapy"] for r in matching_rows if r["Yoga & Physical Therapy"]]))

        return {
            "name": herb_name.title(),
            "occurrences": len(matching_rows),
            "conditions": conditions,
            "doshas": doshas,
            "formulations": formulations,
            "remedies": remedies,
            "diets": diets[:3],
            "yogas": yogas[:3],
            "sample_records": [
                {
                    "disease": r["Disease"],
                    "symptoms": r["Symptoms"],
                    "severity": r["Symptom Severity"],
                    "formulation": r["Formulation"]
                }
                for r in matching_rows[:5]
            ]
        }

    def get_all_conditions(self) -> List[Dict[str, Any]]:
        """
        Get all unique conditions/diseases with key metadata.
        """
        conditions = []
        for _, row in self.df.iterrows():
            conditions.append({
                "disease": row.get("Disease", "Condition"),
                "hindi_name": row.get("Hindi Name", ""),
                "marathi_name": row.get("Marathi Name", ""),
                "symptoms": row.get("Symptoms", ""),
                "severity": row.get("Symptom Severity", ""),
                "doshas": row.get("Doshas", ""),
                "herbs": row.get("Ayurvedic Herbs", ""),
                "formulation": row.get("Formulation", ""),
                "season": row.get("Seasonal Variation", "")
            })
        return conditions

    def get_condition_detail(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive details for a single disease or condition.
        """
        target = disease_name.lower().strip()
        matched = self.df[self.df["Disease"].str.lower() == target]
        if matched.empty:
            # Try partial substring match
            matched = self.df[self.df["Disease"].str.lower().str.contains(target, regex=False)]
        
        if matched.empty:
            return None

        row = matched.iloc[0]
        return {
            "disease": row.get("Disease", "Condition"),
            "hindi_name": row.get("Hindi Name", "N/A"),
            "marathi_name": row.get("Marathi Name", "N/A"),
            "symptoms": row.get("Symptoms", "Not specified"),
            "diagnosis_tests": row.get("Diagnosis & Tests", "Clinical consultation"),
            "symptom_severity": row.get("Symptom Severity", "Moderate"),
            "duration": row.get("Duration of Treatment", "As directed"),
            "medical_history": row.get("Medical History", "None noted"),
            "current_medications": row.get("Current Medications", "None"),
            "risk_factors": row.get("Risk Factors", "Stress, Diet"),
            "environmental_factors": row.get("Environmental Factors", "Weather shifts"),
            "sleep_patterns": row.get("Sleep Patterns", "Normal to Irregular"),
            "stress_levels": row.get("Stress Levels", "Moderate"),
            "physical_activity": row.get("Physical Activity Levels", "Moderate"),
            "dietary_habits": row.get("Dietary Habits", "Regular"),
            "allergies": row.get("Allergies (Food/Env)", "None reported"),
            "seasonal_variation": row.get("Seasonal Variation", "All seasons"),
            "age_group": row.get("Age Group", "All ages"),
            "gender": row.get("Gender", "All genders"),
            "occupation_lifestyle": row.get("Occupation and Lifestyle", "General"),
            "cultural_preferences": row.get("Cultural Preferences", "Traditional Indian"),
            "herbal_remedies": row.get("Herbal/Alternative Remedies", "Herbal infusions"),
            "ayurvedic_herbs": row.get("Ayurvedic Herbs", "Traditional herbs"),
            "formulation": row.get("Formulation", "Churna / Vati / Kwath"),
            "doshas": row.get("Doshas", "Tridosha"),
            "prakriti": row.get("Constitution/Prakriti", "General"),
            "diet_lifestyle": row.get("Diet and Lifestyle Recommendations", "Fresh warm foods"),
            "yoga_therapy": row.get("Yoga & Physical Therapy", "Pranayama and Asanas"),
            "medical_intervention": row.get("Medical Intervention", "Consult qualified doctor"),
            "prevention": row.get("Prevention", "Dinacharya and balanced lifestyle"),
            "prognosis": row.get("Prognosis", "Good"),
            "complications": row.get("Complications", "None if managed promptly"),
            "patient_recommendations": row.get("Patient Recommendations", "Adequate rest and hydration")
        }

    def get_summary_metrics(self) -> Dict[str, int]:
        """
        Get dataset summary counts for analytics dashboard.
        """
        unique_diseases = self.df["Disease"].nunique()
        
        # Count unique herbs
        herbs_set = set()
        for h_str in self.df["Ayurvedic Herbs"].dropna():
            items = [h.strip().title() for h in re.split(r"[,;/]\s*", str(h_str)) if h.strip() and "consult" not in h.lower()]
            herbs_set.update(items)

        # Count unique symptoms tokens
        symptoms_set = set()
        for s_str in self.df["Symptoms"].dropna():
            items = [s.strip().lower() for s in re.split(r"[,;]\s*", str(s_str)) if s.strip()]
            symptoms_set.update(items)

        # Count unique formulations
        formulations_set = set()
        for f_str in self.df["Formulation"].dropna():
            items = [f.strip().title() for f in re.split(r"[,;/]\s*|\sand\s", str(f_str)) if f.strip()]
            formulations_set.update(items)

        return {
            "total_records": len(self.df),
            "unique_diseases": unique_diseases,
            "unique_herbs": len(herbs_set),
            "unique_symptoms": len(symptoms_set),
            "unique_formulations": len(formulations_set)
        }

    def get_popular_symptoms_list(self) -> List[str]:
        """
        Get curated list of common symptoms for structured click chips in UI.
        """
        common_symptoms = [
            "Cough", "Sore throat", "Fever", "Chest congestion", "Headache",
            "Indigestion", "Joint pain", "Fatigue", "Acidity", "Constipation",
            "Sneezing", "Runny nose", "Nausea", "Insomnia", "Loss of appetite",
            "Dry skin", "Body ache", "Abdominal pain", "Bloating", "Stress"
        ]
        return common_symptoms


def main():
    """Standalone test script for the recommendation engine."""
    print("=" * 70)
    print("AYURHERB - RECOMMENDATION ENGINE TEST SUITE")
    print("=" * 70)
    
    recommender = AyurvedicRecommender()
    
    test_queries = [
        "I have cough, sore throat and mild chest congestion.",
        "I am suffering from severe headache and migraine with nausea.",
        "Feeling indigestion, stomach acidity and bloating after meals.",
        "Joint pain and morning stiffness in knees.",
        "High fever with shivering and body pain."
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        results = recommender.recommend(q, top_k=3)
        if not results:
            print("  No matching records found above threshold.")
        for idx, res in enumerate(results, 1):
            print(f"  [{idx}] Match: {res['disease']} (Hindi: {res['hindi_name']})")
            print(f"      Symptom Similarity Score: {res['similarity_percentage']}% ({res['similarity_score']})")
            print(f"      Ayurvedic Herbs: {res['ayurvedic_herbs']}")
            print(f"      Formulation: {res['formulation']}")
            print(f"      Doshas: {res['doshas']}")

    print("\nMetrics Summary:")
    print(recommender.get_summary_metrics())
    print("=" * 70)


if __name__ == "__main__":
    main()
