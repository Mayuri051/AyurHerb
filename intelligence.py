"""Explainable, rule-assisted wellness intelligence for AyurHerb.

This module deliberately refines dataset retrieval; it does not diagnose a
condition or estimate a treatment outcome.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


# Common English, Hindi, and Marathi spellings that can be safely mapped to
# the symptom vocabulary used by the educational dataset.
SYMPTOM_ALIASES = {
    "khansi": "cough", "खांसी": "cough", "खोकला": "cough",
    "bukhar": "fever", "बुखार": "fever", "ताप": "fever",
    "sir dard": "headache", "सिर दर्द": "headache", "डोकेदुखी": "headache",
    "pet dard": "abdominal pain", "पेट दर्द": "abdominal pain",
    "acidity": "acidity", "gas": "bloating", "gale mein dard": "sore throat",
    "thakan": "fatigue", "कमजोरी": "fatigue", "joint stiffness": "joint pain",
}

KNOWN_SYMPTOMS = (
    "cough", "sore throat", "fever", "chest congestion", "headache",
    "migraine", "indigestion", "joint pain", "fatigue", "acidity",
    "constipation", "sneezing", "runny nose", "nausea", "insomnia",
    "loss of appetite", "dry skin", "body ache", "abdominal pain",
    "bloating", "stress", "shivering", "dizziness",
)


def normalize_symptoms(query: str) -> Dict[str, Any]:
    """Normalize familiar multilingual phrases and extract recognized symptoms."""
    normalized = (query or "").lower().strip()
    applied_aliases: List[Dict[str, str]] = []
    for source, target in SYMPTOM_ALIASES.items():
        if source in normalized:
            normalized = normalized.replace(source, target)
            applied_aliases.append({"source": source, "normalized_to": target})

    recognized = [item for item in KNOWN_SYMPTOMS if item in normalized]
    return {
        "normalized_query": re.sub(r"\s+", " ", normalized).strip(),
        "recognized_symptoms": recognized,
        "applied_aliases": applied_aliases,
    }


def profile_from_form(form: Any) -> Dict[str, str]:
    """Read only optional, non-identifying context from a Flask form."""
    allowed = {
        "age_group": {"", "All ages", "0-18 years", "19-30 years", "30-60 years", "60+ years"},
        "season": {"", "Winter", "Summer", "Rainy", "All seasons"},
        "dosha": {"", "Vata", "Pitta", "Kapha", "Vata, Pitta", "Pitta, Kapha", "Vata, Kapha", "Tridosha"},
        "lifestyle": {"", "Low activity", "Moderate activity", "High stress", "Irregular sleep"},
    }
    profile = {}
    for key, choices in allowed.items():
        value = str(form.get(key, "")).strip()
        profile[key] = value if value in choices else ""
    return profile


def enrich_results(results: List[Dict[str, Any]], profile: Dict[str, str], recognized: Iterable[str]) -> List[Dict[str, Any]]:
    """Add transparent context evidence without changing the base retrieval score."""
    recognized_set = set(recognized)
    for item in results:
        evidence = [f"Matched symptom: {term}" for term in item.get("matched_terms", []) if term in recognized_set]
        context_matches = []
        if profile.get("dosha") and profile["dosha"].lower() in str(item.get("doshas", "")).lower():
            context_matches.append("selected dosha context")
        if profile.get("season") and profile["season"].lower() in str(item.get("seasonal_variation", "")).lower():
            context_matches.append("selected seasonal context")
        if profile.get("age_group") and profile["age_group"].lower() in str(item.get("age_group", "")).lower():
            context_matches.append("selected age-group context")
        item["explanation"] = {
            "symptom_evidence": evidence or ["Textual similarity to the dataset symptom record"],
            "context_matches": context_matches,
            "method": "TF-IDF cosine similarity; profile context is displayed as supporting evidence and does not diagnose.",
        }
    return results


def score_dosha(answers: Dict[str, str]) -> Dict[str, Any]:
    """Educational, non-clinical dosha questionnaire scoring."""
    scores = {"Vata": 0, "Pitta": 0, "Kapha": 0}
    answer_map = {
        "body_frame": {"light": "Vata", "medium": "Pitta", "solid": "Kapha"},
        "skin": {"dry": "Vata", "warm": "Pitta", "soft": "Kapha"},
        "appetite": {"irregular": "Vata", "strong": "Pitta", "steady": "Kapha"},
        "stress_response": {"anxious": "Vata", "irritable": "Pitta", "calm": "Kapha"},
    }
    for question, options in answer_map.items():
        dosha = options.get(answers.get(question, ""))
        if dosha:
            scores[dosha] += 1
    highest = max(scores, key=scores.get)
    ties = [name for name, score in scores.items() if score == scores[highest]]
    return {"scores": scores, "indicative_dosha": " / ".join(ties), "answered": sum(scores.values())}
