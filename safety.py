"""
AyurHerb - Medical Safety & Triage Module (safety.py)
Author: Data Science Team
Description: Implements critical clinical red-flag symptom detection and emergency warnings.
             Ensures users with severe, acute, or potentially life-threatening conditions are
             immediately redirected to professional emergency medical care.
"""

import re
from typing import Dict, Any, List


# Defined emergency red-flag patterns with regex for robust semantic matching
RED_FLAG_PATTERNS = {
    "Severe Breathing Difficulty": [
        r"\bsevere difficulty breathing\b",
        r"\bcan'?t breathe\b",
        r"\bsevere shortness of breath\b",
        r"\bgasping for air\b",
        r"\bbreathing failure\b",
        r"\bchoking\b",
        r"\bthroat closing\b",
        r"\bstridor\b"
    ],
    "Acute Chest Pain / Cardiac Concern": [
        r"\bsevere chest pain\b",
        r"\bcrushing chest pain\b",
        r"\bchest pressure radiating\b",
        r"\bheart attack\b",
        r"\bcardiac arrest\b",
        r"\bpain radiating to left arm\b",
        r"\bchest tightness and sweating\b"
    ],
    "Loss of Consciousness / Neurological Emergency": [
        r"\bunconscious\b",
        r"\bunresponsiv\w*\b",
        r"\bfainted\b",
        r"\bpassing out\b",
        r"\bseizure\w*\b",
        r"\bconvulsion\w*\b",
        r"\bparalysis\b",
        r"\bslurred speech\b",
        r"\bsudden facial drooping\b",
        r"\bsevere confusion\b"
    ],
    "Severe Bleeding / Hemorrhage": [
        r"\bsevere bleeding\b",
        r"\buncontrolled bleeding\b",
        r"\bvomiting blood\b",
        r"\bhematemesis\b",
        r"\bcoughing blood\b",
        r"\bhemoptysis\b",
        r"\bblack tarry stools\b"
    ],
    "Severe Dehydration / Critical Vital Instability": [
        r"\bsevere dehydration\b",
        r"\bunable to keep fluids\b",
        r"\bsunken eyes and lethargy\b",
        r"\bcyanosis\b",
        r"\bturning blue\b"
    ],
    "Dangerous High / Persistent Fever": [
        r"\bvery high fever\b",
        r"\bfever above 104\b",
        r"\bfever above 40\b",
        r"\bpersistent high fever\b",
        r"\bstiff neck and high fever\b"
    ]
}


def evaluate_safety(query: str) -> Dict[str, Any]:
    """
    Evaluate user input against clinical red-flag indicators.
    
    Returns:
        Dict containing:
            - is_emergency (bool): True if acute red flags are detected.
            - emergency_message (str): Priority medical alert text.
            - detected_flags (List[str]): List of categories triggered.
            - advice (str): Actionable guidance for emergency healthcare contact.
    """
    if not query or not query.strip():
        return {
            "is_emergency": False,
            "emergency_message": "",
            "detected_flags": [],
            "advice": ""
        }

    cleaned_text = query.lower()
    detected_flags = []

    for category, patterns in RED_FLAG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, cleaned_text):
                detected_flags.append(category)
                break  # Category matched, move to next

    if detected_flags:
        return {
            "is_emergency": True,
            "emergency_message": (
                "Your symptoms may require prompt medical evaluation. "
                "Please consult a qualified healthcare professional or seek emergency care if the situation is urgent."
            ),
            "detected_flags": detected_flags,
            "advice": (
                "Ayurvedic and herbal remedies are intended for gentle, chronic wellness support "
                "and MUST NOT be used for acute emergencies. Call emergency services (112 / 911 / 108) "
                "or visit the nearest emergency medical department immediately."
            )
        }

    return {
        "is_emergency": False,
        "emergency_message": "",
        "detected_flags": [],
        "advice": ""
    }


def format_allergy_warning(allergy_text: str) -> str:
    """
    Format and highlight known allergen / contraindication notes from dataset.
    """
    if not allergy_text or allergy_text.lower() in ["none reported", "none", "not specified", ""]:
        return "No specific allergen contraindications documented in dataset."
    return f"Caution / Known Allergies: {allergy_text}"


if __name__ == "__main__":
    # Self-test safety module
    test_cases = [
        "I have a mild cough and runny nose.",
        "I have severe chest pain and severe difficulty breathing.",
        "Patient is unconscious and had a sudden seizure.",
        "I am vomiting blood with extreme abdominal pain.",
        "I have a headache after working late."
    ]
    
    print("=" * 60)
    print("AYURHERB SAFETY & RED-FLAG MODULE TEST")
    print("=" * 60)
    for t in test_cases:
        res = evaluate_safety(t)
        status = "[EMERGENCY ALERT]" if res["is_emergency"] else "[SAFE - WELLNESS ADVISORY]"
        print(f"\nInput: '{t}'")
        print(f"Status: {status}")
        if res["is_emergency"]:
            print(f"Triggered Flags: {', '.join(res['detected_flags'])}")
            print(f"Warning: {res['emergency_message']}")
    print("=" * 60)
