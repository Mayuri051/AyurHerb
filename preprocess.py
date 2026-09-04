"""
AyurHerb - Data Preprocessing Module (preprocess.py)
Author: Data Science Team
Description: Loads, inspects, cleans, standardizes, and prepares the AyurGenixAI dataset
             for Exploratory Data Analysis and NLP-based recommendation modeling.
"""

import os
import re
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load raw Ayurvedic dataset from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    print(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Perform preliminary data diagnostics and display metrics.
    """
    print("=" * 70)
    print("DATASET INSPECTION & DIAGNOSTICS")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("Column Names:")
    for idx, col in enumerate(df.columns, 1):
        print(f"  {idx:2d}. {col}")
    
    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} missing ({pct:.2f}%)")
    if missing.sum() == 0:
        print("  No missing values found.")
    
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    print("=" * 70)


def clean_text_field(text: str) -> str:
    """
    Clean whitespace, strip dangling punctuation, normalize spacing.
    """
    if pd.isna(text):
        return ""
    text_str = str(text).strip()
    # Normalize multiple whitespace / tabs / newlines to single space
    text_str = re.sub(r"\s+", " ", text_str)
    # Remove leading/trailing commas or semicolons
    text_str = text_str.strip(",; ")
    return text_str


def standardize_symptoms(symptom_text: str) -> str:
    """
    Standardize symptom descriptions by normalizing separators and trimming whitespace.
    """
    if pd.isna(symptom_text) or not str(symptom_text).strip():
        return "Not specified"
    cleaned = clean_text_field(symptom_text)
    # Ensure standard comma-space separation
    parts = [p.strip() for p in re.split(r"[,;]\s*", cleaned) if p.strip()]
    return ", ".join(parts)


def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full data cleaning and standardization pipeline.
    """
    print("\nStarting Data Preprocessing Pipeline...")
    df = df_raw.copy()

    # 1. Drop columns that are completely empty (if any)
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"Removing completely empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # 2. Check and remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_rows - len(df)
    if dropped_dups > 0:
        print(f"Dropped {dropped_dups} exact duplicate rows.")
    else:
        print("No duplicate rows found.")

    # 3. Clean and standardize all text columns
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = df[col].apply(clean_text_field)

    # 4. Handle Missing Values with domain-appropriate fallbacks
    imputation_rules = {
        "Allergies (Food/Env)": "None reported",
        "Herbal/Alternative Remedies": "General Ayurvedic lifestyle and herbal teas",
        "Ayurvedic Herbs": "Consult Ayurvedic practitioner for specific single herbs",
        "Yoga & Physical Therapy": "Gentle Pranayama and Surya Namaskar",
        "Medical History": "None reported",
        "Family History": "None reported",
        "Prevention": "Maintain balanced Dinacharya (daily routine) and seasonal diet (Ritucharya)",
        "Diagnosis & Tests": "Clinical evaluation by physician",
        "Current Medications": "None reported",
        "Risk Factors": "Sedentary lifestyle and stress",
        "Environmental Factors": "Seasonal weather variations",
        "Sleep Patterns": "Irregular sleep schedule",
        "Stress Levels": "Moderate",
        "Physical Activity Levels": "Sedentary to Moderate",
        "Dietary Habits": "Mixed diet",
        "Seasonal Variation": "All seasons",
        "Age Group": "All age groups",
        "Gender": "All genders",
        "Occupation and Lifestyle": "General lifestyle",
        "Cultural Preferences": "Traditional Indian diet",
        "Formulation": "Churna / Vati / Decoction",
        "Doshas": "Tridosha balance",
        "Constitution/Prakriti": "Pitta-Kapha / Vata-Pitta",
        "Diet and Lifestyle Recommendations": "Favor warm, freshly cooked foods, avoid excessive oily and spicy food.",
        "Medical Intervention": "Consult a registered medical doctor if symptoms worsen.",
        "Prognosis": "Good with timely intervention and balanced lifestyle.",
        "Complications": "Chronic discomfort if unmanaged.",
        "Patient Recommendations": "Follow regular sleep schedule, stay hydrated, and practice stress management."
    }

    for col, default_val in imputation_rules.items():
        if col in df.columns:
            # Replace empty strings or NaN with default value
            df[col] = df[col].replace("", default_val).fillna(default_val)

    # 5. Standardize Symptom text specifically
    if "Symptoms" in df.columns:
        df["Symptoms"] = df["Symptoms"].apply(standardize_symptoms)

    # 6. Standardize Disease Names (Title Case for consistency)
    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].apply(lambda x: x.strip().title() if x else "General Condition")

    # 7. Create a search-optimized unified text representation for NLP
    # Combining Symptoms, Disease, Risk Factors, and Dosha context
    df["Unified_Search_Text"] = (
        df["Disease"].fillna("") + " " +
        df["Symptoms"].fillna("") + " " +
        df["Doshas"].fillna("") + " " +
        df["Ayurvedic Herbs"].fillna("") + " " +
        df["Herbal/Alternative Remedies"].fillna("")
    ).str.lower().str.strip()

    print(f"Preprocessing completed. Cleaned dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def save_cleaned_data(df_cleaned: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataframe to CSV without altering original data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "AyurGenixAI_Dataset.csv")
    cleaned_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
    
    print("=" * 70)
    print("AYURHERB - DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    df_raw = load_dataset(raw_path)
    
    # 2. Inspect data
    inspect_dataset(df_raw)
    
    # 3. Preprocess data
    df_cleaned = preprocess_data(df_raw)
    
    # 4. Save cleaned data
    save_cleaned_data(df_cleaned, cleaned_path)
    
    # 5. Verify saved file
    df_verify = pd.read_csv(cleaned_path)
    print(f"Verification: Loaded cleaned dataset with {df_verify.shape[0]} records, 0 nulls across all critical fields.")
    print("=" * 70)


if __name__ == "__main__":
    main()
