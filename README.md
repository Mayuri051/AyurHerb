# AyurHerb – Data-Driven Ayurvedic Wellness Recommendation System

**AyurHerb** is an educational, data-driven web application and college mini-project built for the **Data Science using Python Lab**. The application analyzes the authentic **AyurGenixAI_Dataset.csv** dataset to recommend traditional Ayurvedic herbs, classical formulations, dosha considerations, dietary suggestions (Pathya/Apathya), and lifestyle routines based on user-entered symptoms.

---

## 1. Problem Statement
Ayurveda is a comprehensive traditional healing philosophy based on balancing the Tridoshas (Vata, Pitta, and Kapha). However, navigating hundreds of classical medicinal herbs (*dravyas*), multi-ingredient formulations (*kalpana*), and lifestyle guidelines based on complex symptom combinations is difficult without domain reference. Many existing digital solutions either provide generic advice or rely on ungrounded external generative models prone to medical hallucination. 

**AyurHerb** bridges this gap by grounding recommendations directly in a curated dataset of 446 clinical Ayurvedic records using transparent **Natural Language Processing (NLP)** and **TF-IDF Cosine Similarity Vector Space Modeling**.

---

## 2. Project Objectives
- Programmatically preprocess and clean the **AyurGenixAI** dataset without altering original historical records.
- Perform comprehensive **Exploratory Data Analysis (EDA)** and generate statistical visual charts.
- Implement an in-memory **TF-IDF + Cosine Similarity** NLP recommendation engine without third-party AI APIs.
- Enforce a strict **Clinical Red-Flag Safety Triage** filter that intercepts acute or emergency medical symptoms.
- Deliver an intuitive, responsive, and healthcare-themed web application using **Flask** and **Bootstrap 5**.
- Provide audit tracking via an embedded **SQLite** search history database.

---

## 3. Key Features
1. **Symptom Checker (`/symptom-checker`)**: Allows free-text symptom descriptions or quick selection using interactive symptom tags/chips.
2. **Transparent Similarity Metric**: Results are explicitly scored as **"Symptom Similarity Score"** (e.g. 67.5%), rather than medical confidence or cure probability.
3. **Multilingual Context**: Displays condition names in **English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**.
4. **Ayurvedic Herb Explorer (`/herbs`, `/herb/<name>`)**: Search and filter 40+ documented herbs by name, formulation type, and Dosha affiliation.
5. **Condition & Disease Directory (`/conditions`, `/condition/<name>`)**: Comprehensive directory covering 360+ conditions with diagnostic tests, severity, lifestyle, and yoga therapy.
6. **Visual Analytics Dashboard (`/analytics`)**: Interactive dashboard displaying 11 statistical EDA visualizations.
7. **Emergency Red-Flag Safety System (`safety.py`)**: Intercepts emergency inputs (e.g. chest pain, severe breathing failure) and prioritizes emergency medical care advice.
8. **Local SQLite History (`/history`)**: Tracks past searches locally with a 1-click history clearance utility.

---

## 4. Dataset Description
The system is built on **`data/AyurGenixAI_Dataset.csv`** containing **446 records** and **34 clinical parameters**:
- **Disease & Nomenclature**: `Disease`, `Hindi Name`, `Marathi Name`
- **Clinical Presentation**: `Symptoms`, `Diagnosis & Tests`, `Symptom Severity`, `Duration of Treatment`
- **Patient Background**: `Medical History`, `Current Medications`, `Risk Factors`, `Environmental Factors`, `Sleep Patterns`, `Stress Levels`, `Physical Activity Levels`, `Family History`, `Dietary Habits`, `Allergies (Food/Env)`
- **Demographics**: `Age Group`, `Gender`, `Occupation and Lifestyle`, `Cultural Preferences`, `Seasonal Variation`
- **Ayurvedic Management**: `Herbal/Alternative Remedies`, `Ayurvedic Herbs`, `Formulation`, `Doshas`, `Constitution/Prakriti`, `Diet and Lifestyle Recommendations`, `Yoga & Physical Therapy`
- **Clinical Prognosis**: `Medical Intervention`, `Prevention`, `Prognosis`, `Complications`, `Patient Recommendations`

---

## 5. Data Science & Machine Learning Methodology

### A. Preprocessing Pipeline (`preprocess.py`)
- Standardizes inconsistent capitalization, whitespace, and delimiter formatting.
- Handles sparse/missing values using domain-appropriate defaults (e.g., "None reported", "General wellness routine").
- Normalizes multi-term symptom descriptions.
- Generates a composite search feature column (`Unified_Search_Text`).
- Saves the clean output to `data/ayurgenixai_cleaned.csv`.

### B. Exploratory Data Analysis (`eda.py`)
Generates 11 publication-grade charts saved in `static/plots/`:
1. `disease_frequency.png` – Top documented conditions.
2. `severity_distribution.png` – Mild, Moderate, High, Severe distributions.
3. `age_distribution.png` – Target age group demographics.
4. `gender_distribution.png` – Gender representation.
5. `seasonal_variation.png` – Aggravation across seasonal cycles (Ritucharya).
6. `dosha_distribution.png` – Vata, Pitta, and Kapha imbalance distributions.
7. `top_herbs.png` – Top 15 recommended Ayurvedic herbs (Tulsi, Ashwagandha, Brahmi, etc.).
8. `top_remedies.png` – Most frequent home remedies.
9. `disease_vs_severity.png` – Cross-tabulation of condition vs. severity.
10. `disease_vs_season.png` – Heatmap linking diseases to seasonal triggers.
11. `formulation_types.png` – Distribution of delivery forms (Churna, Vati, Kwath, Taila).

### C. Recommendation Engine (`recommendation_engine.py`)
1. **NLP Text Vectorization**: Converts symptom descriptions into high-dimensional TF-IDF vectors with unigram and bigram ranges `(1, 2)` and sublinear TF scaling.
2. **Cosine Similarity**: Calculates vector angle similarity between query vector $\vec{u}$ and dataset records $\vec{v}$:
   $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
3. **Threshold Filtering**: Configurable relevance threshold (`SIMILARITY_THRESHOLD = 0.12 - 0.15`) returns only meaningful, ranked matches.

---

## 6. Technology Stack
- **Programming Language**: Python 3.13+
- **Backend Framework**: Flask (Jinja2 Templates)
- **Data Analysis**: Pandas, NumPy
- **Machine Learning / NLP**: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Data Visualization**: Matplotlib, Seaborn
- **Database**: SQLite 3 (`ayurherb.db`)
- **Frontend UI**: Bootstrap 5, CSS3, JavaScript

---

## 7. Project Architecture & Folder Structure

```
ayurherb/
│
├── app.py                      # Flask main entrypoint & web routes
├── preprocess.py               # Data cleaning, normalization & feature engineering
├── eda.py                      # Exploratory Data Analysis & plot generator
├── recommendation_engine.py    # TF-IDF & Cosine Similarity recommendation engine
├── safety.py                   # Medical red-flag detection & emergency triage
├── database.py                 # SQLite database helper for search history
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
│
├── data/
│   ├── AyurGenixAI_Dataset.csv # Original raw dataset
│   └── ayurgenixai_cleaned.csv # Cleaned & normalized dataset
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── symptom_checker.html
│   ├── results.html
│   ├── herbs.html
│   ├── herb_detail.html
│   ├── conditions.html
│   ├── condition_detail.html
│   ├── analytics.html
│   ├── history.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom healthcare styling & design tokens
│   ├── js/
│   │   └── script.js           # Interactive symptom tag toggling & sync
│   └── plots/                  # 11 Generated high-resolution EDA figures
│       ├── disease_frequency.png
│       ├── severity_distribution.png
│       ├── age_distribution.png
│       ├── gender_distribution.png
│       ├── seasonal_variation.png
│       ├── dosha_distribution.png
│       ├── top_herbs.png
│       ├── top_remedies.png
│       ├── disease_vs_severity.png
│       ├── disease_vs_season.png
│       └── formulation_types.png
│
└── instance/
    └── ayurherb.db             # Local SQLite database
```

---

## 8. Installation & Setup Guide

### Step 1: Clone or Navigate to Project Directory
```bash
cd C:\Users\DELL\.gemini\antigravity\scratch\ayurherb
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Data Preprocessing
```bash
python preprocess.py
```

### Step 5: Generate EDA Visualizations
```bash
python eda.py
```

### Step 6: Start the Flask Application
```bash
python app.py
```

### Step 7: Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Verification & Test Case Examples

| Test Case | Input Symptom Description | Expected System Output |
|---|---|---|
| **1. Respiratory** | *"I have cough and sore throat"* | Recommends Cough / Strep throat, Tulsi, Ashwagandha, Ginger & Honey formulation. |
| **2. Neurological** | *"I have headache and migraine"* | Recommends Migraine condition, Brahmi, Jatamansi, Peppermint oil drops. |
| **3. Fever** | *"I have fever with shivering"* | Identifies Pitta aggravation fever conditions, Neem, Papaya leaves. |
| **4. Digestive** | *"I have indigestion, acidity and bloating"* | Recommends Indigestion / Peptic ulcers, Ajwain, Triphala, warm water. |
| **5. Empty Input** | `""` (Empty string) | Validates input gracefully and prompts user to provide symptoms. |
| **6. Unrelated Query** | *"quantum physics rocket propulsion"* | Displays *"No sufficiently similar record was found in the current dataset"*. |
| **7. Emergency Triage** | *"I have severe chest pain and severe difficulty breathing"* | **Safety System Triggers**: Withholds self-care remedies and displays immediate emergency medical alert with emergency hotlines. |

---

## 10. Educational & Safety Disclaimer
> **IMPORTANT NOTICE**:
> AyurHerb is developed purely as an educational data science project and wellness reference. It does **NOT** provide clinical medical diagnoses, individualized medical prescriptions, or guaranteed cures. Ayurvedic formulations and lifestyle suggestions must be evaluated under the supervision of a licensed healthcare professional or qualified Ayurvedic doctor.

---

## 11. Future Enhancements
- Integration of Prakriti self-assessment questionnaire (Dosha quiz).
- Voice-enabled symptom input using Web Speech API.
- Support for regional language translations beyond Hindi and Marathi (e.g., Tamil, Telugu, Kannada).
- Exportable PDF wellness summary reports for consultations.
