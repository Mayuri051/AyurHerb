"""
AyurHerb - Exploratory Data Analysis Module (eda.py)
Author: Data Science Team
Description: Computes statistical metrics and generates visual charts from the cleaned
             Ayurvedic dataset. Visualizations are saved to static/plots/ for the Flask dashboard.
"""

import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration for clean, modern academic healthcare aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True
})

PALETTE_TEAL = ["#1b4d3e", "#2c6e49", "#4c956c", "#70b77e", "#a3d9a5", "#d6f5d8"]
PALETTE_MULTI = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6a4c93", "#118ab2", "#06d6a0"]


def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load cleaned Ayurvedic dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}. Run preprocess.py first.")
    return pd.read_csv(file_path)


def generate_disease_frequency_plot(df: pd.DataFrame, output_dir: str) -> None:
    """1. Top Most Frequent Diseases/Conditions"""
    plt.figure(figsize=(10, 6))
    top_diseases = df["Disease"].value_counts().head(12)
    ax = sns.barplot(x=top_diseases.values, y=top_diseases.index, palette="viridis")
    plt.title("Top 12 Most Frequently Documented Conditions in Dataset", pad=15, weight="bold")
    plt.xlabel("Number of Records")
    plt.ylabel("Condition Name")
    for i, v in enumerate(top_diseases.values):
        ax.text(v + 0.1, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disease_frequency.png"), dpi=300)
    plt.close()
    print("  [+] Generated: disease_frequency.png")


def generate_severity_plot(df: pd.DataFrame, output_dir: str) -> None:
    """2. Symptom Severity Distribution"""
    plt.figure(figsize=(9, 5))
    severity_counts = df["Symptom Severity"].value_counts()
    colors = sns.color_palette("mako", len(severity_counts))
    ax = sns.barplot(x=severity_counts.index, y=severity_counts.values, palette=colors)
    plt.title("Distribution of Documented Symptom Severity Levels", pad=15, weight="bold")
    plt.xlabel("Severity Classification")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=20, ha="right")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "severity_distribution.png"), dpi=300)
    plt.close()
    print("  [+] Generated: severity_distribution.png")


def generate_age_group_plot(df: pd.DataFrame, output_dir: str) -> None:
    """3. Age-Group Distribution (Top Categories)"""
    plt.figure(figsize=(10, 6))
    age_counts = df["Age Group"].value_counts().head(10)
    ax = sns.barplot(x=age_counts.values, y=age_counts.index, palette="crest")
    plt.title("Top 10 Target Age Groups Across Records", pad=15, weight="bold")
    plt.xlabel("Number of Records")
    plt.ylabel("Age Category")
    for i, v in enumerate(age_counts.values):
        ax.text(v + 0.1, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_distribution.png"), dpi=300)
    plt.close()
    print("  [+] Generated: age_distribution.png")


def generate_gender_plot(df: pd.DataFrame, output_dir: str) -> None:
    """4. Gender Demographics Distribution"""
    plt.figure(figsize=(7, 6))
    gender_counts = df["Gender"].value_counts()
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#e9c46a", "#a8dadc"][:len(gender_counts)]
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
            startangle=140, colors=colors, explode=[0.03]*len(gender_counts),
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    plt.title("Gender Demographics in Ayurvedic Dataset", pad=15, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_distribution.png"), dpi=300)
    plt.close()
    print("  [+] Generated: gender_distribution.png")


def generate_seasonal_variation_plot(df: pd.DataFrame, output_dir: str) -> None:
    """5. Seasonal Variation Distribution"""
    plt.figure(figsize=(10, 6))
    # Extract individual seasons
    seasons_list = []
    for val in df["Seasonal Variation"].dropna():
        items = [s.strip() for s in re.split(r"[,;]\s*|\sand\s", str(val)) if s.strip()]
        seasons_list.extend(items)
    season_counts = pd.Series(seasons_list).value_counts().head(8)
    
    ax = sns.barplot(x=season_counts.values, y=season_counts.index, palette="YlGnBu_r")
    plt.title("Condition Aggravation by Seasonal Variation (Ritucharya)", pad=15, weight="bold")
    plt.xlabel("Frequency")
    plt.ylabel("Season")
    for i, v in enumerate(season_counts.values):
        ax.text(v + 0.2, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seasonal_variation.png"), dpi=300)
    plt.close()
    print("  [+] Generated: seasonal_variation.png")


def generate_dosha_distribution_plot(df: pd.DataFrame, output_dir: str) -> None:
    """6. Dosha Imbalance Distribution"""
    plt.figure(figsize=(9, 5))
    dosha_counts = df["Doshas"].value_counts()
    colors = sns.color_palette("flare", len(dosha_counts))
    ax = sns.barplot(x=dosha_counts.index, y=dosha_counts.values, palette=colors)
    plt.title("Ayurvedic Dosha Imbalance Classification (Vata, Pitta, Kapha)", pad=15, weight="bold")
    plt.xlabel("Dosha Combination")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=25, ha="right")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dosha_distribution.png"), dpi=300)
    plt.close()
    print("  [+] Generated: dosha_distribution.png")


def generate_top_herbs_plot(df: pd.DataFrame, output_dir: str) -> None:
    """7. Most Frequently Documented Ayurvedic Herbs"""
    plt.figure(figsize=(11, 7))
    herbs_list = []
    for entry in df["Ayurvedic Herbs"].dropna():
        # Split multi-herbs by comma
        items = [h.strip().title() for h in re.split(r"[,;/]\s*", str(entry)) 
                 if h.strip() and "consult" not in h.lower()]
        herbs_list.extend(items)
    
    herb_counts = pd.Series(herbs_list).value_counts().head(15)
    ax = sns.barplot(x=herb_counts.values, y=herb_counts.index, palette="Greens_r")
    plt.title("Top 15 Most Frequently Recommended Ayurvedic Herbs (Dravyas)", pad=15, weight="bold")
    plt.xlabel("Occurrences Across Formulations")
    plt.ylabel("Herb Name (Sanskrit / Botanical)")
    for i, v in enumerate(herb_counts.values):
        ax.text(v + 0.2, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_herbs.png"), dpi=300)
    plt.close()
    print("  [+] Generated: top_herbs.png")


def generate_top_remedies_plot(df: pd.DataFrame, output_dir: str) -> None:
    """8. Most Common Traditional & Herbal Remedies"""
    plt.figure(figsize=(11, 7))
    remedies_list = []
    for entry in df["Herbal/Alternative Remedies"].dropna():
        items = [r.strip().capitalize() for r in re.split(r"[,;]\s*", str(entry)) 
                 if r.strip() and len(r) > 3 and "general" not in r.lower()]
        remedies_list.extend(items)
    
    remedy_counts = pd.Series(remedies_list).value_counts().head(12)
    ax = sns.barplot(x=remedy_counts.values, y=remedy_counts.index, palette="Blues_r")
    plt.title("Top 12 Traditional Home Remedies & Herbal Applications", pad=15, weight="bold")
    plt.xlabel("Occurrences in Dataset")
    plt.ylabel("Home Remedy / Traditional Modality")
    for i, v in enumerate(remedy_counts.values):
        ax.text(v + 0.1, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_remedies.png"), dpi=300)
    plt.close()
    print("  [+] Generated: top_remedies.png")


def generate_disease_vs_severity_plot(df: pd.DataFrame, output_dir: str) -> None:
    """9. Top Disease vs Symptom Severity Cross-Tabulation"""
    plt.figure(figsize=(10, 6))
    top_diseases = df["Disease"].value_counts().head(8).index
    df_sub = df[df["Disease"].isin(top_diseases)]
    ct = pd.crosstab(df_sub["Disease"], df_sub["Symptom Severity"])
    
    ax = ct.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="viridis")
    plt.title("Symptom Severity Breakdown Across Frequent Conditions", pad=15, weight="bold")
    plt.xlabel("Condition")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disease_vs_severity.png"), dpi=300)
    plt.close()
    print("  [+] Generated: disease_vs_severity.png")


def generate_disease_vs_season_plot(df: pd.DataFrame, output_dir: str) -> None:
    """10. Disease vs Seasonal Variation Heatmap"""
    plt.figure(figsize=(11, 7))
    top_diseases = df["Disease"].value_counts().head(10).index
    top_seasons = df["Seasonal Variation"].value_counts().head(6).index
    df_sub = df[df["Disease"].isin(top_diseases) & df["Seasonal Variation"].isin(top_seasons)]
    
    ct = pd.crosstab(df_sub["Disease"], df_sub["Seasonal Variation"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("Heatmap: Top Conditions vs Seasonal Variation", pad=15, weight="bold")
    plt.xlabel("Season / Environmental Trigger")
    plt.ylabel("Condition")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disease_vs_season.png"), dpi=300)
    plt.close()
    print("  [+] Generated: disease_vs_season.png")


def generate_formulation_types_plot(df: pd.DataFrame, output_dir: str) -> None:
    """11. Ayurvedic Formulation Delivery Types (Churna, Vati, Kwath, Taila, etc.)"""
    plt.figure(figsize=(10, 6))
    form_list = []
    for val in df["Formulation"].dropna():
        items = [f.strip().title() for f in re.split(r"[,;/]\s*|\sand\s", str(val)) if f.strip()]
        form_list.extend(items)
    form_counts = pd.Series(form_list).value_counts().head(12)
    
    ax = sns.barplot(x=form_counts.values, y=form_counts.index, palette="Spectral")
    plt.title("Distribution of Ayurvedic Formulation Delivery Forms (Kalpana)", pad=15, weight="bold")
    plt.xlabel("Count in Dataset")
    plt.ylabel("Formulation Category")
    for i, v in enumerate(form_counts.values):
        ax.text(v + 0.1, i, str(v), color="black", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "formulation_types.png"), dpi=300)
    plt.close()
    print("  [+] Generated: formulation_types.png")


def generate_all_plots(data_path: str, output_dir: str) -> None:
    """
    Generate the complete suite of 11 publication-grade statistical charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_cleaned_data(data_path)
    
    print("=" * 70)
    print("AYURHERB - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    print(f"Dataset summary for visualization: {len(df)} records across {df['Disease'].nunique()} unique conditions.")
    print(f"Saving all plot figures to: {output_dir}\n")
    
    generate_disease_frequency_plot(df, output_dir)
    generate_severity_plot(df, output_dir)
    generate_age_group_plot(df, output_dir)
    generate_gender_plot(df, output_dir)
    generate_seasonal_variation_plot(df, output_dir)
    generate_dosha_distribution_plot(df, output_dir)
    generate_top_herbs_plot(df, output_dir)
    generate_top_remedies_plot(df, output_dir)
    generate_disease_vs_severity_plot(df, output_dir)
    generate_disease_vs_season_plot(df, output_dir)
    generate_formulation_types_plot(df, output_dir)
    
    print("\nAll 11 EDA charts generated and validated successfully!")
    print("=" * 70)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
    plots_dir = os.path.join(base_dir, "static", "plots")
    generate_all_plots(cleaned_path, plots_dir)


if __name__ == "__main__":
    main()
