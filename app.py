"""
AyurHerb - Flask Web Application (app.py)
Author: Data Science Team
Description: Multi-paradigm cognitive healthcare web application integrating:
             1. NLP & TF-IDF Semantic Retrieval
             2. Supervised Machine Learning (Random Forest & AdaBoost) [Unit V]
             3. Fuzzy Logic Dosage Controller (Mamdani FIS) [Unit III]
             4. Bayesian Network Probabilistic Inference [Unit I]
             5. Clinical Safety & Emergency Triage
             6. SQLite Search History Audit
"""

import os
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from recommendation_engine import AyurvedicRecommender, SIMILARITY_THRESHOLD
from safety import evaluate_safety
from intelligence import enrich_results, normalize_symptoms, profile_from_form, score_dosha
from plant_identifier import PlantImageIdentifier, allowed_image, validate_image
from ml_classifier import AyurvedicMLClassifier
from fuzzy_engine import AyurvedicFuzzyInferenceSystem
from bayesian_engine import AyurvedicBayesianNetwork
import database as db

app = Flask(__name__)
app.secret_key = os.environ.get("AYURHERB_SECRET_KEY", "development-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Initialize SQLite database
db.init_db()

# Paths & Core Engines
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
upload_dir = os.path.join(base_dir, "instance", "uploads")
os.makedirs(upload_dir, exist_ok=True)

# 1. Recommendation Engine (TF-IDF + Cosine Similarity)
recommender = AyurvedicRecommender(data_path)

# 2. Supervised ML Classifier (Random Forest & AdaBoost - Unit V)
ml_classifier = AyurvedicMLClassifier(data_path)

# 3. Fuzzy Logic Inference Engine (Mamdani FIS - Unit III)
fuzzy_engine = AyurvedicFuzzyInferenceSystem()

# 4. Bayesian Probabilistic Network (Unit I)
bayesian_engine = AyurvedicBayesianNetwork(data_path)

# 5. Plant Identifier (Deep Learning / Vision hook - Unit IV)
plant_identifier = PlantImageIdentifier(os.path.join(base_dir, "models", "ayurvedic_plant_classifier"))


@app.route("/")
def index():
    """Home landing page with system overview, dataset stats, and ML highlights."""
    metrics = recommender.get_summary_metrics()
    popular_symptoms = recommender.get_popular_symptoms_list()[:8]
    ml_metrics = ml_classifier.metrics_
    return render_template(
        "index.html",
        metrics=metrics,
        popular_symptoms=popular_symptoms,
        ml_metrics=ml_metrics
    )


@app.route("/symptom-checker")
def symptom_checker():
    """Interactive symptom input and chip selector page."""
    popular_symptoms = recommender.get_popular_symptoms_list()
    return render_template(
        "symptom_checker.html",
        popular_symptoms=popular_symptoms
    )


@app.route("/wellness-profile", methods=["GET", "POST"])
def wellness_profile():
    """Educational questionnaire; non-diagnostic assessment."""
    result = None
    if request.method == "POST":
        answers = {key: request.form.get(key, "") for key in ("body_frame", "skin", "appetite", "stress_response")}
        result = score_dosha(answers)
    return render_template("wellness_profile.html", result=result)


@app.route("/analyze", methods=["POST"])
def analyze_symptoms():
    """
    Multi-stage cognitive & ML analysis pipeline:
    1. Safety & Emergency Triage
    2. Multilingual Normalization
    3. TF-IDF Semantic Ranking
    4. Random Forest & AdaBoost ML Classification (Unit V)
    5. Bayesian Probabilistic Inference (Unit I)
    6. Fuzzy Logic Dosage Calculation (Unit III)
    7. SQLite Logging
    """
    query = request.form.get("symptoms", "").strip()
    profile = profile_from_form(request.form)

    # Extract optional fuzzy tuning parameters
    try:
        severity_val = float(request.form.get("severity_score", 5.0))
    except (ValueError, TypeError):
        severity_val = 5.0

    try:
        duration_val = float(request.form.get("duration_days", 7.0))
    except (ValueError, TypeError):
        duration_val = 7.0

    if not query:
        flash("Please enter or select at least one symptom to analyze.", "warning")
        return redirect(url_for("symptom_checker"))

    # Stage 1: Safety & Emergency Triage
    safety_result = evaluate_safety(query)
    if safety_result["is_emergency"]:
        if request.form.get("save_history") == "yes":
            db.log_search(query=query, top_condition="Emergency Medical Warning Triggered",
                          similarity_score=0.0, results_count=0, safety_triggered=True)
        return render_template(
            "results.html",
            query=query,
            is_emergency=True,
            safety=safety_result,
            results=[],
            threshold=SIMILARITY_THRESHOLD,
            symptom_analysis={"recognized_symptoms": [], "applied_aliases": []},
            profile=profile,
            ml_result=None,
            bayesian_result=None,
            fuzzy_result=None
        )

    # Stage 2: Normalization & TF-IDF Retrieval
    symptom_analysis = normalize_symptoms(query)
    normalized_query = symptom_analysis["normalized_query"]
    results = recommender.recommend(query=normalized_query, top_k=6, threshold=SIMILARITY_THRESHOLD)
    results = enrich_results(results, profile, symptom_analysis["recognized_symptoms"])

    # Stage 3: Supervised ML Classification (Random Forest & AdaBoost)
    ml_result = ml_classifier.predict(normalized_query, top_k=3)

    # Stage 4: Bayesian Network Probabilistic Inference (Unit I)
    bayesian_result = bayesian_engine.infer(
        symptoms_query=normalized_query,
        dosha_evidence=profile.get("dosha"),
        season_evidence=profile.get("season"),
        top_k=3
    )

    # Stage 5: Fuzzy Logic Dosage & Formulation Potency (Unit III)
    fuzzy_result = fuzzy_engine.infer(
        severity_score=severity_val,
        duration_days=duration_val,
        agni_score=6.0
    )

    top_condition = results[0]["disease"] if results else (ml_result.get("rf_prediction") or "No Direct Match")
    top_score = results[0]["similarity_score"] if results else 0.0

    # Stage 6: SQLite Search History Logging
    if request.form.get("save_history") == "yes":
        db.log_search(query=query, top_condition=top_condition, similarity_score=top_score,
                      results_count=len(results), safety_triggered=False)

    return render_template(
        "results.html",
        query=query,
        is_emergency=False,
        safety=safety_result,
        results=results,
        threshold=SIMILARITY_THRESHOLD,
        symptom_analysis=symptom_analysis,
        profile=profile,
        ml_result=ml_result,
        bayesian_result=bayesian_result,
        fuzzy_result=fuzzy_result
    )


@app.route("/ml-evaluation")
def ml_evaluation():
    """Machine Learning & Ensemble Classification Benchmark Dashboard (Unit V)."""
    metrics = ml_classifier.metrics_
    features = ml_classifier.get_feature_importance(top_n=12)
    
    plots = [
        {
            "filename": "ml_model_comparison.png",
            "title": "Supervised Model Comparison (Unit V)",
            "description": "Accuracy, Precision, Recall, and F1-Scores across Random Forest, AdaBoost, Decision Tree, and Naive Bayes."
        },
        {
            "filename": "ml_feature_importance.png",
            "title": "Random Forest Symptom Feature Importance",
            "description": "Gini-importance scores of key symptom vocabulary nodes extracted by the tree ensemble."
        },
        {
            "filename": "ml_confusion_matrix.png",
            "title": "Confusion Matrix for Top Classes",
            "description": "True vs. Predicted class distribution showing classification accuracy across frequent conditions."
        }
    ]
    return render_template(
        "ml_evaluation.html",
        metrics=metrics,
        features=features,
        plots=plots
    )


@app.route("/fuzzy-dosage", methods=["GET", "POST"])
def fuzzy_dosage():
    """Interactive Fuzzy Logic Ayurvedic Dosage & Potency Calculator (Unit III)."""
    result = None
    inputs = {"severity": 5.0, "duration": 7.0, "agni": 5.5}
    if request.method == "POST":
        try:
            inputs["severity"] = float(request.form.get("severity", 5.0))
            inputs["duration"] = float(request.form.get("duration", 7.0))
            inputs["agni"] = float(request.form.get("agni", 5.5))
        except (ValueError, TypeError):
            pass
        result = fuzzy_engine.infer(
            severity_score=inputs["severity"],
            duration_days=inputs["duration"],
            agni_score=inputs["agni"]
        )
    return render_template("fuzzy_dosage.html", result=result, inputs=inputs)


@app.route("/plant-identifier", methods=["GET", "POST"])
def plant_identifier_page():
    """Deep Learning vision hook for plant leaf identification (Unit IV)."""
    prediction = None
    if request.method == "POST":
        image = request.files.get("plant_image")
        if not image or not image.filename:
            flash("Choose a JPG, PNG, or WEBP plant image first.", "warning")
        elif not allowed_image(image.filename):
            flash("Only JPG, JPEG, PNG, and WEBP files are accepted.", "warning")
        else:
            extension = image.filename.rsplit(".", 1)[1].lower()
            filename = f"{uuid4().hex}.{extension}"
            image_path = os.path.join(upload_dir, secure_filename(filename))
            image.save(image_path)
            try:
                if validate_image(image_path):
                    prediction = plant_identifier.identify(image_path)
                else:
                    flash("That file is not a valid decodable image.", "warning")
            finally:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
    return render_template("plant_identifier.html", prediction=prediction, model_ready=plant_identifier.is_ready)


@app.errorhandler(413)
def file_too_large(_error):
    flash("The image is too large. Maximum upload size is 5 MB.", "warning")
    return redirect(url_for("plant_identifier_page"))


@app.route("/herbs")
def herbs_explorer():
    """Herbs Explorer page with search and filtering."""
    search_query = request.args.get("q", "").strip().lower()
    selected_dosha = request.args.get("dosha", "").strip().lower()

    all_herbs = recommender.get_all_herbs()

    filtered_herbs = all_herbs
    if search_query:
        filtered_herbs = [
            h for h in filtered_herbs
            if search_query in h["name"].lower()
            or any(search_query in c.lower() for c in h["conditions"])
            or search_query in h["formulations"].lower()
        ]

    if selected_dosha:
        filtered_herbs = [
            h for h in filtered_herbs
            if selected_dosha in h["doshas"].lower()
        ]

    return render_template(
        "herbs.html",
        herbs=filtered_herbs,
        total_count=len(all_herbs),
        search_query=search_query,
        selected_dosha=selected_dosha
    )


@app.route("/herb/<path:herb_name>")
def herb_detail(herb_name):
    """Detailed view for a specific Ayurvedic herb."""
    detail = recommender.get_herb_detail(herb_name)
    if not detail:
        flash(f"Information for herb '{herb_name}' could not be found in the dataset.", "warning")
        return redirect(url_for("herbs_explorer"))

    return render_template("herb_detail.html", herb=detail)


@app.route("/conditions")
def conditions_explorer():
    """Conditions Explorer page with search and severity filter."""
    search_query = request.args.get("q", "").strip().lower()
    selected_severity = request.args.get("severity", "").strip().lower()
    selected_dosha = request.args.get("dosha", "").strip().lower()

    all_conditions = recommender.get_all_conditions()

    filtered_conditions = all_conditions
    if search_query:
        filtered_conditions = [
            c for c in filtered_conditions
            if search_query in c["disease"].lower()
            or search_query in c["hindi_name"].lower()
            or search_query in c["marathi_name"].lower()
            or search_query in c["symptoms"].lower()
            or search_query in c["herbs"].lower()
        ]

    if selected_severity:
        filtered_conditions = [
            c for c in filtered_conditions
            if selected_severity in c["severity"].lower()
        ]

    if selected_dosha:
        filtered_conditions = [
            c for c in filtered_conditions
            if selected_dosha in c["doshas"].lower()
        ]

    return render_template(
        "conditions.html",
        conditions=filtered_conditions,
        total_count=len(all_conditions),
        search_query=search_query,
        selected_severity=selected_severity,
        selected_dosha=selected_dosha
    )


@app.route("/condition/<path:condition_name>")
def condition_detail(condition_name):
    """Comprehensive condition detail page with multilingual names and Ayurvedic details."""
    detail = recommender.get_condition_detail(condition_name)
    if not detail:
        flash(f"Condition '{condition_name}' was not found in the dataset.", "warning")
        return redirect(url_for("conditions_explorer"))

    return render_template("condition_detail.html", condition=detail)


@app.route("/analytics")
def analytics():
    """Exploratory Data Analysis Dashboard showcasing dataset metrics and generated charts."""
    metrics = recommender.get_summary_metrics()
    
    plot_items = [
        {
            "filename": "disease_frequency.png",
            "title": "Top Documented Conditions",
            "description": "Frequency distribution of the most prevalent conditions documented in the AyurGenixAI dataset."
        },
        {
            "filename": "top_herbs.png",
            "title": "Most Frequent Ayurvedic Herbs (Dravyas)",
            "description": "Analysis of the most widely referenced medicinal herbs across Ayurvedic formulations."
        },
        {
            "filename": "dosha_distribution.png",
            "title": "Dosha Imbalance Classification",
            "description": "Distribution of primary Vata, Pitta, and Kapha imbalances and their dual presentations."
        },
        {
            "filename": "severity_distribution.png",
            "title": "Symptom Severity Levels",
            "description": "Breakdown of documented conditions across Mild, Moderate, High, and Severe stages."
        },
        {
            "filename": "seasonal_variation.png",
            "title": "Seasonal Aggravation (Ritucharya)",
            "description": "Classification of seasonal triggers and climate factors associated with condition onset."
        },
        {
            "filename": "top_remedies.png",
            "title": "Traditional Home Remedies",
            "description": "Most common home and dietary remedy modalities recommended in the dataset."
        },
        {
            "filename": "formulation_types.png",
            "title": "Ayurvedic Delivery Formulations (Kalpana)",
            "description": "Prevalence of Ayurvedic formulation types such as Churna, Vati, Kwath, and Taila."
        },
        {
            "filename": "disease_vs_severity.png",
            "title": "Condition vs. Severity Breakdown",
            "description": "Cross-tabulation displaying severity distribution across frequent conditions."
        },
        {
            "filename": "disease_vs_season.png",
            "title": "Condition vs. Season Heatmap",
            "description": "Co-occurrence heatmap linking major condition categories to environmental seasons."
        },
        {
            "filename": "age_distribution.png",
            "title": "Target Age Demographics",
            "description": "Distribution of target age brackets associated with recorded conditions."
        },
        {
            "filename": "gender_distribution.png",
            "title": "Gender Demographics",
            "description": "Demographic distribution of condition records across gender categories."
        }
    ]

    return render_template(
        "analytics.html",
        metrics=metrics,
        plots=plot_items
    )


@app.route("/history")
def history():
    """User search history log from SQLite."""
    history_logs = db.get_history(limit=50)
    return render_template("history.html", history=history_logs)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear all past search queries from SQLite."""
    db.clear_history()
    flash("Search history has been cleared successfully.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """About page explaining Data Science methodology, syllabus coverage, and architecture."""
    metrics = recommender.get_summary_metrics()
    return render_template("about.html", metrics=metrics)


if __name__ == "__main__":
    print("Starting AyurHerb Multi-Paradigm Healthcare Server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)
