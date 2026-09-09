"""
AyurHerb - Flask Web Application (app.py)
Author: Data Science Team
Description: Core web application integrating the Ayurvedic recommendation engine,
             safety triage, SQLite search history, and exploratory data analysis dashboard.
"""

import os
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from recommendation_engine import AyurvedicRecommender, SIMILARITY_THRESHOLD
from safety import evaluate_safety
from intelligence import enrich_results, normalize_symptoms, profile_from_form, score_dosha
from plant_identifier import PlantImageIdentifier, allowed_image, validate_image
import database as db

app = Flask(__name__)
app.secret_key = os.environ.get("AYURHERB_SECRET_KEY", "development-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Initialize SQLite database
db.init_db()

# Initialize Recommendation Engine
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "ayurgenixai_cleaned.csv")
recommender = AyurvedicRecommender(data_path)
upload_dir = os.path.join(base_dir, "instance", "uploads")
os.makedirs(upload_dir, exist_ok=True)
plant_identifier = PlantImageIdentifier(os.path.join(base_dir, "models", "ayurvedic_plant_classifier"))


@app.route("/")
def index():
    """Home landing page with system overview and dataset statistics."""
    metrics = recommender.get_summary_metrics()
    popular_symptoms = recommender.get_popular_symptoms_list()[:8]
    return render_template(
        "index.html",
        metrics=metrics,
        popular_symptoms=popular_symptoms
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
    """Educational questionnaire; it is not a diagnostic assessment."""
    result = None
    if request.method == "POST":
        answers = {key: request.form.get(key, "") for key in ("body_frame", "skin", "appetite", "stress_response")}
        result = score_dosha(answers)
    return render_template("wellness_profile.html", result=result)


@app.route("/analyze", methods=["POST"])
def analyze_symptoms():
    """
    Process entered symptoms:
    1. Check for empty or invalid query.
    2. Evaluate clinical safety & emergency red-flags.
    3. Run TF-IDF & Cosine Similarity recommendation engine.
    4. Log search history into SQLite.
    5. Render results page.
    """
    query = request.form.get("symptoms", "").strip()
    profile = profile_from_form(request.form)

    if not query:
        flash("Please enter or select at least one symptom to analyze.", "warning")
        return redirect(url_for("symptom_checker"))

    # Step 1: Safety & Emergency Triage
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
        )

    # Step 2: Normalize common multilingual phrases then retrieve dataset references.
    symptom_analysis = normalize_symptoms(query)
    normalized_query = symptom_analysis["normalized_query"]
    results = recommender.recommend(query=normalized_query, top_k=6, threshold=SIMILARITY_THRESHOLD)
    results = enrich_results(results, profile, symptom_analysis["recognized_symptoms"])

    top_condition = results[0]["disease"] if results else "No Match"
    top_score = results[0]["similarity_score"] if results else 0.0

    # Step 3: Save only when the visitor explicitly opts in to local history.
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
    )


@app.route("/plant-identifier", methods=["GET", "POST"])
def plant_identifier_page():
    """Accept a plant photograph and pass it only to an installed local model."""
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
                # Images are transient input: delete after local inference.
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
    
    # List of generated plots with descriptions
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
    """About page explaining Data Science methodology, TF-IDF, dataset, and safety."""
    metrics = recommender.get_summary_metrics()
    return render_template("about.html", metrics=metrics)


if __name__ == "__main__":
    # Host on 127.0.0.1:5000
    print("Starting AyurHerb Web Server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)
