"""
AyurHerb - Flask Web Application (app.py)
Author: Data Science Team
Description: Multi-paradigm cognitive healthcare web application integrating:
             1. NLP & TF-IDF Semantic Retrieval
             2. Fuzzy Logic Dosage Controller (Mamdani FIS) [Unit III]
             3. Bayesian Network Probabilistic Inference [Unit I]
             4. Clinical Safety & Emergency Triage
             5. SQLite Results History
"""

import os
from datetime import datetime
from uuid import uuid4
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename
from recommendation_engine import AyurvedicRecommender, SIMILARITY_THRESHOLD
from safety import evaluate_safety
from intelligence import enrich_results, normalize_symptoms, profile_from_form, score_dosha
from plant_identifier import PlantImageIdentifier, allowed_image, validate_image
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

# 2. Fuzzy Logic Inference Engine (Mamdani FIS - Unit III)
fuzzy_engine = AyurvedicFuzzyInferenceSystem()

# 3. Bayesian Probabilistic Network (Unit I)
bayesian_engine = AyurvedicBayesianNetwork(data_path)

# 4. Plant Identifier (Deep Learning / Vision hook - Unit IV)
plant_identifier = PlantImageIdentifier(os.path.join(base_dir, "models"))


PLANT_HERB_NAME_ALIASES = {
    "Aloevera": "Aloe Vera",
    "Avacado": "Avocado",
    "Betel_Nut": "Betel Nut",
    "Curry_Leaf": "Curry Leaf",
    "Lemon_grass": "Lemongrass",
    "Pappaya": "Papaya",
    "Tulasi": "Tulsi",
    "Wood_sorel": "Wood Sorrel",
}


def _attach_herb_explorer_details(prediction):
    """Attach exact herb-dataset matches without changing model predictions."""
    available_herbs = {
        " ".join(herb["name"].casefold().split()): herb
        for herb in recommender.get_all_herbs()
    }
    for candidate in prediction.get("candidates", []):
        class_name = candidate.get("class_name", "")
        herb_name = PLANT_HERB_NAME_ALIASES.get(
            class_name,
            class_name.replace("_", " "),
        )
        herb = available_herbs.get(" ".join(herb_name.casefold().split()))
        if not herb:
            continue

        details = recommender.get_herb_detail(herb["name"])
        candidate["herb_explorer_query"] = herb["name"]
        candidate["associated_conditions"] = details["conditions"] if details else []
    return prediction


@app.route("/")
def index():
    """Home landing page with system overview and dataset stats."""
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
    """Educational questionnaire; non-diagnostic assessment."""
    result = None
    if request.method == "POST":
        answers = {key: request.form.get(key, "") for key in ("body_frame", "skin", "appetite", "stress_response")}
        result = score_dosha(answers)
    return render_template("wellness_profile.html", result=result)


@app.route("/analyze", methods=["POST"])
def analyze_symptoms():
    """
    Symptom analysis pipeline:
    1. Safety & Emergency Triage
    2. Multilingual Normalization
    3. TF-IDF Semantic Ranking
    4. Bayesian Probabilistic Inference (Unit I)
    5. Fuzzy Logic Dosage Calculation (Unit III)
    6. SQLite History Logging
    """
    query = request.form.get("symptoms", "").strip()
    profile = profile_from_form(request.form)

    # Extract optional fuzzy tuning parameters
    raw_sev = str(request.form.get("severity_score", "5.5")).strip().lower()
    if "mild" in raw_sev or raw_sev in ["1", "2", "3"]:
        severity_val = 3.0
    elif "severe" in raw_sev or "high" in raw_sev or raw_sev in ["8", "9", "10"]:
        severity_val = 8.5
    elif "mod" in raw_sev or raw_sev in ["4", "5", "6", "7"]:
        severity_val = 5.5
    else:
        try:
            severity_val = float(raw_sev)
        except (ValueError, TypeError):
            severity_val = 5.5

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
        history_id = db.log_symptom_result(
            query=query,
            prediction="Emergency Medical Warning Triggered",
            herbs="",
            similarity_score=0.0,
            results_count=0,
            safety_triggered=True,
            details={
                "is_emergency": True,
                "safety": {
                    "emergency_message": safety_result.get("emergency_message", ""),
                    "detected_flags": safety_result.get("detected_flags", []),
                    "advice": safety_result.get("advice", ""),
                },
            },
        )
        if history_id < 0:
            flash("Your safety result was shown, but could not be saved to Results.", "warning")
        return render_template(
            "results.html",
            query=query,
            is_emergency=True,
            safety=safety_result,
            results=[],
            threshold=SIMILARITY_THRESHOLD,
            symptom_analysis={"recognized_symptoms": [], "applied_aliases": []},
            profile=profile,
            bayesian_result=None,
            fuzzy_result=None
        )

    # Stage 2: Normalization & TF-IDF Retrieval
    symptom_analysis = normalize_symptoms(query)
    normalized_query = symptom_analysis["normalized_query"]
    results = recommender.recommend(query=normalized_query, top_k=6, threshold=SIMILARITY_THRESHOLD)
    results = enrich_results(results, profile, symptom_analysis["recognized_symptoms"])

    # Stage 3: Bayesian Network Probabilistic Inference (Unit I)
    bayesian_result = bayesian_engine.infer(
        symptoms_query=normalized_query,
        dosha_evidence=profile.get("dosha"),
        season_evidence=profile.get("season"),
        top_k=3
    )

    # Stage 4: Fuzzy Logic Dosage & Formulation Potency (Unit III)
    fuzzy_result = fuzzy_engine.infer(
        severity_score=severity_val,
        duration_days=duration_val,
        agni_score=6.0
    )

    top_condition = results[0]["disease"] if results else "No Direct Match"
    top_score = results[0]["similarity_score"] if results else 0.0

    # Save only the symptom input and the result details required for history.
    primary_result = results[0] if results else {}
    history_fields = (
        "disease", "hindi_name", "marathi_name", "similarity_score",
        "similarity_percentage", "symptoms", "symptom_severity",
        "duration_of_treatment", "ayurvedic_herbs", "herbal_remedies",
        "formulation", "doshas", "prakriti", "yoga_and_therapy",
        "diet_and_lifestyle", "prevention", "medical_intervention",
    )
    history_details = {
        "is_emergency": False,
        "recommendation": {
            key: primary_result[key]
            for key in history_fields
            if key in primary_result
        },
    }
    history_id = db.log_symptom_result(
        query=query,
        prediction=top_condition,
        herbs=str(primary_result.get("ayurvedic_herbs", "")),
        similarity_score=top_score,
        results_count=len(results),
        details=history_details,
    )
    if history_id < 0:
        flash("Your results were shown, but could not be saved to Results.", "warning")

    return render_template(
        "results.html",
        query=query,
        is_emergency=False,
        safety=safety_result,
        results=results,
        threshold=SIMILARITY_THRESHOLD,
        symptom_analysis=symptom_analysis,
        profile=profile,
        bayesian_result=bayesian_result,
        fuzzy_result=fuzzy_result
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
            keep_image = False
            try:
                if validate_image(image_path):
                    try:
                        prediction = plant_identifier.identify(image_path)
                        prediction = _attach_herb_explorer_details(prediction)
                    except (RuntimeError, ValueError) as error:
                        app.logger.exception("Plant image identification failed")
                        flash(f"Plant identification failed: {error}", "danger")
                    else:
                        if prediction.get("candidates"):
                            candidate = prediction["candidates"][0]
                            history_id = db.log_plant_result(
                                plant_name=str(candidate.get("display_name", "Unknown plant")),
                                scientific_name=str(candidate.get("botanical_name", "")),
                                confidence=float(candidate.get("confidence", 0.0)),
                                image_path=filename,
                                details=candidate,
                            )
                            if history_id > 0:
                                keep_image = True
                            else:
                                flash("The plant was identified, but could not be saved to Results.", "warning")
                else:
                    flash("That file is not a valid decodable image.", "warning")
            finally:
                if not keep_image:
                    try:
                        os.remove(image_path)
                    except OSError:
                        app.logger.exception("Unable to remove temporary plant scan image")
    return render_template(
        "plant_identifier.html",
        prediction=prediction,
        model_ready=plant_identifier.is_ready,
        class_names=plant_identifier.class_names,
    )


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
    """Combined local history of symptom checks and plant scans."""
    requested_type = request.args.get("type", "all").strip().lower()
    selected_type = requested_type if requested_type in {"all", "symptom", "plant"} else "all"
    history_records = db.get_results_history(
        record_type=None if selected_type == "all" else selected_type,
        limit=100,
    )
    for record in history_records:
        record["formatted_timestamp"] = _format_history_timestamp(record.get("timestamp"))
    return render_template(
        "history.html",
        history=history_records,
        selected_type=selected_type,
    )


@app.route("/history/<int:history_id>")
def history_detail(history_id):
    """Show the saved details for one symptom check or plant scan."""
    record = db.get_result_history(history_id)
    if record is None:
        abort(404)
    record["formatted_timestamp"] = _format_history_timestamp(record.get("timestamp"))
    return render_template("history_detail.html", record=record)


@app.route("/history/<int:history_id>/image")
def history_image(history_id):
    """Serve an image only when it belongs to a saved plant-scan record."""
    record = db.get_result_history(history_id)
    if not record or record.get("record_type") != "plant" or not record.get("image_path"):
        abort(404)
    filename = os.path.basename(record["image_path"])
    if filename != record["image_path"]:
        abort(404)
    return send_from_directory(upload_dir, filename)


def _remove_history_image(image_path):
    """Remove a stored upload by its database-recorded filename only."""
    filename = os.path.basename(image_path or "")
    if not filename or filename != image_path:
        return
    try:
        os.remove(os.path.join(upload_dir, filename))
    except FileNotFoundError:
        return
    except OSError:
        app.logger.exception("Unable to remove plant image for deleted history")


def _format_history_timestamp(value):
    if not value:
        return "Date unavailable"
    try:
        return datetime.fromisoformat(value).strftime("%d %B %Y, %I:%M %p")
    except (TypeError, ValueError):
        return str(value)


@app.route("/history/<int:history_id>/delete", methods=["POST"])
def delete_history_record(history_id):
    """Delete one saved history entry and only its associated scan image."""
    record = db.get_result_history(history_id)
    if record is None:
        abort(404)
    if db.delete_result_history(history_id):
        _remove_history_image(record.get("image_path"))
        flash("The selected result has been deleted.", "info")
    else:
        flash("The selected result could not be deleted.", "warning")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear saved history records and their associated plant-scan images."""
    image_paths = db.get_history_image_paths()
    if db.clear_history():
        for image_path in image_paths:
            _remove_history_image(image_path)
        flash("Search history has been cleared successfully.", "info")
    else:
        flash("History could not be cleared. Please try again.", "warning")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """About page explaining Data Science methodology, syllabus coverage, and architecture."""
    metrics = recommender.get_summary_metrics()
    return render_template("about.html", metrics=metrics)


if __name__ == "__main__":
    print("Starting AyurHerb Multi-Paradigm Healthcare Server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)
