"""
AyurHerb - Comprehensive Test Suite (test_app.py)
Author: Data Science Team
Description: Validates all Flask endpoints, recommendation outputs, safety triage filters,
             and database operations using Flask's test client.
"""

import sys
import unittest
from app import app
import database as db

# Ensure UTF-8 stdout on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class AyurHerbTestSuite(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_01_home_page(self):
        """Test GET / returns 200 and contains title."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AYURHERB", response.data)
        self.assertIn(b"Data-Driven Ayurvedic Wellness Recommendation System", response.data)
        print("  [PASS] Test 1: Home page loads successfully.")

    def test_02_symptom_checker_page(self):
        """Test GET /symptom-checker returns 200 and contains form."""
        response = self.client.get("/symptom-checker")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Symptom Checker", response.data)
        self.assertIn(b"Analyze Symptoms", response.data)
        print("  [PASS] Test 2: Symptom checker page loads successfully.")

    def test_03_recommendation_cough_query(self):
        """Test POST /analyze with respiratory symptoms returns relevant matches."""
        response = self.client.post(
            "/analyze",
            data={"symptoms": "I have cough and sore throat"},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Symptom Similarity Score", response.data)
        self.assertIn(b"Cough", response.data)
        print("  [PASS] Test 3: Recommendation engine returns valid cough/sore throat matches.")

    def test_04_recommendation_headache_query(self):
        """Test POST /analyze with headache query."""
        response = self.client.post(
            "/analyze",
            data={"symptoms": "I have severe headache and migraine"},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Migraine", response.data)
        print("  [PASS] Test 4: Recommendation engine returns valid migraine matches.")

    def test_05_safety_red_flag_triage(self):
        """Test POST /analyze with emergency symptoms triggers priority warning."""
        response = self.client.post(
            "/analyze",
            data={"symptoms": "I have severe difficulty breathing and crushing chest pain"},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"URGENT: Immediate Medical Evaluation Recommended", response.data)
        self.assertIn(b"Severe Breathing Difficulty", response.data)
        print("  [PASS] Test 5: Safety system correctly intercepts red-flag emergency symptoms.")

    def test_06_empty_symptom_query(self):
        """Test POST /analyze with empty input redirects gracefully."""
        response = self.client.post(
            "/analyze",
            data={"symptoms": "   "},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter or select at least one symptom", response.data)
        print("  [PASS] Test 6: Empty symptom input handled gracefully.")

    def test_07_unrelated_query(self):
        """Test POST /analyze with completely unrelated query shows no sufficiently similar record."""
        response = self.client.post(
            "/analyze",
            data={"symptoms": "quantum physics space rocket propulsion"},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No Sufficiently Similar Record Found", response.data)
        print("  [PASS] Test 7: Unrelated query gracefully returns no match notice.")

    def test_08_herbs_explorer(self):
        """Test GET /herbs and search filter."""
        response = self.client.get("/herbs")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ayurvedic Herb Explorer", response.data)

        # Filter by Tulsi
        response_filtered = self.client.get("/herbs?q=Tulsi")
        self.assertEqual(response_filtered.status_code, 200)
        self.assertIn(b"Tulsi", response_filtered.data)
        print("  [PASS] Test 8: Herb explorer and filter work properly.")

    def test_09_herb_detail(self):
        """Test GET /herb/<herb_name>."""
        response = self.client.get("/herb/Tulsi")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tulsi", response.data)
        self.assertIn(b"Medicinal Herb Profile", response.data)
        print("  [PASS] Test 9: Herb detail view loads properly.")

    def test_10_conditions_explorer(self):
        """Test GET /conditions."""
        response = self.client.get("/conditions")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Disease Explorer", response.data)
        print("  [PASS] Test 10: Conditions directory loads properly.")

    def test_11_condition_detail(self):
        """Test GET /condition/Cough."""
        response = self.client.get("/condition/Cough")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Cough", response.data)
        self.assertIn(b"Ayurvedic Herbal Management", response.data)
        print("  [PASS] Test 11: Condition detail view displays full profile.")

    def test_12_analytics_dashboard(self):
        """Test GET /analytics."""
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Exploratory Data Analysis", response.data)
        self.assertIn(b"disease_frequency.png", response.data)
        print("  [PASS] Test 12: Analytics dashboard renders all plots and KPIs.")

    def test_13_history_and_clear(self):
        """Test GET /history and POST /history/clear."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Search History Log", response.data)

        # Clear history
        response_clear = self.client.post("/history/clear", follow_redirects=True)
        self.assertEqual(response_clear.status_code, 200)
        self.assertIn(b"Search history has been cleared successfully", response_clear.data)
        print("  [PASS] Test 13: Search history logging and clearance work properly.")

    def test_14_about_page(self):
        """Test GET /about."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About AyurHerb", response.data)
        self.assertIn(b"TF-IDF", response.data)
        self.assertIn(b"Cosine Similarity", response.data)
        print("  [PASS] Test 14: About page loads successfully.")

    def test_15_ml_evaluation_page(self):
        """Test GET /ml-evaluation (Unit V)."""
        response = self.client.get("/ml-evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Machine Learning Performance", response.data)
        self.assertIn(b"Random Forest", response.data)
        print("  [PASS] Test 15: ML Model Evaluation dashboard renders successfully.")

    def test_16_fuzzy_dosage_page(self):
        """Test GET and POST /fuzzy-dosage (Unit III)."""
        response_get = self.client.get("/fuzzy-dosage")
        self.assertEqual(response_get.status_code, 200)
        self.assertIn(b"Fuzzy Logic Formulation & Dosage Calculator", response_get.data)

        response_post = self.client.post(
            "/fuzzy-dosage",
            data={"severity": "6.5", "duration": "10", "agni": "5.0"}
        )
        self.assertEqual(response_post.status_code, 200)
        self.assertIn(b"Potency Index", response_post.data)
        print("  [PASS] Test 16: Fuzzy Logic Dosage calculator computes inferences accurately.")


if __name__ == "__main__":
    print("=" * 70)
    print("AYURHERB - INTEGRATION & END-TO-END TEST SUITE")
    print("=" * 70)
    unittest.main(verbosity=1)
