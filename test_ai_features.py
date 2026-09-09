"""Non-destructive smoke tests for the AI enhancement routes."""
import io
import unittest

from PIL import Image

from app import app
from intelligence import normalize_symptoms, score_dosha


class AIFeatureTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_multilingual_normalization(self):
        result = normalize_symptoms("मुझे खांसी और bukhar है")
        self.assertIn("cough", result["normalized_query"])
        self.assertIn("fever", result["normalized_query"])

    def test_educational_dosha_score(self):
        result = score_dosha({"body_frame": "light", "skin": "dry", "appetite": "irregular", "stress_response": "anxious"})
        self.assertEqual(result["indicative_dosha"], "Vata")

    def test_enhanced_routes_render(self):
        for path in ("/", "/symptom-checker", "/wellness-profile", "/plant-identifier", "/analytics"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_profile_and_explainable_analysis_render(self):
        profile = self.client.post("/wellness-profile", data={
            "body_frame": "light", "skin": "dry", "appetite": "irregular", "stress_response": "anxious",
        })
        self.assertIn(b"Indicative pattern: Vata", profile.data)
        response = self.client.post("/analyze", data={"symptoms": "khansi and bukhar", "season": "Winter", "dosha": "Vata"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"What the system understood", response.data)

    def test_plant_upload_rejects_invalid_image(self):
        response = self.client.post("/plant-identifier", data={"plant_image": (io.BytesIO(b"not an image"), "plant.jpg")}, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"not a valid decodable image", response.data)

    def test_valid_plant_upload_reports_local_model_status(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(image_buffer, format="PNG")
        image_buffer.seek(0)
        response = self.client.post("/plant-identifier", data={"plant_image": (image_buffer, "leaf.png")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"no classifier has been installed yet", response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
