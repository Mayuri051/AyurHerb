"""Focused tests for combined SQLite result history and plant-image retention."""

import io
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import app as app_module
import database as db


class ResultsHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "history.db")
        self.db_path_patcher = patch.object(db, "get_db_path", return_value=self.database_path)
        self.db_path_patcher.start()
        db.init_db()
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        image_paths = db.get_history_image_paths()
        db.clear_history()
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            if filename == image_path:
                try:
                    os.remove(os.path.join(app_module.upload_dir, filename))
                except FileNotFoundError:
                    pass
        self.db_path_patcher.stop()
        self.temp_dir.cleanup()

    def test_empty_state_and_symptom_result_details(self):
        empty_response = self.client.get("/history")
        self.assertEqual(empty_response.status_code, 200)
        self.assertIn(b"No results yet", empty_response.data)
        self.assertIn(b"Check Symptoms", empty_response.data)
        self.assertIn(b"Scan a Plant", empty_response.data)

        response = self.client.post(
            "/analyze",
            data={"symptoms": "cough and sore throat"},
        )
        self.assertEqual(response.status_code, 200)
        records = db.get_results_history(record_type="symptom")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["query"], "cough and sore throat")
        self.assertIn("recommendation", record["details"])
        self.assertNotIn("ml_prediction", record["details"])

        detail_response = self.client.get(f"/history/{record['id']}")
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn(b"Educational Wellness Disclaimer", detail_response.data)
        self.assertIn(b"cough and sore throat", detail_response.data)
        self.assertEqual(self.client.get("/history?type=plant").status_code, 200)

    def test_plant_scan_persists_image_and_deletes_only_that_record(self):
        candidate = {
            "display_name": "Tulsi",
            "botanical_name": "Ocimum tenuiflorum",
            "confidence": 94.0,
            "sanskrit_name": "Tulasi",
            "dosha_balance": "Balances Vata and Kapha",
            "key_benefits": "Traditional wellness reference",
            "safe_preparation": "Infuse in hot water.",
            "dosage": "As advised.",
            "precautions": "Verify identification.",
        }
        image_bytes = io.BytesIO()
        Image.new("RGB", (16, 16), color=(40, 120, 60)).save(image_bytes, format="PNG")
        image_bytes.seek(0)

        with patch.object(
            app_module.plant_identifier,
            "identify",
            return_value={
                "status": "identified",
                "message": "Test identification",
                "candidates": [candidate],
            },
        ):
            response = self.client.post(
                "/plant-identifier",
                data={"plant_image": (image_bytes, "tulsi.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        records = db.get_results_history(record_type="plant")
        self.assertEqual(len(records), 1)
        record = records[0]
        image_path = os.path.join(app_module.upload_dir, record["image_path"])
        self.assertTrue(os.path.isfile(image_path))
        self.assertEqual(record["scientific_name"], "Ocimum tenuiflorum")
        self.assertEqual(record["confidence"], 94.0)
        self.assertEqual(
            self.client.get(f"/history/{record['id']}/image").status_code,
            200,
        )
        details_response = self.client.get(f"/history/{record['id']}")
        self.assertIn(b"Precautions", details_response.data)
        self.assertIn(b"Verify identification.", details_response.data)

        symptom_id = db.log_symptom_result(
            query="headache",
            prediction="Headache",
            herbs="Tulsi",
            similarity_score=0.5,
            results_count=1,
            details={},
        )
        delete_response = self.client.post(
            f"/history/{record['id']}/delete",
            follow_redirects=True,
        )
        self.assertEqual(delete_response.status_code, 200)
        self.assertFalse(os.path.exists(image_path))
        self.assertIsNone(db.get_result_history(record["id"]))
        self.assertIsNotNone(db.get_result_history(symptom_id))

        clear_response = self.client.post("/history/clear", follow_redirects=True)
        self.assertEqual(clear_response.status_code, 200)
        self.assertEqual(db.get_results_history(), [])

    def test_existing_search_history_table_is_migrated(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    top_condition TEXT,
                    similarity_score REAL,
                    results_count INTEGER DEFAULT 0,
                    safety_triggered INTEGER DEFAULT 0
                )
            """)
            connection.execute(
                "INSERT INTO search_history (query, top_condition) VALUES (?, ?)",
                ("legacy cough query", "Cough"),
            )

        db.init_db()
        records = db.get_results_history(record_type="symptom")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query"], "legacy cough query")
        self.assertEqual(records[0]["record_type"], "symptom")
        self.assertEqual(records[0]["details"], {})


if __name__ == "__main__":
    unittest.main()
