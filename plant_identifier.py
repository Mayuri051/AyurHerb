"""Secure, local-model integration point for Ayurvedic plant image recognition.

The application never guesses a plant when a trained local classifier is absent.
To enable classification, place a Hugging Face image-classification model and a
labels.json file in models/ayurvedic_plant_classifier/.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
PLANT_KNOWLEDGE = {
    "tulsi": {"display_name": "Tulsi (Holy Basil)", "botanical_name": "Ocimum tenuiflorum", "traditional_use": "Traditionally documented for respiratory wellness and daily herbal preparations."},
    "ashwagandha": {"display_name": "Ashwagandha", "botanical_name": "Withania somnifera", "traditional_use": "Traditionally documented in wellness routines associated with stress and vitality."},
    "neem": {"display_name": "Neem", "botanical_name": "Azadirachta indica", "traditional_use": "Traditionally documented in topical and general wellness practices."},
    "aloe_vera": {"display_name": "Aloe Vera", "botanical_name": "Aloe barbadensis miller", "traditional_use": "Traditionally documented for topical skin-care uses."},
    "ginger": {"display_name": "Ginger", "botanical_name": "Zingiber officinale", "traditional_use": "Traditionally documented in digestive and warming preparations."},
}


def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(image_path: str) -> bool:
    """Verify that an upload is a decodable image, not only a renamed file."""
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


class PlantImageIdentifier:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.labels_path = os.path.join(model_dir, "labels.json")

    @property
    def is_ready(self) -> bool:
        return os.path.isfile(self.labels_path) and os.path.isdir(self.model_dir)

    def identify(self, image_path: str) -> Dict[str, Any]:
        """Run a local Transformers pipeline only when the trained model exists."""
        if not self.is_ready:
            return {
                "status": "model_unavailable",
                "message": "Plant recognition is ready for a local trained model, but no classifier has been installed yet.",
                "candidates": [],
            }
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
            model = AutoModelForImageClassification.from_pretrained(self.model_dir, local_files_only=True)
            with Image.open(image_path) as image:
                inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                probabilities = torch.softmax(model(**inputs).logits[0], dim=-1)
            with open(self.labels_path, encoding="utf-8") as labels_file:
                labels = json.load(labels_file)
            label_names = labels if isinstance(labels, list) else labels.get("labels", [])
            ranked = torch.topk(probabilities, k=min(3, len(label_names)))
            predictions = [
                {"label": label_names[index], "score": score}
                for score, index in zip(ranked.values.tolist(), ranked.indices.tolist())
            ]
            candidates: List[Dict[str, Any]] = []
            for prediction in predictions:
                key = prediction["label"].lower().replace(" ", "_")
                info = PLANT_KNOWLEDGE.get(key, {"display_name": prediction["label"], "botanical_name": "Not documented", "traditional_use": "No local plant note is available."})
                candidates.append({**info, "confidence": round(float(prediction["score"]) * 100, 1)})
            return {"status": "identified", "message": "Local model prediction. Verify the plant with a qualified botanist before use.", "candidates": candidates}
        except Exception as error:
            return {"status": "model_error", "message": f"The local classifier could not run: {error}", "candidates": []}
