"""Fine-tuned ResNet-50 plant image classification and botanical profiles."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CURATED_BOTANICAL_PROFILES = {
    "Tulasi": {
        "display_name": "Tulsi (Holy Basil)",
        "botanical_profile_available": True,
        "sanskrit_name": "Surasa / Tulasi",
        "botanical_name": "Ocimum tenuiflorum",
        "dosha_balance": "Balances Vata & Kapha, mildly increases Pitta",
        "key_benefits": "Respiratory health, immunity boost, adaptogenic stress relief, anti-microbial",
        "safe_preparation": "Wash 4–5 fresh leaves thoroughly. Crush lightly and infuse in 1 cup of boiling water for 3–5 minutes. Drink warm with a spoonful of raw honey.",
        "dosage": "5–10 fresh leaves daily or 1–2 cups of mild infusion",
        "precautions": "Avoid excessive intake during pregnancy. May mildly lower blood sugar levels."
    },
    "Neem": {
        "display_name": "Neem (Indian Lilac)",
        "botanical_profile_available": True,
        "sanskrit_name": "Nimba / Arishta",
        "botanical_name": "Azadirachta indica",
        "dosha_balance": "Pacifies Pitta & Kapha",
        "key_benefits": "Blood purification, skin health, anti-fungal, dental hygiene, natural detox",
        "safe_preparation": "For skin wash: Boil 10–12 leaves in 500ml water until reduced by half; cool and rinse. For internal: Consume only under practitioner guidance.",
        "dosage": "External rinse as needed; 2–4 fresh tender leaves chewed in morning (short duration only)",
        "precautions": "Bitter and cooling. Strictly avoid in high doses, pregnancy, or for small infants."
    },
    "Aloevera": {
        "display_name": "Aloe Vera (Ghritkumari)",
        "botanical_profile_available": True,
        "sanskrit_name": "Ghritakumari / Kanya",
        "botanical_name": "Aloe barbadensis miller",
        "dosha_balance": "Balances all three doshas (Tridoshic, especially Pitta)",
        "key_benefits": "Skin rejuvenation, soothing burns, digestive cooling, liver support",
        "safe_preparation": "Cut fresh leaf, let yellow aloin latex drain completely for 15 mins. Wash clear inner gel with water. Apply topically or blend with cumin water.",
        "dosage": "Topical application as required; 15–20ml purified inner gel juice",
        "precautions": "Always remove the yellow latex (aloin) before internal consumption. Avoid during pregnancy."
    },
    "Mint": {
        "display_name": "Mint / Pudina",
        "botanical_profile_available": True,
        "sanskrit_name": "Putiha / Pudina",
        "botanical_name": "Mentha spicata",
        "dosha_balance": "Pacifies Pitta and Kapha, light on digestion",
        "key_benefits": "Instant digestive relief, cooling carminative, nausea remedy, oral freshness",
        "safe_preparation": "Muddle 6–8 fresh mint leaves with roasted cumin powder, black salt, and warm water for instant gas or acidity relief.",
        "dosage": "5–10 fresh leaves in tea or fresh herbal chutney",
        "precautions": "Generally safe. Avoid excessive concentrated essential oil without dilution."
    },
    "Ashwagandha": {
        "display_name": "Ashwagandha (Indian Ginseng)",
        "botanical_profile_available": True,
        "sanskrit_name": "Ashwagandha / Hayagandha",
        "botanical_name": "Withania somnifera",
        "dosha_balance": "Pacifies Vata & Kapha",
        "key_benefits": "Rasayana (rejuvenation), nervous system calming, sleep enhancement, muscle strength",
        "safe_preparation": "Traditional root powder: Mix 1/2 teaspoon (2–3g) in warm cow's milk or almond milk with a pinch of nutmeg before bedtime.",
        "dosage": "3–5g root powder daily with warm milk or water",
        "precautions": "Avoid during acute fever, severe congestion (Ama), or pregnancy."
    },
    "Ginger": {
        "display_name": "Ginger / Adrak (Shunthi)",
        "botanical_profile_available": True,
        "sanskrit_name": "Ardraka (fresh) / Shunthi (dry)",
        "botanical_name": "Zingiber officinale",
        "dosha_balance": "Pacifies Vata & Kapha, stimulates Agni (digestive fire)",
        "key_benefits": "Digestion booster, anti-inflammatory, clears throat phlegm, reduces nausea",
        "safe_preparation": "Grate 1/2 inch fresh ginger, boil in 1.5 cups water for 5 minutes. Strain, add rock salt or jaggery, and drink warm before meals.",
        "dosage": "1–3g fresh rhizome daily",
        "precautions": "Avoid excessive intake if experiencing active stomach ulcers or hyperacidity (high Pitta)."
    }
}

CLASS_PROFILE_KEYS = {
    "Aloevera": "Aloevera",
    "Ashwagandha": "Ashwagandha",
    "Mint": "Mint",
    "Neem": "Neem",
    "Tulasi": "Tulasi",
}

ADDITIONAL_BOTANICAL_PROFILES = {
    "Amla": {
        "display_name": "Amla",
        "botanical_name": "Phyllanthus emblica",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Phyllanthus%20emblica",
    },
    "Amruta_Balli": {
        "display_name": "Amruta Balli",
        "botanical_name": "Tinospora cordifolia",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Tinospora%20cordifolia",
    },
    "Arali": {
        "display_name": "Arali",
        "botanical_profile_available": False,
        "precautions": "The plant identity is ambiguous. Do not consume or use it medicinally until confirmed by a qualified botanist.",
        "profile_note": "The common name can refer to different plants; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Ashoka": {
        "display_name": "Ashoka",
        "botanical_profile_available": False,
        "precautions": "The plant identity is not confirmed from this label. Do not use it medicinally until identified by a qualified botanist.",
        "profile_note": "Several plants may be called Ashoka; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Avacado": {
        "display_name": "Avocado",
        "botanical_name": "Persea americana",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Persea%20americana",
    },
    "Bamboo": {
        "display_name": "Bamboo",
        "botanical_name": "Bambusoideae (subfamily; species not determined)",
        "profile_source": "",
    },
    "Basale": {
        "display_name": "Basale",
        "botanical_name": "Basella alba",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Basella%20alba",
    },
    "Betel": {
        "display_name": "Betel Leaf",
        "botanical_name": "Piper betle",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Piper%20betle",
    },
    "Betel_Nut": {
        "display_name": "Areca Nut (Betel Nut)",
        "botanical_name": "Areca catechu",
        "precautions": "Areca nut and betel-quid use are associated with oral cancer. Do not use as a health remedy.",
        "profile_source": "https://www.who.int/news-room/fact-sheets/detail/oral-health",
    },
    "Brahmi": {
        "display_name": "Brahmi",
        "botanical_profile_available": False,
        "precautions": "The plant identity is ambiguous. Do not consume or use it medicinally until confirmed by a qualified botanist.",
        "profile_note": "Brahmi may refer to different species; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Castor": {
        "display_name": "Castor Plant",
        "botanical_name": "Ricinus communis",
        "precautions": "Castor beans contain ricin. Chewing or swallowing the beans can cause serious poisoning; keep them away from children and pets.",
        "profile_source": "https://www.cdc.gov/chemical-emergencies/chemical-fact-sheets/ricin.html",
    },
    "Curry_Leaf": {
        "display_name": "Curry Leaf",
        "botanical_name": "Murraya koenigii",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Murraya%20koenigii",
    },
    "Doddapatre": {
        "display_name": "Doddapatre",
        "botanical_name": "Coleus amboinicus",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Coleus%20amboinicus",
    },
    "Ekka": {
        "display_name": "Ekka",
        "botanical_profile_available": False,
        "precautions": "The plant identity is not confirmed. Do not handle sap or use this plant medicinally until it has been identified by a qualified botanist.",
        "profile_note": "The local label alone is insufficient to confirm a scientific identity.",
        "profile_source": "",
    },
    "Ganike": {
        "display_name": "Ganike",
        "botanical_profile_available": False,
        "precautions": "The plant identity is not confirmed. Do not consume or use it medicinally until identified by a qualified botanist.",
        "profile_note": "A possible taxonomic match is not confirmed for this local label.",
        "profile_source": "",
    },
    "Gauva": {
        "display_name": "Guava",
        "botanical_name": "Psidium guajava",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Psidium%20guajava",
    },
    "Geranium": {
        "display_name": "Geranium",
        "botanical_profile_available": False,
        "precautions": "The plant identity is ambiguous; do not use essential oils or preparations from an unidentified plant.",
        "profile_note": "Geranium and scented geranium labels may refer to different genera; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Henna": {
        "display_name": "Henna",
        "botanical_name": "Lawsonia inermis",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Lawsonia%20inermis",
    },
    "Hibiscus": {
        "display_name": "Hibiscus",
        "botanical_name": "Hibiscus spp. (species not determined)",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Hibiscus",
    },
    "Honge": {
        "display_name": "Honge",
        "botanical_name": "Pongamia pinnata",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Pongamia%20pinnata",
    },
    "Insulin": {
        "display_name": "Insulin Plant",
        "botanical_profile_available": False,
        "precautions": "Do not use as a diabetes treatment or replace prescribed medication. The common name does not establish plant identity or clinical effect.",
        "profile_note": "“Insulin plant” is used for multiple unrelated species; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Jasmine": {
        "display_name": "Jasmine",
        "botanical_name": "Jasminum spp. (species not determined)",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Jasminum",
    },
    "Lemon": {
        "display_name": "Lemon",
        "botanical_name": "Citrus × limon",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Citrus%20limon",
    },
    "Lemon_grass": {
        "display_name": "Lemongrass",
        "botanical_name": "Cymbopogon spp. (species not determined)",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Cymbopogon",
    },
    "Mango": {
        "display_name": "Mango",
        "botanical_name": "Mangifera indica",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Mangifera%20indica",
    },
    "Nagadali": {
        "display_name": "Nagadali",
        "botanical_profile_available": False,
        "precautions": "The plant identity is unverified. Do not consume or use it medicinally until identified by a qualified botanist.",
        "profile_note": "No reliable scientific identity was established for this label.",
        "profile_source": "",
    },
    "Nithyapushpa": {
        "display_name": "Nithyapushpa (Madagascar Periwinkle)",
        "botanical_name": "Catharanthus roseus",
        "precautions": "This plant contains potent alkaloids and is not for self-medication or ingestion.",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Catharanthus%20roseus",
    },
    "Nooni": {
        "display_name": "Noni",
        "botanical_name": "Morinda citrifolia",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Morinda%20citrifolia",
    },
    "Pappaya": {
        "display_name": "Papaya",
        "botanical_name": "Carica papaya",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Carica%20papaya",
    },
    "Pepper": {
        "display_name": "Pepper",
        "botanical_profile_available": False,
        "precautions": "The plant identity is ambiguous. Do not use medicinally until the plant is botanically identified.",
        "profile_note": "“Pepper” can refer to unrelated plants; no scientific identity is assigned.",
        "profile_source": "",
    },
    "Pomegranate": {
        "display_name": "Pomegranate",
        "botanical_name": "Punica granatum",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Punica%20granatum",
    },
    "Raktachandini": {
        "display_name": "Raktachandini",
        "botanical_profile_available": False,
        "precautions": "The plant identity is not confirmed. Do not use medicinally until identified by a qualified botanist.",
        "profile_note": "Red sandalwood is a possible name match, but the classifier label is not verified against that species.",
        "profile_source": "",
    },
    "Rose": {
        "display_name": "Rose",
        "botanical_name": "Rosa spp. (species not determined)",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Rosa",
    },
    "Sapota": {
        "display_name": "Sapodilla",
        "botanical_name": "Manilkara zapota",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Manilkara%20zapota",
    },
    "Wood_sorel": {
        "display_name": "Wood Sorrel",
        "botanical_name": "Oxalis spp. (species not determined)",
        "profile_source": "https://api.gbif.org/v1/species/match?name=Oxalis",
    },
}

def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(image_path: str) -> bool:
    """Verify that an upload is a decodable image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


class PlantImageIdentifier:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or os.path.join(BASE_DIR, "models")
        self.model_path = os.path.join(self.model_dir, "plant_resnet50_finetuned.pth")
        self.class_names_path = os.path.join(self.model_dir, "class_names.json")
        self.class_names: list[str] = []
        self.botanical_profiles: dict[str, Dict[str, Any]] = {}
        self.model: Any = None
        self.device: Any = None
        self.transform: Any = None
        self.torch: Any = None
        self.load_error: Exception | None = None
        self._load_model()

    def _load_model(self) -> None:
        print("[PLANT MODEL] Loading fine-tuned ResNet-50...")
        print(f"[PLANT MODEL] Model: {os.path.abspath(self.model_path)}")
        try:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            if not os.path.isfile(self.class_names_path):
                raise FileNotFoundError(f"Class names file not found: {self.class_names_path}")

            with open(self.class_names_path, encoding="utf-8") as class_names_file:
                class_names = json.load(class_names_file)
            if not isinstance(class_names, list) or len(class_names) != 40:
                raise ValueError("class_names.json must contain exactly 40 class names.")
            if not all(isinstance(name, str) and name.strip() for name in class_names):
                raise ValueError("class_names.json must contain 40 non-empty string class names.")
            self.class_names = class_names
            self.botanical_profiles = {
                class_name: {
                    "display_name": class_name.replace("_", " "),
                    "botanical_profile_available": False,
                    "botanical_name": "",
                    "sanskrit_name": "",
                    "dosha_balance": "",
                    "key_benefits": "",
                    "safe_preparation": "",
                    "dosage": "",
                    "precautions": "",
                    "profile_note": "",
                    "profile_source": "",
                }
                for class_name in self.class_names
            }
            self.botanical_profiles.update({
                class_name: {
                    **self.botanical_profiles[class_name],
                    **profile,
                    "botanical_profile_available": profile.get(
                        "botanical_profile_available",
                        bool(profile.get("botanical_name")),
                    ),
                }
                for class_name, profile in ADDITIONAL_BOTANICAL_PROFILES.items()
                if class_name in self.botanical_profiles
            })
            self.botanical_profiles.update({
                class_name: {
                    **self.botanical_profiles[class_name],
                    **CURATED_BOTANICAL_PROFILES[profile_key],
                }
                for class_name, profile_key in CLASS_PROFILE_KEYS.items()
                if class_name in self.botanical_profiles
            })
            print(f"[PLANT MODEL] Classes: {len(self.class_names)}")

            import torch
            from torch import nn
            from torchvision import models, transforms

            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[PLANT MODEL] Device: {self.device}")

            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(self.class_names))
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self.model = model
            print("[PLANT MODEL] ResNet-50 loaded successfully.")
        except Exception as error:
            self.load_error = error
            print(f"[PLANT MODEL] Failed to load ResNet-50: {error}")

    @property
    def is_ready(self) -> bool:
        """Whether the fine-tuned ResNet-50 was loaded successfully."""
        return self.model is not None

    def identify(self, image_path: str) -> Dict[str, Any]:
        """Classify an image with the loaded fine-tuned ResNet-50."""
        if self.model is None:
            detail = f": {self.load_error}" if self.load_error else ""
            raise RuntimeError(f"The plant ResNet-50 model is unavailable{detail}")
        if not validate_image(image_path):
            raise ValueError("The uploaded file is not a valid decodable image.")

        print("[PLANT MODEL] Running ResNet-50 inference...")
        with Image.open(image_path) as image:
            input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = self.torch.softmax(logits, dim=1)
            top_probabilities, top_indices = self.torch.topk(probabilities, k=3, dim=1)

        candidates = []
        for probability, index in zip(
            top_probabilities[0].tolist(),
            top_indices[0].tolist(),
        ):
            class_name = self.class_names[index]
            profile = self.botanical_profiles[class_name]
            candidate = {
                **profile,
                "class_name": class_name,
                "confidence": round(float(probability) * 100, 2),
            }
            candidates.append(candidate)

        if candidates:
            print(f"[PLANT MODEL] Top prediction: {candidates[0]['class_name']}")
            print(f"[PLANT MODEL] Confidence: {candidates[0]['confidence']}%")

        return {
            "status": "identified",
            "message": "Identified using fine-tuned ResNet-50 classifier.",
            "candidates": candidates,
        }
