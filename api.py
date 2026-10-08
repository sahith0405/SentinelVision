import json
import tempfile
from pathlib import Path

import cv2
import joblib
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.backbone import WideResNet50Backbone
from src.localization import PatchCoreLocalizer
from src.patchcore import PatchCore
from src.patchcore.features import PatchFeatureExtractor
from src.utils import get_transform

BASE_DIR = Path(__file__).resolve().parent
HEATMAP_DIR = BASE_DIR / "results" / "heatmaps" / "api"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

app = FastAPI(title="SentinelVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform()
models = {}

category_classifier = joblib.load(BASE_DIR / "models" / "category_classifier.joblib")
category_feature_extractor = PatchFeatureExtractor(WideResNet50Backbone().to(device))




def get_model(category):
    if category not in models:
        backbone = WideResNet50Backbone().to(device)
        model = PatchCore(backbone)
        memory_path = BASE_DIR / "models" / f"{category}_memory_bank.npy"

        if not memory_path.exists():
            raise HTTPException(status_code=404, detail="Memory bank not found")

        model.load_memory_bank(memory_path)
        models[category] = model

    return models[category]

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "categories": len(CATEGORIES)
    }

@app.get("/api/evaluation")
def evaluation():
    metrics_dir = BASE_DIR / "results" / "metrics"
    pixel_path = metrics_dir / "all_categories_pixel_localization.json"

    with pixel_path.open() as f:
        pixel_data = json.load(f)

    pixel_map = {item["category"]: item for item in pixel_data}
    results = []

    for category in CATEGORIES:
        metrics_path = metrics_dir / f"{category}_metrics.json"
        threshold_path = metrics_dir / f"{category}_threshold.json"

        with metrics_path.open() as f:
            metrics = json.load(f)

        with threshold_path.open() as f:
            threshold_data = json.load(f)

        pixel = pixel_map.get(category, {})

        results.append({
            "category": category,
            "image_auroc": round(float(metrics["image_auroc"]) * 100, 2),
            "average_precision": round(float(metrics["average_precision"]) * 100, 2),
            "pixel_auroc": round(float(pixel.get("pixel_auroc", 0)) * 100, 2),
            "test_samples": int(metrics.get("num_samples", pixel.get("test_samples", 0))),
            "threshold": round(float(threshold_data["threshold"]), 4)
        })

    return {"categories": results}

@app.post("/api/inspect")
async def inspect(
    file: UploadFile = File(...),
    category: str = Form(...),
):
    suffix = Path(file.filename or ".jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp_path = Path(temp.name)
        temp.write(await file.read())

    try:
        image = Image.open(temp_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        if category not in CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid product category")

        predicted_category = category
        category_confidence = 100.0

        model = get_model(category)
        result = model.predict(tensor)

        score = float(result["image_scores"][0])
        patch_scores = result["patch_scores"][0]

        threshold_path = (
            BASE_DIR / "results" / "metrics" / f"{category}_threshold.json"
        )

        with threshold_path.open() as f:
            threshold = float(json.load(f)["threshold"])

        status = "ANOMALY" if score >= threshold else "NORMAL"
        deviation = max(0.0, ((score - threshold) / threshold) * 100.0)

        patch_scores = [float(x) for x in patch_scores]
        total_patches = len(patch_scores)
        top_patch_score = max(patch_scores) if patch_scores else 0.0
        high_score_patches = [x for x in patch_scores if x >= threshold]
        high_score_patch_count = len(high_score_patches)
        high_score_patch_ratio = (high_score_patch_count / total_patches * 100.0) if total_patches else 0.0
        score_ratio = score / threshold if threshold > 0 else 0.0
        if status == "NORMAL":
            severity = "LOW"
            concentration = "LIMITED"
        elif score_ratio >= 3.0 or high_score_patch_ratio >= 10.0:
            severity = "CRITICAL"
            concentration = "WIDESPREAD"
        elif score_ratio >= 2.0 or high_score_patch_ratio >= 5.0:
            severity = "HIGH"
            concentration = "SIGNIFICANT"
        elif score_ratio >= 1.25 or high_score_patch_ratio >= 2.0:
            severity = "MODERATE"
            concentration = "LOCALIZED"
        else:
            severity = "LOW"
            concentration = "LIMITED"

        metrics_path = BASE_DIR / "results" / "metrics" / f"{category}_metrics.json"
        pixel_path = BASE_DIR / "results" / "metrics" / "all_categories_pixel_localization.json"

        with metrics_path.open() as f:
            metrics = json.load(f)

        with pixel_path.open() as f:
            pixel_metrics = json.load(f)

        pixel_result = next(
            (item for item in pixel_metrics if item["category"] == category),
            None
        )

        original = cv2.imread(str(temp_path))
        localizer = PatchCoreLocalizer(output_size=(224, 224))
        _, overlay = localizer.overlay(original, patch_scores, alpha=0.5)

        output_name = f"{category}_{Path(temp_path).stem}.png"
        output_path = HEATMAP_DIR / output_name
        cv2.imwrite(str(output_path), overlay)

        return {
            "status": status,
            "category": category,
            "predicted_category": category,
            "category_confidence": round(category_confidence, 2),
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "deviation": round(deviation, 2),
            "patch_count": total_patches,
            "patch_scores": [round(x, 4) for x in patch_scores],
            "top_patch_score": round(top_patch_score, 6),
            "high_score_patch_count": high_score_patch_count,
            "high_score_patch_ratio": round(high_score_patch_ratio, 2),
            "score_ratio": round(score_ratio, 2),
            "severity": severity,
            "anomaly_concentration": concentration,
            "image_auroc": round(float(metrics["image_auroc"]) * 100, 2),
            "average_precision": round(float(metrics["average_precision"]) * 100, 2),
            "pixel_auroc": round(float(pixel_result["pixel_auroc"]) * 100, 2) if pixel_result else None,
            "test_samples": int(metrics.get("num_samples", pixel_result.get("test_samples", 0) if pixel_result else 0)),
            "heatmap": f"/heatmaps/{output_name}"
        }

    finally:
        temp_path.unlink(missing_ok=True)
