import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from src.backbone import WideResNet50Backbone
from src.localization import PatchCoreLocalizer
from src.patchcore import PatchCore
from src.utils import get_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument(
        "--memory-bank",
        default=None,
    )
    parser.add_argument(
        "--threshold",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory_path = (
        args.memory_bank
        if args.memory_bank
        else f"models/{args.category}_memory_bank.npy"
    )

    threshold_path = (
        Path(args.threshold)
        if args.threshold
        else Path(f"results/metrics/{args.category}_threshold.json")
    )

    output_path = (
        args.output
        if args.output
        else f"results/heatmaps/{args.category}_inference.png"
    )

    with open(threshold_path, "r") as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])

    image = Image.open(args.image).convert("RGB")
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    backbone = WideResNet50Backbone().to(device)

    model = PatchCore(backbone)
    memory_info = model.load_memory_bank(memory_path)

    result = model.predict(tensor)

    score = float(result["image_scores"][0])
    patch_scores = result["patch_scores"][0]

    status = "ANOMALY" if score >= threshold else "NORMAL"

    localizer = PatchCoreLocalizer(output_size=(224, 224))

    original = cv2.imread(args.image)
    heatmap, overlay = localizer.overlay(
        original,
        patch_scores,
        alpha=0.5,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), overlay)

    print(f"Device: {device}")
    print(f"Memory bank loaded: {memory_info}")
    print(f"Anomaly score: {score:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Status: {status}")
    print(f"Patch count: {len(patch_scores)}")
    print(f"Result saved to: {output_file}")


if __name__ == "__main__":
    main()
