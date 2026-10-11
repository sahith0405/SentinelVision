import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.backbone import WideResNet50Backbone
from src.evaluation import (
    classification_metrics,
    image_average_precision,
    image_auroc,
)
from src.patchcore import PatchCore
from src.utils import MVTecDataset, get_transform


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SentinelVision on MVTec AD."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/mvtec",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Category: {args.category}")

    transform = get_transform()

    dataset = MVTecDataset(
        root=args.data,
        category=args.category,
        split="test",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    backbone = WideResNet50Backbone().to(device)

    model = PatchCore(backbone)

    memory_path = (
        f"models/{args.category}_memory_bank.npy"
    )

    model.load_memory_bank(memory_path)

    labels = []
    scores = []

    for batch in loader:
        images = batch["image"].to(device)

        result = model.predict(images)

        for label, score in zip(
            batch["label"],
            result["image_scores"],
        ):
            labels.append(
                int(label)
            )

            scores.append(score)

    auroc = image_auroc(labels, scores)
    average_precision = image_average_precision(
        labels,
        scores,
    )

    metrics = {
        "category": args.category,
        "image_auroc": auroc,
        "average_precision": average_precision,
        "num_samples": len(labels),
    }

    if args.threshold is not None:
        predictions = [
            1 if score >= args.threshold else 0
            for score in scores
        ]

        metrics.update(
            classification_metrics(
                labels,
                predictions,
            )
        )

    output_dir = Path("results/metrics")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        f"{args.category}_metrics.json"
    )

    with open(output_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\nEvaluation Results")
    print("------------------")
    print(f"Image AUROC: {auroc:.6f}")
    print(f"Average Precision: {average_precision:.6f}")
    print(f"Samples: {len(labels)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
