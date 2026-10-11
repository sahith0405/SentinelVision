import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.backbone import WideResNet50Backbone
from src.patchcore import PatchCore
from src.utils import MVTecDataset, get_transform


def main():
    parser = argparse.ArgumentParser(
        description="Build SentinelVision PatchCore memory bank."
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
        default=8,
    )

    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-memory-size",
        type=int,
        default=10000,
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
        split="train",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    backbone = WideResNet50Backbone().to(device)

    model = PatchCore(
        backbone=backbone,
        sampling_ratio=args.sampling_ratio,
        max_memory_size=args.max_memory_size,
    )

    feature_batches = []

    print(f"Training images: {len(dataset)}")
    print("Extracting normal patch features...")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)

            features = model.feature_extractor.extract(
                images
            )

            features = features.reshape(
                -1,
                features.shape[-1],
            )

            feature_batches.append(
                features.cpu()
            )

            print(
                f"Batch {batch_index}/{len(loader)} "
                f"processed"
            )

    all_features = torch.cat(
        feature_batches,
        dim=0,
    )

    print(
        f"Total patch features: "
        f"{all_features.shape[0]}"
    )

    memory_features = model.coreset_sampler.sample(
        all_features
    )

    from src.patchcore import FAISSNearestNeighbor
    from src.patchcore import PatchCoreScorer

    model.nearest_neighbor = FAISSNearestNeighbor(
        memory_features
    )

    model.scorer = PatchCoreScorer(
        model.nearest_neighbor
    )

    output_dir = Path("models")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_path = (
        output_dir /
        f"{args.category}_memory_bank.npy"
    )

    model.save_memory_bank(memory_path)

    print("\nPatchCore training complete.")
    print(
        f"Memory bank size: "
        f"{len(memory_features)}"
    )
    print(
        f"Feature dimension: "
        f"{memory_features.shape[1]}"
    )
    print(
        f"Saved to: {memory_path}"
    )


if __name__ == "__main__":
    main()
