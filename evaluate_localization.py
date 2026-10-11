import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.backbone import WideResNet50Backbone
from src.patchcore import PatchCore
from src.utils import MVTecDataset, get_transform


CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def load_ground_truth_mask(data_root, category, image_path):
    image_path = Path(image_path)

    relative = image_path.relative_to(
        Path(data_root) / category / "test"
    )

    defect_type = relative.parts[0]

    if defect_type == "good":
        return np.zeros((224, 224), dtype=np.uint8)

    mask_path = (
        Path(data_root)
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found: {mask_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    mask = cv2.resize(
        mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST,
    )

    return (mask > 0).astype(np.uint8)


def patch_scores_to_map(patch_scores):
    patch_scores = np.asarray(
        patch_scores,
        dtype=np.float32,
    )

    grid_size = int(np.sqrt(patch_scores.size))

    if grid_size * grid_size != patch_scores.size:
        raise ValueError(
            f"Expected square patch grid, got "
            f"{patch_scores.size} patches."
        )

    score_map = patch_scores.reshape(
        grid_size,
        grid_size,
    )

    score_map = cv2.resize(
        score_map,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    return score_map


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = []

    for category in CATEGORIES:
        print("\n" + "=" * 70)
        print(f"CATEGORY: {category}")
        print("=" * 70)

        memory_path = (
            f"models/{category}_memory_bank.npy"
        )

        dataset = MVTecDataset(
            root="data",
            category=category,
            split="test",
            transform=get_transform(),
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        backbone = WideResNet50Backbone().to(device)
        model = PatchCore(backbone)
        model.load_memory_bank(memory_path)

        all_masks = []
        all_maps = []

        for index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)

            result = model.predict(image)

            patch_scores = result["patch_scores"][0]

            anomaly_map = patch_scores_to_map(
                patch_scores
            )

            image_path = batch["path"][0]

            mask = load_ground_truth_mask(
                "data",
                category,
                image_path,
            )

            all_maps.append(
                anomaly_map.flatten()
            )

            all_masks.append(
                mask.flatten()
            )

            if index % 25 == 0:
                print(
                    f"Processed {index}/{len(dataset)}"
                )

        masks = np.concatenate(all_masks)
        maps = np.concatenate(all_maps)

        if np.unique(masks).size < 2:
            pixel_auroc = None
            print("Pixel AUROC: unavailable")
        else:
            pixel_auroc = float(
                roc_auc_score(masks, maps)
            )
            print(
                f"Pixel AUROC: {pixel_auroc:.6f}"
            )

        results.append({
            "category": category,
            "pixel_auroc": pixel_auroc,
            "test_samples": len(dataset),
        })

    output_path = Path(
        "results/metrics/"
        "all_categories_pixel_localization.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("PIXEL LOCALIZATION SUMMARY")
    print("=" * 70)

    for result in results:
        score = result["pixel_auroc"]

        if score is None:
            score_text = "N/A"
        else:
            score_text = f"{score:.6f}"

        print(
            f"{result['category']:<15}"
            f"{score_text}"
        )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
