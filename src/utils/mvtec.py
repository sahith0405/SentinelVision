from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(
        self,
        root,
        category,
        split="train",
        transform=None,
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform

        self.category_root = (
            self.root / category
        )

        if split == "train":
            self.image_root = (
                self.category_root /
                "train" /
                "good"
            )
        elif split == "test":
            self.image_root = (
                self.category_root /
                "test"
            )
        else:
            raise ValueError(
                "split must be 'train' or 'test'"
            )

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"MVTec directory not found: "
                f"{self.image_root}"
            )

        self.samples = []

        if split == "train":
            for image_path in sorted(
                self.image_root.glob("*.png")
            ):
                self.samples.append(
                    {
                        "path": image_path,
                        "label": 0,
                    }
                )

        else:
            for defect_type in sorted(
                self.image_root.iterdir()
            ):
                if not defect_type.is_dir():
                    continue

                label = (
                    0
                    if defect_type.name == "good"
                    else 1
                )

                for image_path in sorted(
                    defect_type.glob("*.png")
                ):
                    self.samples.append(
                        {
                            "path": image_path,
                            "label": label,
                        }
                    )

        if not self.samples:
            raise RuntimeError(
                f"No images found in "
                f"{self.image_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = sample["path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "path": str(image_path),
            "label": sample["label"],
        }
