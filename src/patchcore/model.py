from pathlib import Path

import numpy as np
import torch

from .coreset import GreedyCoresetSampler
from .faiss_search import FAISSNearestNeighbor
from .features import PatchFeatureExtractor
from .scorer import PatchCoreScorer


class PatchCore:
    def __init__(
        self,
        backbone,
        sampling_ratio=0.1,
        max_memory_size=10000,
    ):
        self.feature_extractor = PatchFeatureExtractor(backbone)

        self.coreset_sampler = GreedyCoresetSampler(
            sampling_ratio=sampling_ratio,
            max_samples=max_memory_size,
        )

        self.nearest_neighbor = None
        self.scorer = None

    @torch.no_grad()
    def build_memory_bank(self, images):
        patch_features = self.feature_extractor.extract(images)

        batch_size, num_patches, feature_dim = patch_features.shape

        features = patch_features.reshape(
            batch_size * num_patches,
            feature_dim,
        )

        memory_features = self.coreset_sampler.sample(features)

        self.nearest_neighbor = FAISSNearestNeighbor(
            memory_features
        )

        self.scorer = PatchCoreScorer(
            self.nearest_neighbor
        )

        return {
            "num_training_images": batch_size,
            "patches_per_image": num_patches,
            "feature_dimension": feature_dim,
            "memory_bank_size": len(memory_features),
        }

    def save_memory_bank(self, path):
        if self.nearest_neighbor is None:
            raise RuntimeError(
                "Memory bank has not been built."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.save(
            path,
            self.nearest_neighbor.features,
        )

    def load_memory_bank(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Memory bank not found: {path}"
            )

        features = np.load(path).astype(np.float32)

        self.nearest_neighbor = FAISSNearestNeighbor(
            features
        )

        self.scorer = PatchCoreScorer(
            self.nearest_neighbor
        )

        return {
            "memory_bank_size": len(features),
            "feature_dimension": features.shape[1],
        }

    @torch.no_grad()
    def predict(self, images):
        if self.nearest_neighbor is None or self.scorer is None:
            raise RuntimeError(
                "PatchCore memory bank has not been built."
            )

        patch_features = self.feature_extractor.extract(images)

        batch_scores = []
        batch_patch_scores = []

        for image_features in patch_features:
            patch_scores = self.scorer.score_patches(
                image_features
            )

            image_score = self.scorer.score_image(
                image_features
            )

            batch_patch_scores.append(patch_scores)
            batch_scores.append(image_score)

        return {
            "image_scores": batch_scores,
            "patch_scores": batch_patch_scores,
        }
