import numpy as np
import torch


class PatchCoreScorer:
    def __init__(self, nearest_neighbor):
        self.nearest_neighbor = nearest_neighbor

    def score_patches(self, patch_features):
        distances, _ = self.nearest_neighbor.search(patch_features, k=1)

        distances = np.asarray(distances, dtype=np.float32)

        if distances.ndim == 2:
            distances = distances[:, 0]

        return distances

    def score_image(self, patch_features):
        patch_scores = self.score_patches(patch_features)

        if patch_scores.size == 0:
            raise ValueError("patch_features cannot be empty")

        top_k = max(1, int(np.ceil(patch_scores.size * 0.10)))
        top_scores = np.partition(patch_scores, -top_k)[-top_k:]

        return float(np.mean(top_scores))
