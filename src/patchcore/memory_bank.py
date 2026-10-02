import torch


class MemoryBank:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.features = None

    def fit(self, patch_features):
        if patch_features.ndim == 3:
            patch_features = patch_features.reshape(-1, patch_features.shape[-1])

        if patch_features.shape[0] > self.max_samples:
            indices = torch.linspace(
                0,
                patch_features.shape[0] - 1,
                self.max_samples,
            ).long()

            patch_features = patch_features[indices]

        self.features = patch_features.contiguous()

        return self

    def get_features(self):
        if self.features is None:
            raise RuntimeError("Memory bank has not been fitted.")

        return self.features

    def __len__(self):
        return 0 if self.features is None else self.features.shape[0]
