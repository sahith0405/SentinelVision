import torch


class GreedyCoresetSampler:
    """
    Fast approximate coreset sampler.

    The sampler first reduces the candidate pool using a
    deterministic evenly spaced selection, then performs
    farthest-point selection on that smaller pool.
    """

    def __init__(
        self,
        sampling_ratio=0.1,
        max_samples=None,
        candidate_multiplier=3,
    ):
        self.sampling_ratio = sampling_ratio
        self.max_samples = max_samples
        self.candidate_multiplier = candidate_multiplier

    @torch.no_grad()
    def sample(self, features):
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [N, D]"
            )

        n_samples = features.shape[0]

        if n_samples == 0:
            raise ValueError(
                "features cannot be empty"
            )

        target = max(
            1,
            int(n_samples * self.sampling_ratio),
        )

        if self.max_samples is not None:
            target = min(
                target,
                self.max_samples,
            )

        if target >= n_samples:
            return features

        # For very large feature sets, first create a
        # manageable candidate pool.
        candidate_count = min(
            n_samples,
            max(
                target * self.candidate_multiplier,
                5000,
            ),
        )

        if candidate_count < n_samples:
            indices = torch.linspace(
                0,
                n_samples - 1,
                candidate_count,
                device=features.device,
            ).long()

            candidates = features[indices]
        else:
            candidates = features

        candidate_count = candidates.shape[0]

        # If the candidate pool is already small enough,
        # return it directly.
        if target >= candidate_count:
            return candidates

        # Normalize only for distance calculation.
        normalized = torch.nn.functional.normalize(
            candidates,
            p=2,
            dim=1,
        )

        selected = torch.zeros(
            target,
            dtype=torch.long,
            device=features.device,
        )

        # Start with the feature farthest from the mean.
        center = normalized.mean(
            dim=0,
            keepdim=True,
        )

        distances = 1 - torch.mm(
            normalized,
            center.T,
        ).squeeze(1)

        selected[0] = torch.argmax(
            distances
        )

        min_distances = torch.full(
            (candidate_count,),
            float("inf"),
            device=features.device,
        )

        for i in range(1, target):
            selected_feature = normalized[
                selected[i - 1]
            ].unsqueeze(0)

            distances = 1 - torch.mm(
                normalized,
                selected_feature.T,
            ).squeeze(1)

            min_distances = torch.minimum(
                min_distances,
                distances,
            )

            selected[i] = torch.argmax(
                min_distances
            )

        return candidates[selected]
