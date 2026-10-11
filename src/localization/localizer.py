import numpy as np

from .heatmap import AnomalyHeatmap


class PatchCoreLocalizer:
    def __init__(self, output_size=(224, 224)):
        self.heatmap_generator = AnomalyHeatmap(output_size)

    def generate(self, patch_scores):
        patch_scores = np.asarray(patch_scores, dtype=np.float32)

        num_patches = patch_scores.size
        grid_size = int(np.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"Patch count {num_patches} does not form a square grid."
            )

        heatmap = self.heatmap_generator.generate(
            patch_scores,
            grid_size,
            grid_size,
        )

        return heatmap

    def overlay(self, image, patch_scores, alpha=0.5):
        heatmap = self.generate(patch_scores)

        overlay = self.heatmap_generator.overlay(
            image,
            heatmap,
            alpha=alpha,
        )

        return heatmap, overlay
