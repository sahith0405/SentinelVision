import cv2
import numpy as np


class AnomalyHeatmap:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def generate(self, patch_scores, feature_height, feature_width):
        patch_scores = np.asarray(patch_scores, dtype=np.float32)

        expected_size = feature_height * feature_width

        if patch_scores.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} patch scores, "
                f"but received {patch_scores.size}"
            )

        heatmap = patch_scores.reshape(feature_height, feature_width)

        heatmap = cv2.resize(
            heatmap,
            self.output_size,
            interpolation=cv2.INTER_LINEAR,
        )

        minimum = heatmap.min()
        maximum = heatmap.max()

        if maximum > minimum:
            heatmap = (heatmap - minimum) / (maximum - minimum)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)

    def overlay(self, image, heatmap, alpha=0.5):
        if image is None:
            raise ValueError("image cannot be None")

        heatmap_uint8 = np.uint8(255 * heatmap)

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        if image.shape[:2] != heatmap.shape[:2]:
            image = cv2.resize(
                image,
                (heatmap.shape[1], heatmap.shape[0]),
            )

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            colored_heatmap,
            alpha,
            0,
        )

        return overlay
