import torch
import torch.nn.functional as F


class PatchFeatureExtractor:
    def __init__(self, backbone):
        self.backbone = backbone
        self.backbone.eval()

    @torch.no_grad()
    def extract(self, images):
        features = self.backbone(images)

        target_size = features[0].shape[-2:]

        resized = [
            features[0],
            F.interpolate(
                features[1],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
        ]

        features = torch.cat(resized, dim=1)

        batch_size, channels, height, width = features.shape

        patches = features.permute(0, 2, 3, 1)
        patches = patches.reshape(batch_size, height * width, channels)

        return patches
