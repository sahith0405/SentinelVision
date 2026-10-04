import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


class FAISSNearestNeighbor:
    def __init__(self, features):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        features = np.asarray(features, dtype=np.float32)

        if features.ndim != 2:
            raise ValueError("features must have shape [N, D]")

        self.features = features

    def search(self, query_features, k=1):
        if k != 1:
            raise ValueError("The current FAISS worker supports k=1 only.")

        if isinstance(query_features, torch.Tensor):
            query_features = query_features.detach().cpu().numpy()

        query_features = np.asarray(query_features, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            memory_path = temp_dir / "memory.npy"
            query_path = temp_dir / "query.npy"
            output_path = temp_dir / "output.npz"

            np.save(memory_path, self.features)
            np.save(query_path, query_features)

            subprocess.run(
                [
                    sys.executable,
                    "src/patchcore/faiss_worker/search.py",
                    str(memory_path),
                    str(query_path),
                    str(output_path),
                ],
                check=True,
            )

            result = np.load(output_path)

            return result["distances"], result["indices"]
