import sys
import numpy as np
import faiss


def search(memory_path, query_path, output_path):
    memory = np.load(memory_path).astype("float32")
    query = np.load(query_path).astype("float32")

    index = faiss.IndexFlatL2(memory.shape[1])
    index.add(memory)

    distances, indices = index.search(query, 1)

    np.savez(
        output_path,
        distances=distances,
        indices=indices,
    )


if __name__ == "__main__":
    search(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
    )
