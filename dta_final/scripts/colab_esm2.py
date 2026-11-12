"""Compute ESM-2 protein embeddings for the DAVIS and KIBA targets.

RUN THIS ON COLAB (or Kaggle) WITH A GPU, not on the project machine.
Runtime -> Change runtime type -> T4 GPU, then paste this whole file into one cell
and run it. It takes about 3 to 5 minutes for the 650M model and under a minute for
the 35M one.

It downloads the protein sequences itself, so nothing needs uploading first. At the
end it writes four .npy files plus an ID-order JSON and triggers a browser download.
Put those files in the project's data/processed/ directory.

Why chunking rather than truncating: 109 of the 442 DAVIS targets and 42 of the 229
KIBA targets are longer than ESM-2's 1022-residue window, up to 4128 residues. DeepDTA
simply cuts proteins at 1000 characters, which discards the C-terminal kinase domain of
a quarter of the panel. We instead embed overlapping windows and average them weighted
by window length, so every residue contributes exactly once in expectation.
"""

import json

import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:                                    # Colab sometimes needs this
    import subprocess
    subprocess.run(["pip", "-q", "install", "transformers"], check=True)
    from transformers import AutoModel, AutoTokenizer

# Two sizes so Task 3 can ablate the protein encoder's capacity.
MODELS = {
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",      # 480-dim
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",    # 1280-dim
}
BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
WINDOW = 1022          # ESM-2's usable residue context, excluding BOS and EOS
STRIDE = 511           # 50% overlap, so every residue is seen by two windows
BATCH = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


def fetch_targets(dataset: str) -> dict[str, str]:
    import urllib.request

    url = f"{BASE}/{dataset}/proteins.txt"
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def windows(sequence: str) -> list[str]:
    """Split a sequence into overlapping windows of at most WINDOW residues."""
    if len(sequence) <= WINDOW:
        return [sequence]
    out, start = [], 0
    while start < len(sequence):
        out.append(sequence[start:start + WINDOW])
        if start + WINDOW >= len(sequence):
            break
        start += STRIDE
    return out


@torch.no_grad()
def mean_pool(model, tokenizer, chunks: list[str]) -> np.ndarray:
    """Mean-pool the final hidden states over real residues, excluding special tokens."""
    vectors = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=WINDOW + 2)
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state          # B x L x D
        mask = encoded["attention_mask"].clone()
        mask[:, 0] = 0                                       # drop BOS
        lengths = mask.sum(dim=1, keepdim=True)
        # Drop EOS: it is the last attended position of each row.
        for row in range(mask.shape[0]):
            last = int(encoded["attention_mask"][row].sum()) - 1
            mask[row, last] = 0
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def embed_dataset(dataset: str, model, tokenizer) -> tuple[np.ndarray, list[str]]:
    targets = fetch_targets(dataset)
    ids = list(targets)
    out = []
    for n, target_id in enumerate(ids, 1):
        chunks = windows(targets[target_id])
        pooled = mean_pool(model, tokenizer, chunks)
        # Weight each window by how many residues it covers.
        weights = np.array([len(c) for c in chunks], dtype=np.float32)
        out.append((pooled * weights[:, None]).sum(axis=0) / weights.sum())
        if n % 50 == 0 or n == len(ids):
            print(f"  {dataset}: {n}/{len(ids)}", flush=True)
    return np.vstack(out).astype(np.float32), ids


def main() -> None:
    order: dict[str, list[str]] = {}
    written: list[str] = []
    for tag, checkpoint in MODELS.items():
        print(f"\nloading {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint).to(DEVICE).eval()
        if DEVICE == "cuda":
            model = model.half()
        for dataset in ("davis", "kiba"):
            matrix, ids = embed_dataset(dataset, model, tokenizer)
            name = f"{tag}_{dataset}.npy"
            np.save(name, matrix)
            written.append(name)
            order[dataset] = ids
            print(f"  saved {name} with shape {matrix.shape}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open("esm2_target_order.json", "w") as fh:
        json.dump(order, fh)
    written.append("esm2_target_order.json")

    print("\nRow i of every .npy is the target at index i of the matching list in "
          "esm2_target_order.json, which is the same order as proteins.txt.")
    try:                                               # Colab only
        from google.colab import files

        for name in written:
            files.download(name)
    except ImportError:
        print("Not on Colab. Collect these files manually:", ", ".join(written))


if __name__ == "__main__":
    main()
