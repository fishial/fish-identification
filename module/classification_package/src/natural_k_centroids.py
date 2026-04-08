import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import h5py
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append('/home/andrew/Andrew/Fishial2402/fish-identification')

from module.classification_package.src.model_v2 import StableEmbeddingModelViT
from module.classification_package.src.data.module.v2 import ImageEmbeddingDataModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spherical K-Means
# Euclidean K-Means is wrong for L2-normalized embeddings — it minimises
# Euclidean distance, which is NOT equivalent to cosine distance on the sphere.
# Spherical K-Means minimises cosine distance directly by always normalising
# cluster centres back onto the unit sphere after each update step.
# ---------------------------------------------------------------------------

def spherical_kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Spherical K-Means for L2-normalized embeddings.

    Args:
        X:            (n, d) array of L2-normalized vectors.
        k:            Number of clusters.
        n_init:       Number of independent restarts; best result is kept.
        max_iter:     Maximum EM iterations per restart.
        random_state: Seed for reproducibility.
        tol:          Convergence tolerance (L2 change in centres).

    Returns:
        (k, d) array of L2-normalised cluster centres.
    """
    assert k >= 1, "k must be >= 1"
    assert X.ndim == 2, "X must be 2-D"

    rng = np.random.default_rng(random_state)
    best_centers: Optional[np.ndarray] = None
    best_inertia = -np.inf  # we maximise total cosine similarity

    for _ in range(n_init):
        # K-Means++ style initialisation on the sphere
        idx = [rng.integers(len(X))]
        for _ in range(1, k):
            # cosine similarity to nearest existing centre
            sims = X @ X[idx].T             # (n, len(idx))
            nearest_sim = sims.max(axis=1)  # (n,)
            # sample proportional to (1 - sim) so distant points are preferred
            probs = np.clip(1.0 - nearest_sim, 0, None)
            total = probs.sum()
            if total == 0:
                # All points are identical — just duplicate the centre
                idx.append(idx[-1])
            else:
                probs /= total
                idx.append(rng.choice(len(X), p=probs))

        centers = X[idx].copy()             # (k, d), already normalised

        for _ in range(max_iter):
            # Assignment: cosine similarity = dot product on unit sphere
            sims = X @ centers.T            # (n, k)
            labels = sims.argmax(axis=1)    # (n,)

            # Update: mean of assigned points → normalise back to sphere
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    mean_vec = X[mask].mean(axis=0)
                    norm = np.linalg.norm(mean_vec)
                    new_centers[j] = mean_vec / norm if norm > 0 else centers[j]
                else:
                    # Empty cluster — keep old centre
                    new_centers[j] = centers[j]

            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift < tol:
                break

        # Inertia = total cosine similarity (higher = better)
        inertia = float((X * centers[labels]).sum())
        if inertia > best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()

    return best_centers  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Centroid generation
# ---------------------------------------------------------------------------

def generate_natural_centroids(
    h5_filepath: Path,
    k_centers: int = 3,
    keep_ratio: float = 0.90,
    use_adaptive_keep: bool = True,
    min_samples_for_cleaning: int = 5,
) -> Dict[str, Any]:
    """
    Build a gallery of (up to) k_centers L2-normalised centroids per class.

    Pipeline per class:
      1. Load raw embeddings from HDF5.
      2. L2-normalise ONCE — all subsequent operations live on the unit sphere.
      3. Outlier removal (skipped for tiny classes):
           a. Compute a robust centre (geometric median approximation via
              L2-normalised mean of the normalised vectors).
           b. Measure cosine similarity of every sample to that centre.
           c. Drop the bottom (1 - effective_keep_ratio)% of samples.
              If use_adaptive_keep=True the ratio is tightened for high-
              variance classes (many dissimilar images of the same species).
      4. Spherical K-Means with n_init restarts on the cleaned, normalised
         embeddings.  Euclidean K-Means is NOT used because its Voronoi
         tesselation on the sphere is not equivalent to cosine-distance
         Voronoi.
      5. L2-normalise cluster centres (projects them back onto the sphere).
      6. If a class has fewer samples than k_centers, the actual number of
         centres is capped at n_samples — no duplicate padding.

    Args:
        h5_filepath:              Path to the temporary HDF5 file.
        k_centers:                Desired number of centroids per class.
        keep_ratio:               Base fraction of samples to retain (0–1).
        use_adaptive_keep:        Tighten keep_ratio for high-variance classes.
        min_samples_for_cleaning: Minimum class size before cleaning is applied.

    Returns:
        dict with keys:
          'embeddings': np.ndarray (N_total_centroids, d)
          'labels':     np.ndarray (N_total_centroids,)  — class label per row
          'actual_k':   dict {label: actual_k}           — real k per class
    """
    logger.info("=" * 60)
    logger.info(f"STEP 2: Generating up to {k_centers} CLEANED CENTROIDS per class")
    logger.info(f"Base keep_ratio={keep_ratio:.2f}  adaptive={use_adaptive_keep}")
    logger.info("=" * 60)

    final_centroids: List[np.ndarray] = []
    final_labels:    List[np.ndarray] = []
    actual_k_map:    Dict[int, int]   = {}

    with h5py.File(h5_filepath, 'r') as hf:
        labels_all   = hf['labels'][:]
        unique_labels = np.unique(labels_all)

        for label in tqdm(unique_labels, desc='Calculating Centroids'):
            class_mask = labels_all == label
            raw_embs   = hf['embeddings'][class_mask]     # (n, d)
            n_samples  = len(raw_embs)

            # ----------------------------------------------------------------
            # Step 2: L2-normalise ONCE — work on the sphere from here on.
            # FIX: previously raw_embs and norm_embs were used interchangeably;
            #      the outlier mask was applied to raw_embs while similarities
            #      were computed on norm_embs, creating an inconsistency.
            # ----------------------------------------------------------------
            norm_embs = normalize(raw_embs, axis=1, norm='l2')  # (n, d)

            # ----------------------------------------------------------------
            # Step 3: Outlier removal
            # ----------------------------------------------------------------
            if n_samples > min_samples_for_cleaning:
                # Robust centre: L2-normalised mean of unit vectors
                # (geometric median approximation; faster and good enough)
                rough_center = norm_embs.mean(axis=0, keepdims=True)
                rough_center = normalize(rough_center, axis=1, norm='l2')   # (1, d)

                # Cosine similarity = dot product on unit sphere
                similarities = (norm_embs @ rough_center.T).flatten()       # (n,)

                # Adaptive keep_ratio: tighten for high-variance classes
                if use_adaptive_keep:
                    sim_std = float(similarities.std())
                    # Higher variance → more aggressive cleaning,
                    # but never below 0.70 or above keep_ratio.
                    effective_keep = float(np.clip(
                        keep_ratio - sim_std * 0.5, 0.70, keep_ratio
                    ))
                else:
                    effective_keep = keep_ratio

                threshold  = np.percentile(similarities, (1.0 - effective_keep) * 100)
                clean_mask = similarities >= threshold

                # FIX: filter norm_embs (NOT raw_embs)
                norm_embs = norm_embs[clean_mask]
                n_samples = len(norm_embs)

            # ----------------------------------------------------------------
            # Step 4: Spherical K-Means (or simple mean for k=1)
            # FIX: Euclidean K-Means replaced by Spherical K-Means.
            #      Also, actual_k is now capped at n_samples without padding.
            # ----------------------------------------------------------------
            actual_k = min(k_centers, n_samples)

            if actual_k > 1:
                centers = spherical_kmeans(
                    norm_embs,
                    k=actual_k,
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                )
            else:
                # Single centre: L2-normalised mean (Fréchet mean on sphere)
                mean_vec = norm_embs.mean(axis=0, keepdims=True)
                centers  = normalize(mean_vec, axis=1, norm='l2')

            # ----------------------------------------------------------------
            # Step 5: Final L2-normalisation (centres from K-Means are already
            #         normalised inside spherical_kmeans, but an extra pass is
            #         harmless and makes the guarantee explicit).
            # ----------------------------------------------------------------
            centers = normalize(centers, axis=1, norm='l2')   # (actual_k, d)

            # ----------------------------------------------------------------
            # FIX: No duplicate padding.
            # Previously: while len(centers) < k_centers: centers = vstack(...)
            # This created phantom duplicates that pollute nearest-neighbour
            # search — the same vector appears under multiple logical IDs.
            # Instead we store actual_k and let the caller handle the variable
            # number of centres per class.
            # ----------------------------------------------------------------
            actual_k_map[int(label)] = actual_k

            final_centroids.append(centers)
            final_labels.append(np.full(actual_k, label, dtype=labels_all.dtype))

    return {
        'embeddings': np.vstack(final_centroids),      # (total_centers, d)
        'labels':     np.concatenate(final_labels),    # (total_centers,)
        'actual_k':   actual_k_map,                    # {label: actual_k}
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_to_hdf5(
    model:          torch.nn.Module,
    dataloader,
    device:         str,
    output_filepath: Path,
    embedding_dim:  int = 768,
) -> None:
    """
    Run the model in eval mode over `dataloader` and stream embeddings to HDF5.
    Uses chunked/resizable datasets to avoid loading everything into RAM.
    """
    model.eval()
    with h5py.File(output_filepath, 'w') as hf:
        emb_ds = hf.create_dataset(
            'embeddings',
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype='float32',
            chunks=(256, embedding_dim),
        )
        lbl_ds = hf.create_dataset(
            'labels',
            shape=(0,),
            maxshape=(None,),
            dtype='int64',
            chunks=(256,),
        )

        with torch.no_grad():
            for images, targets, _, _ in tqdm(dataloader, desc='Extracting Embeddings'):
                emb_norm, _, _, _ = model(images.to(device))
                batch_embs = emb_norm.cpu().numpy()
                batch_lbls = targets.cpu().numpy()

                curr_len   = emb_ds.shape[0]
                batch_size = len(batch_embs)

                emb_ds.resize(curr_len + batch_size, axis=0)
                lbl_ds.resize(curr_len + batch_size, axis=0)

                emb_ds[curr_len:] = batch_embs
                lbl_ds[curr_len:] = batch_lbls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a centroid gallery from a trained embedding model.'
    )
    parser.add_argument('--checkpoint',          type=str, required=True,
                        help='Path to the model checkpoint (.ckpt)')
    parser.add_argument('--output_dir',          type=str, required=True,
                        help='Directory to save the gallery .pt file')
    parser.add_argument('--label_path',          type=str, required=True,
                        help='Path to the labels file')
    parser.add_argument('--class_mapping_path',  type=str, required=True,
                        help='Path to the class mapping JSON')
    parser.add_argument('--k_centers',           type=int, default=3,
                        help='Number of centroids per class (default: 3)')
    parser.add_argument('--keep_ratio',          type=float, default=0.90,
                        help='Fraction of samples to keep after outlier removal (default: 0.90)')
    parser.add_argument('--no_adaptive_keep',    action='store_true',
                        help='Disable adaptive keep_ratio (use fixed keep_ratio for all classes)')
    parser.add_argument('--dataset_name',        type=str,
                        default='segmentation_dataset_v0.11_with_meta')
    parser.add_argument('--batch_size',          type=int, default=128)
    parser.add_argument('--embedding_dim',       type=int, default=768,
                        help='Embedding dimensionality (default: 768)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Starting Natural Centroids Pipeline  K={args.k_centers}  "
                f"keep_ratio={args.keep_ratio}  "
                f"adaptive={'off' if args.no_adaptive_keep else 'on'}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = StableEmbeddingModelViT.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).eval().to(device)

    # ------------------------------------------------------------------
    # Build datamodule
    # ------------------------------------------------------------------
    db_module = ImageEmbeddingDataModule(
        dataset_name        = args.dataset_name,
        labels_path         = args.label_path,
        class_mapping_path  = args.class_mapping_path,
        image_size          = (154, 434),
        val_tags            = None,
        bg_removal_prob     = 0.0,
        cache_dir           = '/home/andrew/Andrew/Fishial2402/dataset/fo_cache',
        use_cache           = True,
        train_tags          = [],
        instance_data       = True,
        bbox_padding_limit  = 0.0,
        resize_strategy     = 'pad',
        alignment_method    = 'horizontal',
    )
    db_module.setup()

    temp_h5_path = output_dir / 'temp_embeddings.h5'

    try:
        # ------------------------------------------------------------------
        # STEP 1: Extract embeddings → HDF5
        # ------------------------------------------------------------------
        logger.info("STEP 1: Extracting embeddings to HDF5...")
        extract_to_hdf5(
            model,
            db_module.val_dataloader(),
            device,
            temp_h5_path,
            embedding_dim=args.embedding_dim,
        )

        # ------------------------------------------------------------------
        # STEP 2: Build gallery (outlier cleaning + Spherical K-Means)
        # ------------------------------------------------------------------
        gallery = generate_natural_centroids(
            temp_h5_path,
            k_centers        = args.k_centers,
            keep_ratio       = args.keep_ratio,
            use_adaptive_keep= not args.no_adaptive_keep,
        )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        model_name = Path(args.checkpoint).stem
        save_path  = output_dir / f'natural_gallery_K{args.k_centers}_{model_name}.pt'

        torch.save(
            {
                'centroids':   torch.from_numpy(gallery['embeddings']),
                'labels':      gallery['labels'],
                'actual_k':    gallery['actual_k'],     # real k per class (no padding)
                'labels_keys': db_module.labels_keys,   # ID → species name mapping
                'k_centers':   args.k_centers,
            },
            save_path,
        )

        logger.info(f"✅ Natural Gallery saved to: {save_path}")
        logger.info(f"   Total centroid vectors : {gallery['embeddings'].shape[0]}")
        logger.info(f"   Embedding dimension    : {gallery['embeddings'].shape[1]}")

        # Log classes where actual_k < k_centers (small classes)
        sparse = {k: v for k, v in gallery['actual_k'].items() if v < args.k_centers}
        if sparse:
            logger.info(f"   Classes with actual_k < {args.k_centers} "
                        f"(insufficient samples): {len(sparse)}")

    finally:
        if temp_h5_path.exists():
            temp_h5_path.unlink()
            logger.info("Temporary HDF5 file removed.")


if __name__ == '__main__':
    main()