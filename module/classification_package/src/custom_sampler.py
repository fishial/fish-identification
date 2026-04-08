import numpy as np
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    """
    Guarantees batches of a strict size (classes_per_batch * samples_per_class).
    Uses Square-Root Sampling (alpha) to combat Long-Tail distribution.
    """
    def __init__(
        self,
        labels,
        classes_per_batch,
        samples_per_class,
        alpha=0.5,
        min_samples_threshold=1,
    ):
        # Important for PyTorch, even though we are overriding the logic
        super().__init__(data_source=None) 
        
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = classes_per_batch * samples_per_class

        unique_classes, counts = np.unique(self.labels, return_counts=True)

        mask = counts >= min_samples_threshold
        self.classes = unique_classes[mask]
        counts = counts[mask]

        if len(self.classes) < self.classes_per_batch:
            raise ValueError(f"Number of classes ({len(self.classes)}) is less than required per batch ({self.classes_per_batch})!")

        # 🔥 Balancing (alpha smoothing)
        self.class_weights = 1.0 / (counts ** alpha)
        self.class_weights /= self.class_weights.sum()

        self.class_to_indices = {
            c: np.where(self.labels == c)[0]
            for c in self.classes
        }

        # Fix the epoch length
        self.num_batches_per_epoch = max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            # Select which classes will be in this batch based on weights
            selected_classes = np.random.choice(
                self.classes,
                size=self.classes_per_batch,
                replace=False,
                p=self.class_weights,
            )

            batch = []
            for c in selected_classes:
                indices = self.class_to_indices[c]
                
                # If a class has fewer samples than required, allow oversampling (replace=True)
                replace = len(indices) < self.samples_per_class
                
                chosen = np.random.choice(
                    indices,
                    size=self.samples_per_class,
                    replace=replace,
                )
                
                # Add indices to the batch
                batch.extend(chosen.tolist())

            # 💣 This is it! Yield the entire list of indices for the BatchSampler
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        # Now len represents the strict count of BATCHES, not individual images
        return self.num_batches_per_epoch