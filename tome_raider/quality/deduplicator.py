"""Deduplication system for dataset samples."""

import hashlib
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from loguru import logger

from ..sources.base import Sample


class Deduplicator:
    """Remove and detect duplicate samples."""

    def __init__(self, config: Dict = None):
        """
        Initialize deduplicator.

        Args:
            config: Configuration dictionary
        """
        config = config or {}
        self.near_duplicate_threshold = config.get("near_duplicate_threshold", 0.85)

    def remove_exact_duplicates(
        self,
        dataset: List[Sample],
        field: str = "instruction"
    ) -> Tuple[List[Sample], List[int]]:
        """
        Remove exact duplicates based on field.

        Args:
            dataset: Dataset to deduplicate
            field: Field to check for duplicates

        Returns:
            Tuple of (deduplicated dataset, indices of duplicates removed)
        """
        logger.info(f"Removing exact duplicates based on '{field}' field")

        seen = set()
        unique = []
        removed_indices = []

        for idx, sample in enumerate(dataset):
            # Get field value
            if field == "instruction":
                value = sample.instruction
            elif field == "response":
                value = sample.response
            else:
                value = str(sample.to_dict().get(field, ""))

            # Hash the value
            value_hash = hashlib.sha256(value.encode('utf-8')).hexdigest()

            if value_hash not in seen:
                seen.add(value_hash)
                unique.append(sample)
            else:
                removed_indices.append(idx)

        logger.info(
            f"Removed {len(removed_indices)} exact duplicates. "
            f"Remaining: {len(unique)}/{len(dataset)}"
        )

        return unique, removed_indices

    def detect_near_duplicates(
        self,
        dataset: List[Sample],
        threshold: float = None,
        field: str = "instruction"
    ) -> List[Tuple[int, int, float]]:
        """
        Detect near-duplicate samples using fuzzy matching.

        Args:
            dataset: Dataset to check
            threshold: Similarity threshold (0-1)
            field: Field to check for duplicates

        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        if threshold is None:
            threshold = self.near_duplicate_threshold

        logger.info(
            f"Detecting near-duplicates with threshold {threshold} "
            f"on '{field}' field"
        )

        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.error("rapidfuzz not installed. Install with: pip install rapidfuzz")
            return []

        duplicates = []

        # Extract field values
        values = []
        for sample in dataset:
            if field == "instruction":
                values.append(sample.instruction)
            elif field == "response":
                values.append(sample.response)
            else:
                values.append(str(sample.to_dict().get(field, "")))

        # Compare all pairs (optimized with early stopping)
        total_comparisons = len(values) * (len(values) - 1) // 2
        comparisons_done = 0

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                # Calculate similarity
                similarity = fuzz.ratio(values[i], values[j]) / 100.0

                if similarity >= threshold:
                    duplicates.append((i, j, similarity))

                comparisons_done += 1

                # Log progress for large datasets
                if comparisons_done % 10000 == 0:
                    logger.debug(
                        f"Progress: {comparisons_done}/{total_comparisons} comparisons"
                    )

        logger.info(f"Found {len(duplicates)} near-duplicate pairs")

        return duplicates

    def flag_near_duplicates(
        self,
        dataset: List[Sample],
        threshold: float = None,
        field: str = "instruction"
    ) -> List[Sample]:
        """
        Flag near-duplicates in metadata without removing them.

        Args:
            dataset: Dataset to process
            threshold: Similarity threshold
            field: Field to check

        Returns:
            Dataset with flagged samples
        """
        duplicates = self.detect_near_duplicates(dataset, threshold, field)

        # Build set of indices that are duplicates
        duplicate_indices = set()
        for i, j, _ in duplicates:
            duplicate_indices.add(i)
            duplicate_indices.add(j)

        # Flag samples
        for idx in duplicate_indices:
            if idx < len(dataset):
                dataset[idx].metadata.custom["near_duplicate"] = True

        logger.info(f"Flagged {len(duplicate_indices)} samples as near-duplicates")

        return dataset

    def remove_near_duplicates(
        self,
        dataset: List[Sample],
        threshold: float = None,
        field: str = "instruction",
        keep: str = "first"
    ) -> Tuple[List[Sample], List[int]]:
        """
        Remove near-duplicate samples.

        Args:
            dataset: Dataset to deduplicate
            threshold: Similarity threshold
            field: Field to check
            keep: Which duplicate to keep ("first", "last", "best")

        Returns:
            Tuple of (deduplicated dataset, indices removed)
        """
        duplicates = self.detect_near_duplicates(dataset, threshold, field)

        if not duplicates:
            return dataset, []

        # Build groups of duplicates
        groups = self._build_duplicate_groups(duplicates, len(dataset))

        # Decide which to keep from each group
        indices_to_remove = set()

        for group in groups:
            if len(group) <= 1:
                continue

            if keep == "first":
                # Keep first, remove rest
                indices_to_remove.update(group[1:])
            elif keep == "last":
                # Keep last, remove rest
                indices_to_remove.update(group[:-1])
            elif keep == "best":
                # Keep highest quality score, remove rest
                group_samples = [(idx, dataset[idx]) for idx in group]

                # Sort by quality score (if available)
                def get_quality(item):
                    idx, sample = item
                    return sample.metadata.quality_score or 0.0

                group_samples.sort(key=get_quality, reverse=True)

                # Keep best, remove rest
                indices_to_remove.update([idx for idx, _ in group_samples[1:]])

        # Filter dataset
        unique = [sample for idx, sample in enumerate(dataset)
                  if idx not in indices_to_remove]

        removed_indices = sorted(list(indices_to_remove))

        logger.info(
            f"Removed {len(removed_indices)} near-duplicates. "
            f"Remaining: {len(unique)}/{len(dataset)}"
        )

        return unique, removed_indices

    def _build_duplicate_groups(
        self,
        duplicates: List[Tuple[int, int, float]],
        dataset_size: int
    ) -> List[List[int]]:
        """
        Build groups of related duplicates.

        If A is similar to B and B is similar to C,
        they all belong to the same group.

        Args:
            duplicates: List of (idx1, idx2, similarity) tuples
            dataset_size: Size of dataset

        Returns:
            List of groups (each group is a list of indices)
        """
        # Build adjacency list
        graph = defaultdict(set)
        for i, j, _ in duplicates:
            graph[i].add(j)
            graph[j].add(i)

        # Find connected components (groups)
        visited = set()
        groups = []

        for node in range(dataset_size):
            if node in visited or node not in graph:
                continue

            # BFS to find all connected nodes
            group = []
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                group.append(current)

                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            groups.append(sorted(group))

        return groups

    def compute_statistics(self, dataset: List[Sample]) -> Dict:
        """
        Compute deduplication statistics.

        Args:
            dataset: Dataset to analyze

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_samples": len(dataset),
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "unique_samples": len(dataset),
        }

        # Count exact duplicates
        _, exact_dups = self.remove_exact_duplicates(dataset.copy())
        stats["exact_duplicates"] = len(exact_dups)

        # Count near duplicates
        near_dups = self.detect_near_duplicates(dataset)
        stats["near_duplicates"] = len(near_dups)

        stats["unique_samples"] = stats["total_samples"] - stats["exact_duplicates"]

        return stats
