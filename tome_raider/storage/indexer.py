"""Fast indexing and filtering for datasets."""

from typing import List, Dict, Any, Set
from collections import defaultdict
from loguru import logger

from ..sources.base import Sample


class DatasetIndexer:
    """Build and query indexes for fast filtering."""

    def __init__(self):
        """Initialize indexer."""
        self.index = None
        self.dataset = None

    def build_index(self, dataset: List[Sample]) -> Dict[str, Any]:
        """
        Build searchable index for dataset.

        Args:
            dataset: Dataset to index

        Returns:
            Index dictionary
        """
        logger.info(f"Building index for {len(dataset)} samples")

        self.dataset = dataset
        self.index = {
            "by_quality": defaultdict(list),
            "by_source": defaultdict(list),
            "by_tag": defaultdict(list),
            "by_status": defaultdict(list),
        }

        for idx, sample in enumerate(dataset):
            # Index by quality score (bucketed by 0.1)
            quality = sample.metadata.quality_score
            if quality is not None:
                quality_bucket = round(quality, 1)
                self.index["by_quality"][quality_bucket].append(idx)

            # Index by source type
            source = sample.metadata.source_type
            self.index["by_source"][source].append(idx)

            # Index by tags
            for tag in sample.metadata.tags:
                self.index["by_tag"][tag].append(idx)

            # Index by review status
            status = sample.metadata.review_status
            self.index["by_status"][status].append(idx)

        logger.info("Index built successfully")
        return self.index

    def filter(
        self,
        quality_min: float = None,
        quality_max: float = None,
        source_types: List[str] = None,
        tags: List[str] = None,
        tags_mode: str = "any",  # "any" or "all"
        review_statuses: List[str] = None,
    ) -> List[Sample]:
        """
        Filter dataset using index.

        Args:
            quality_min: Minimum quality score
            quality_max: Maximum quality score
            source_types: Filter by source types
            tags: Filter by tags
            tags_mode: "any" (OR) or "all" (AND) for tags
            review_statuses: Filter by review statuses

        Returns:
            Filtered dataset
        """
        if not self.index or not self.dataset:
            raise ValueError("Index not built. Call build_index() first.")

        # Start with all indices
        matching_indices = set(range(len(self.dataset)))

        # Filter by quality
        if quality_min is not None or quality_max is not None:
            quality_indices = set()

            for quality_bucket, indices in self.index["by_quality"].items():
                if quality_min is not None and quality_bucket < quality_min:
                    continue
                if quality_max is not None and quality_bucket > quality_max:
                    continue

                quality_indices.update(indices)

            matching_indices &= quality_indices

        # Filter by source types
        if source_types:
            source_indices = set()
            for source in source_types:
                source_indices.update(self.index["by_source"].get(source, []))

            matching_indices &= source_indices

        # Filter by tags
        if tags:
            if tags_mode == "any":
                # OR: sample must have at least one tag
                tag_indices = set()
                for tag in tags:
                    tag_indices.update(self.index["by_tag"].get(tag, []))

                matching_indices &= tag_indices

            elif tags_mode == "all":
                # AND: sample must have all tags
                for tag in tags:
                    tag_indices = set(self.index["by_tag"].get(tag, []))
                    matching_indices &= tag_indices

        # Filter by review status
        if review_statuses:
            status_indices = set()
            for status in review_statuses:
                status_indices.update(self.index["by_status"].get(status, []))

            matching_indices &= status_indices

        # Get filtered samples
        filtered = [self.dataset[idx] for idx in sorted(matching_indices)]

        logger.info(
            f"Filtered to {len(filtered)}/{len(self.dataset)} samples"
        )

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Statistics dictionary
        """
        if not self.index:
            return {}

        stats = {
            "total_samples": len(self.dataset) if self.dataset else 0,
            "quality_distribution": {
                str(k): len(v) for k, v in self.index["by_quality"].items()
            },
            "source_distribution": {
                k: len(v) for k, v in self.index["by_source"].items()
            },
            "tag_distribution": {
                k: len(v) for k, v in self.index["by_tag"].items()
            },
            "status_distribution": {
                k: len(v) for k, v in self.index["by_status"].items()
            },
        }

        return stats

    def find_by_id(self, sample_id: str) -> Sample:
        """
        Find sample by ID.

        Args:
            sample_id: Sample ID

        Returns:
            Sample or None
        """
        if not self.dataset:
            raise ValueError("Index not built")

        for sample in self.dataset:
            if sample.id == sample_id:
                return sample

        return None

    def find_by_text(
        self,
        query: str,
        field: str = "instruction",
        case_sensitive: bool = False
    ) -> List[Sample]:
        """
        Find samples by text search.

        Args:
            query: Search query
            field: Field to search in
            case_sensitive: Whether search is case-sensitive

        Returns:
            Matching samples
        """
        if not self.dataset:
            raise ValueError("Index not built")

        if not case_sensitive:
            query = query.lower()

        matches = []

        for sample in self.dataset:
            if field == "instruction":
                text = sample.instruction
            elif field == "response":
                text = sample.response
            else:
                text = str(sample.to_dict().get(field, ""))

            if not case_sensitive:
                text = text.lower()

            if query in text:
                matches.append(sample)

        logger.info(f"Found {len(matches)} samples matching '{query}'")
        return matches
