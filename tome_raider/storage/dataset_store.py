"""File-based dataset storage with indexing."""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from loguru import logger
import pandas as pd

from ..sources.base import Sample


class DatasetStore:
    """Store and load datasets with indexing."""

    def __init__(self, base_path: str = "./datasets"):
        """
        Initialize dataset store.

        Args:
            base_path: Base directory for datasets
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.base_path / ".index"
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load dataset index."""
        index_file = self.index_path / "datasets.json"

        if not index_file.exists():
            return {}

        try:
            with open(index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return {}

    def _save_index(self):
        """Save dataset index."""
        index_file = self.index_path / "datasets.json"

        try:
            with open(index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def save(
        self,
        dataset: List[Sample],
        name: str,
        format: str = "jsonl",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save dataset.

        Args:
            dataset: Dataset to save
            name: Dataset name
            format: File format (jsonl, json, csv, parquet)
            metadata: Additional metadata

        Returns:
            Path to saved file
        """
        logger.info(f"Saving dataset '{name}' ({len(dataset)} samples) as {format}")

        # Create filename
        filename = f"{name}.{format}"
        filepath = self.base_path / filename

        # Save file
        if format == "jsonl":
            self._save_jsonl(dataset, filepath)
        elif format == "json":
            self._save_json(dataset, filepath)
        elif format == "csv":
            self._save_csv(dataset, filepath)
        elif format == "parquet":
            self._save_parquet(dataset, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Update index
        self.index[name] = {
            "path": str(filepath),
            "format": format,
            "sample_count": len(dataset),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": self._compute_metadata(dataset),
            "custom_metadata": metadata or {},
        }

        self._save_index()

        logger.info(f"Dataset saved: {filepath}")
        return str(filepath)

    def load(self, name: str) -> List[Sample]:
        """
        Load dataset by name.

        Args:
            name: Dataset name

        Returns:
            List of samples
        """
        if name not in self.index:
            raise ValueError(f"Dataset not found: {name}")

        info = self.index[name]
        filepath = Path(info["path"])
        format = info["format"]

        logger.info(f"Loading dataset '{name}' from {filepath}")

        if format == "jsonl":
            dataset = self._load_jsonl(filepath)
        elif format == "json":
            dataset = self._load_json(filepath)
        elif format == "csv":
            dataset = self._load_csv(filepath)
        elif format == "parquet":
            dataset = self._load_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets.

        Returns:
            List of dataset information dictionaries
        """
        return [
            {"name": name, **info}
            for name, info in self.index.items()
        ]

    def delete(self, name: str):
        """
        Delete dataset.

        Args:
            name: Dataset name
        """
        if name not in self.index:
            raise ValueError(f"Dataset not found: {name}")

        info = self.index[name]
        filepath = Path(info["path"])

        # Delete file
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted file: {filepath}")

        # Remove from index
        del self.index[name]
        self._save_index()

        logger.info(f"Dataset deleted: {name}")

    def update(self, name: str, dataset: List[Sample]):
        """
        Update existing dataset.

        Args:
            name: Dataset name
            dataset: New dataset
        """
        if name not in self.index:
            raise ValueError(f"Dataset not found: {name}")

        info = self.index[name]
        format = info["format"]

        # Preserve existing custom_metadata when updating
        existing_custom_metadata = info.get("custom_metadata", {})

        # Save with same format and preserved metadata
        self.save(dataset, name, format, metadata=existing_custom_metadata)

    def _compute_metadata(self, dataset: List[Sample]) -> Dict[str, Any]:
        """Compute dataset metadata."""
        if not dataset:
            return {}

        # Aggregate statistics
        source_types = defaultdict(int)
        tags = defaultdict(int)
        quality_scores = []
        review_statuses = defaultdict(int)

        for sample in dataset:
            # Source types
            source_type = sample.metadata.source_type
            source_types[source_type] += 1

            # Tags
            for tag in sample.metadata.tags:
                tags[tag] += 1

            # Quality scores
            if sample.metadata.quality_score is not None:
                quality_scores.append(sample.metadata.quality_score)

            # Review status
            status = sample.metadata.review_status
            review_statuses[status] += 1

        metadata = {
            "source_types": dict(source_types),
            "tags": dict(tags),
            "review_statuses": dict(review_statuses),
        }

        if quality_scores:
            import numpy as np
            metadata["quality_statistics"] = {
                "mean": float(np.mean(quality_scores)),
                "std": float(np.std(quality_scores)),
                "min": float(np.min(quality_scores)),
                "max": float(np.max(quality_scores)),
            }

        return metadata

    def _save_jsonl(self, dataset: List[Sample], filepath: Path):
        """Save as JSONL."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in dataset:
                json.dump(sample.to_dict(), f, ensure_ascii=False)
                f.write('\n')

    def _load_jsonl(self, filepath: Path) -> List[Sample]:
        """Load from JSONL."""
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(Sample.from_dict(data))
        return samples

    def _save_json(self, dataset: List[Sample], filepath: Path):
        """Save as JSON."""
        data = [sample.to_dict() for sample in dataset]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, filepath: Path) -> List[Sample]:
        """Load from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return [Sample.from_dict(item) for item in data]
        else:
            return [Sample.from_dict(data)]

    def _save_csv(self, dataset: List[Sample], filepath: Path):
        """Save as CSV."""
        if not dataset:
            return

        # Flatten samples
        rows = []
        for sample in dataset:
            row = {
                "instruction": sample.instruction,
                "response": sample.response,
                "source_type": sample.metadata.source_type,
                "source_url": sample.metadata.source_url,
                "quality_score": sample.metadata.quality_score,
                "review_status": sample.metadata.review_status,
                "tags": ",".join(sample.metadata.tags),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def _load_csv(self, filepath: Path) -> List[Sample]:
        """Load from CSV."""
        df = pd.read_csv(filepath)
        samples = []

        for _, row in df.iterrows():
            from ..sources.base import SourceMetadata

            # Parse tags
            tags = row.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                tags = []

            metadata = SourceMetadata(
                source_type=row.get("source_type", "unknown"),
                source_url=row.get("source_url"),
                quality_score=row.get("quality_score"),
                review_status=row.get("review_status", "pending"),
                tags=tags,
            )

            sample = Sample(
                instruction=row["instruction"],
                response=row["response"],
                metadata=metadata,
            )

            samples.append(sample)

        return samples

    def _save_parquet(self, dataset: List[Sample], filepath: Path):
        """Save as Parquet."""
        if not dataset:
            return

        # Convert to flat dict format
        rows = [sample.to_dict() for sample in dataset]

        # Flatten metadata
        flat_rows = []
        for row in rows:
            flat = {
                "instruction": row["instruction"],
                "response": row["response"],
            }

            # Flatten metadata
            metadata = row.get("metadata", {})
            for key, value in metadata.items():
                if key == "tags":
                    flat["tags"] = ",".join(value) if isinstance(value, list) else str(value)
                else:
                    flat[f"metadata_{key}"] = value

            flat_rows.append(flat)

        df = pd.DataFrame(flat_rows)
        df.to_parquet(filepath, index=False)

    def _load_parquet(self, filepath: Path) -> List[Sample]:
        """Load from Parquet."""
        df = pd.read_parquet(filepath)
        samples = []

        for _, row in df.iterrows():
            from ..sources.base import SourceMetadata

            # Extract metadata fields
            metadata_dict = {}
            tags = []

            for col in df.columns:
                if col.startswith("metadata_"):
                    key = col.replace("metadata_", "")
                    metadata_dict[key] = row[col]
                elif col == "tags":
                    tags_str = row[col]
                    if isinstance(tags_str, str):
                        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            metadata = SourceMetadata(
                source_type=metadata_dict.get("source_type", "unknown"),
                source_url=metadata_dict.get("source_url"),
                quality_score=metadata_dict.get("quality_score"),
                review_status=metadata_dict.get("review_status", "pending"),
                tags=tags,
            )

            sample = Sample(
                instruction=row["instruction"],
                response=row["response"],
                metadata=metadata,
            )

            samples.append(sample)

        return samples
