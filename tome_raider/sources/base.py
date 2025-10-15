"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SourceMetadata:
    """Metadata for a data source."""

    source_type: str
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    collection_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    license: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    review_status: str = "pending"
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_type": self.source_type,
            "source_url": self.source_url,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "collection_timestamp": self.collection_timestamp,
            "license": self.license,
            "author": self.author,
            "tags": self.tags,
            "quality_score": self.quality_score,
            "review_status": self.review_status,
            **self.custom,
        }


@dataclass
class Sample:
    """A single training sample with metadata."""

    instruction: str
    response: str
    metadata: SourceMetadata
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        result = {
            "instruction": self.instruction,
            "response": self.response,
            "metadata": self.metadata.to_dict(),
        }
        if self.id:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create sample from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            # Extract known fields
            metadata_obj = SourceMetadata(
                source_type=metadata.get("source_type", "unknown"),
                source_url=metadata.get("source_url"),
                source_file=metadata.get("source_file"),
                source_line=metadata.get("source_line"),
                collection_timestamp=metadata.get(
                    "collection_timestamp",
                    datetime.now().isoformat()
                ),
                license=metadata.get("license"),
                author=metadata.get("author"),
                tags=metadata.get("tags", []),
                quality_score=metadata.get("quality_score"),
                review_status=metadata.get("review_status", "pending"),
                custom={k: v for k, v in metadata.items() if k not in [
                    "source_type", "source_url", "source_file", "source_line",
                    "collection_timestamp", "license", "author", "tags",
                    "quality_score", "review_status"
                ]},
            )
        else:
            metadata_obj = metadata

        return cls(
            instruction=data.get("instruction", ""),
            response=data.get("response", ""),
            metadata=metadata_obj,
            id=data.get("id"),
        )


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data source.

        Args:
            config: Source configuration
        """
        self.config = config
        self.source_type = self.__class__.__name__.lower().replace("datasource", "")

    @abstractmethod
    def load(self) -> Iterator[Sample]:
        """
        Load data from source.

        Yields:
            Sample objects with metadata

        Raises:
            Exception: If loading fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate source configuration.

        Returns:
            True if config is valid

        Raises:
            ValueError: If config is invalid
        """
        pass

    def _create_metadata(
        self,
        source_url: Optional[str] = None,
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
        license: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> SourceMetadata:
        """
        Create metadata for a sample.

        Args:
            source_url: URL of the source
            source_file: File path of the source
            source_line: Line number in source file
            license: License information
            author: Author information
            tags: Tags for categorization
            **kwargs: Additional custom metadata

        Returns:
            SourceMetadata object
        """
        return SourceMetadata(
            source_type=self.source_type,
            source_url=source_url,
            source_file=source_file,
            source_line=source_line,
            license=license,
            author=author,
            tags=tags or [],
            custom=kwargs,
        )

    def count(self) -> Optional[int]:
        """
        Count total samples (if known).

        Returns:
            Number of samples or None if unknown
        """
        return None

    def __iter__(self) -> Iterator[Sample]:
        """Make data source iterable."""
        return self.load()

    def __repr__(self) -> str:
        """String representation of data source."""
        return f"{self.__class__.__name__}(type={self.source_type})"
