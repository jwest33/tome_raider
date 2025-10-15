"""Config-driven pipeline for dataset operations."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
from loguru import logger

from ..sources.base import Sample
from ..sources.file_loader import FileLoader
from ..sources.dataset_loader import DatasetLoader
from ..generation.generator import DataGenerator
from ..quality.validator import DatasetValidator
from ..quality.quality_scorer import QualityScorer
from ..quality.deduplicator import Deduplicator
from ..storage.dataset_store import DatasetStore
from ..storage.indexer import DatasetIndexer


@dataclass
class Operation:
    """Single operation in pipeline."""

    name: str
    type: str  # "source", "generate", "validate", "quality", "transform", "export"
    config: Dict[str, Any]


class DatasetState:
    """Manages current dataset state."""

    def __init__(self):
        """Initialize dataset state."""
        self.dataset: List[Sample] = []
        self.history: List[Dict[str, Any]] = []

    def update(self, dataset: List[Sample], operation: str):
        """
        Update dataset and record history.

        Args:
            dataset: New dataset
            operation: Operation name
        """
        # Save snapshot to history
        snapshot = {
            "operation": operation,
            "sample_count": len(self.dataset),
        }
        self.history.append(snapshot)

        # Update current dataset
        self.dataset = dataset

    def snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot."""
        return {
            "dataset": self.dataset.copy(),
            "sample_count": len(self.dataset),
        }

    def restore(self, snapshot: Dict[str, Any]):
        """Restore from snapshot."""
        self.dataset = snapshot["dataset"].copy()


class DatasetPipeline:
    """Stateful pipeline for dataset operations."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to config file (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.state = DatasetState()
        self.undo_stack: List[Dict[str, Any]] = []

        # Initialize components
        self.store = DatasetStore(
            self.config.get("storage", {}).get("base_path", "./datasets")
        )
        self.validator = DatasetValidator(self.config.get("validation", {}))
        self.scorer = QualityScorer()
        self.deduplicator = Deduplicator(self.config.get("quality", {}))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def run(self, operations: Optional[List[Operation]] = None):
        """
        Execute pipeline operations.

        Args:
            operations: List of operations (uses config if None)
        """
        if operations is None:
            # Load from config
            operations = self._parse_operations_from_config()

        logger.info(f"Running pipeline with {len(operations)} operations")

        for idx, op in enumerate(operations):
            logger.info(f"[{idx + 1}/{len(operations)}] Executing: {op.name} ({op.type})")

            try:
                self._execute_operation(op)
                logger.info(f"Operation '{op.name}' completed successfully")
            except Exception as e:
                logger.error(f"Operation '{op.name}' failed: {e}")
                raise

        logger.info(
            f"Pipeline complete. Final dataset: {len(self.state.dataset)} samples"
        )

    def _parse_operations_from_config(self) -> List[Operation]:
        """Parse operations from config."""
        operations_config = self.config.get("operations", [])
        operations = []

        for op_config in operations_config:
            op = Operation(
                name=op_config.get("name", "Unnamed"),
                type=op_config.get("type"),
                config=op_config.get("config", {}),
            )
            operations.append(op)

        return operations

    def _execute_operation(self, op: Operation):
        """Execute single operation."""
        # Save state for undo
        self.undo_stack.append(self.state.snapshot())

        if op.type == "source":
            self._execute_source(op)
        elif op.type == "generate":
            self._execute_generate(op)
        elif op.type == "validate":
            self._execute_validate(op)
        elif op.type == "quality":
            self._execute_quality(op)
        elif op.type == "deduplicate":
            self._execute_deduplicate(op)
        elif op.type == "filter":
            self._execute_filter(op)
        elif op.type == "save":
            self._execute_save(op)
        elif op.type == "load":
            self._execute_load(op)
        else:
            raise ValueError(f"Unknown operation type: {op.type}")

    def _execute_source(self, op: Operation):
        """Execute source loading operation."""
        source_type = op.config.get("source_type")

        if source_type == "file":
            source = FileLoader(op.config)
        elif source_type == "huggingface":
            source = DatasetLoader(op.config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Load samples
        new_samples = list(source.load())
        logger.info(f"Loaded {len(new_samples)} samples from {source_type}")

        # Add to dataset (append or replace)
        if op.config.get("append", True):
            self.state.dataset.extend(new_samples)
        else:
            self.state.dataset = new_samples

        self.state.update(self.state.dataset, op.name)

    def _execute_generate(self, op: Operation):
        """Execute generation operation."""
        generator = DataGenerator(self.config.get("llm", {}))

        # Generate samples
        new_samples = generator.generate(
            strategy=op.config.get("strategy"),
            model=op.config.get("model"),
            **op.config
        )

        logger.info(f"Generated {len(new_samples)} samples")

        # Add to dataset
        self.state.dataset.extend(new_samples)
        self.state.update(self.state.dataset, op.name)

        generator.close()

    def _execute_validate(self, op: Operation):
        """Execute validation operation."""
        # Validate all samples
        validation_result = self.validator.validate_all(
            self.state.dataset,
            stop_on_first_error=op.config.get("stop_on_error", False)
        )

        logger.info(
            f"Validation: {validation_result['valid']}/{validation_result['total']} valid"
        )

        # Remove invalid if strict mode
        if op.config.get("remove_invalid", False):
            valid_samples = []
            for idx, sample in enumerate(self.state.dataset):
                result = self.validator.validate_sample(sample)
                if result.valid:
                    valid_samples.append(sample)

            removed = len(self.state.dataset) - len(valid_samples)
            logger.info(f"Removed {removed} invalid samples")

            self.state.dataset = valid_samples
            self.state.update(self.state.dataset, op.name)

    def _execute_quality(self, op: Operation):
        """Execute quality scoring operation."""
        logger.info("Scoring quality for all samples")

        for sample in self.state.dataset:
            score = self.scorer.score_sample(sample)
            sample.metadata.quality_score = score.overall
            sample.metadata.custom["quality_components"] = score.components

        logger.info("Quality scoring complete")
        self.state.update(self.state.dataset, op.name)

    def _execute_deduplicate(self, op: Operation):
        """Execute deduplication operation."""
        if op.config.get("exact", True):
            self.state.dataset, removed = self.deduplicator.remove_exact_duplicates(
                self.state.dataset
            )
            logger.info(f"Removed {len(removed)} exact duplicates")

        if op.config.get("near", False):
            threshold = op.config.get("threshold", 0.85)
            self.state.dataset, removed = self.deduplicator.remove_near_duplicates(
                self.state.dataset,
                threshold=threshold
            )
            logger.info(f"Removed {len(removed)} near duplicates")

        self.state.update(self.state.dataset, op.name)

    def _execute_filter(self, op: Operation):
        """Execute filtering operation."""
        indexer = DatasetIndexer()
        indexer.build_index(self.state.dataset)

        filtered = indexer.filter(
            quality_min=op.config.get("quality_min"),
            quality_max=op.config.get("quality_max"),
            source_types=op.config.get("source_types"),
            tags=op.config.get("tags"),
            tags_mode=op.config.get("tags_mode", "any"),
            review_statuses=op.config.get("review_statuses"),
        )

        self.state.dataset = filtered
        self.state.update(self.state.dataset, op.name)

    def _execute_save(self, op: Operation):
        """Execute save operation."""
        name = op.config.get("name") or self.config.get("name", "dataset")
        format = op.config.get("format", "jsonl")

        self.store.save(self.state.dataset, name, format)
        logger.info(f"Saved dataset '{name}' ({len(self.state.dataset)} samples)")

    def _execute_load(self, op: Operation):
        """Execute load operation."""
        name = op.config.get("name")

        if not name:
            raise ValueError("Dataset name required for load operation")

        self.state.dataset = self.store.load(name)
        self.state.update(self.state.dataset, op.name)
        logger.info(f"Loaded dataset '{name}' ({len(self.state.dataset)} samples)")

    def undo(self):
        """Undo last operation."""
        if not self.undo_stack:
            logger.warning("Nothing to undo")
            return

        previous_state = self.undo_stack.pop()
        self.state.restore(previous_state)
        logger.info(
            f"Undone. Current dataset: {len(self.state.dataset)} samples"
        )

    def get_dataset(self) -> List[Sample]:
        """Get current dataset."""
        return self.state.dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.state.dataset:
            return {"sample_count": 0}

        indexer = DatasetIndexer()
        indexer.build_index(self.state.dataset)

        return {
            "sample_count": len(self.state.dataset),
            "operations_executed": len(self.state.history),
            **indexer.get_statistics(),
        }
