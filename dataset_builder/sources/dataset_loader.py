"""HuggingFace and Kaggle dataset loader."""

from typing import Iterator, Dict, Any, Optional
from loguru import logger

from .base import DataSource, Sample


class DatasetLoader(DataSource):
    """Load datasets from HuggingFace Hub."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader.

        Args:
            config: Configuration with 'dataset' key
        """
        super().__init__(config)

        self.dataset_name = config.get("dataset")
        self.split = config.get("split", "train")
        self.subset = config.get("subset", None)
        self.streaming = config.get("streaming", False)

        self.field_mapping = config.get("field_mapping", {})
        self.instruction_field = self.field_mapping.get("instruction", "instruction")
        self.response_field = self.field_mapping.get("response", "response")

        self.validate_config()

    def validate_config(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.dataset_name:
            raise ValueError("Dataset name is required")

        return True

    def load(self) -> Iterator[Sample]:
        """
        Load dataset from HuggingFace Hub.

        Yields:
            Sample objects
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading dataset: {self.dataset_name} (split: {self.split})")

            # Load dataset
            if self.subset:
                dataset = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split,
                    streaming=self.streaming,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=self.streaming,
                )

            logger.info(f"Dataset loaded successfully")

            # Iterate through dataset
            for idx, example in enumerate(dataset):
                try:
                    sample = self._example_to_sample(example, idx)
                    if sample:
                        yield sample
                except Exception as e:
                    logger.warning(f"Error parsing example {idx}: {e}")

        except ImportError:
            logger.error(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {e}")
            raise

    def _example_to_sample(
        self,
        example: Dict[str, Any],
        idx: int
    ) -> Optional[Sample]:
        """
        Convert dataset example to Sample.

        Args:
            example: Dataset example
            idx: Example index

        Returns:
            Sample object or None if invalid
        """
        # Extract instruction and response with field mapping
        instruction = example.get(self.instruction_field)
        response = example.get(self.response_field)

        # Try common alternative field names
        if not instruction:
            instruction = (
                example.get("input") or
                example.get("question") or
                example.get("query") or
                example.get("prompt") or
                example.get("text")
            )

        if not response:
            response = (
                example.get("output") or
                example.get("answer") or
                example.get("completion") or
                example.get("target") or
                example.get("label")
            )

        # Handle conversational formats
        if not instruction and "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list) and len(messages) >= 2:
                # Extract user and assistant messages
                user_msgs = [m for m in messages if m.get("role") == "user"]
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

                if user_msgs and assistant_msgs:
                    instruction = "\n\n".join([m.get("content", "") for m in user_msgs])
                    response = "\n\n".join([m.get("content", "") for m in assistant_msgs])

        # Handle conversations field
        if not instruction and "conversations" in example:
            convs = example["conversations"]
            if isinstance(convs, list) and len(convs) >= 2:
                instruction = convs[0].get("value", "") if isinstance(convs[0], dict) else str(convs[0])
                response = convs[1].get("value", "") if isinstance(convs[1], dict) else str(convs[1])

        if not instruction or not response:
            logger.debug(f"Missing instruction or response in example {idx}")
            return None

        # Create metadata
        metadata = self._create_metadata(
            source_url=f"https://huggingface.co/datasets/{self.dataset_name}",
            tags=example.get("tags", []),
            license=example.get("license"),
            author=example.get("author"),
        )

        # Create sample
        sample = Sample(
            instruction=str(instruction),
            response=str(response),
            metadata=metadata,
            id=example.get("id", f"{self.dataset_name}_{idx}"),
        )

        return sample

    def count(self) -> Optional[int]:
        """
        Count total samples in dataset.

        Returns:
            Number of samples or None if streaming
        """
        if self.streaming:
            return None

        try:
            from datasets import load_dataset

            if self.subset:
                dataset = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                )

            return len(dataset)
        except Exception as e:
            logger.warning(f"Failed to count dataset samples: {e}")
            return None
