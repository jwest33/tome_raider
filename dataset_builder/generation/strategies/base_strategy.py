"""Base class for generation strategies."""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
from loguru import logger

from ...sources.base import Sample, SourceMetadata


class GenerationStrategy(ABC):
    """Abstract base class for generation strategies."""

    def __init__(self, config: Dict[str, Any], llm_manager):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
            llm_manager: LLM manager instance
        """
        self.config = config
        self.llm = llm_manager

        # Common parameters
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 1024)

    @abstractmethod
    def generate(self) -> Iterator[Sample]:
        """
        Generate samples.

        Yields:
            Generated samples
        """
        pass

    def _create_sample(
        self,
        instruction: str,
        response: str,
        **metadata_kwargs
    ) -> Sample:
        """
        Create a sample with metadata.

        Args:
            instruction: Instruction text
            response: Response text
            **metadata_kwargs: Additional metadata fields

        Returns:
            Sample object
        """
        # Valid SourceMetadata fields
        valid_fields = {
            "source_url", "source_file", "source_line",
            "collection_timestamp", "license", "author",
            "quality_score", "review_status"
        }

        # Separate valid fields from custom fields
        standard_fields = {}
        custom_fields = {}

        for key, value in metadata_kwargs.items():
            if key == "tags":
                standard_fields["tags"] = value
            elif key in valid_fields:
                standard_fields[key] = value
            else:
                custom_fields[key] = value

        # Create metadata with separated fields
        metadata = SourceMetadata(
            source_type="generated",
            tags=standard_fields.get("tags", []),
            custom=custom_fields,
            **{k: v for k, v in standard_fields.items() if k != "tags"}
        )

        return Sample(
            instruction=instruction,
            response=response,
            metadata=metadata,
        )

    def _generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate with retry logic.

        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts
            **kwargs: Generation parameters

        Returns:
            Generated text

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                result = self.llm.generate(
                    prompt,
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", self.top_p),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    **{k: v for k, v in kwargs.items()
                       if k not in ["temperature", "top_p", "max_tokens"]}
                )

                if result:
                    return result

                logger.warning(f"Empty result on attempt {attempt + 1}")

            except Exception as e:
                logger.warning(f"Generation failed on attempt {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Generation failed after {max_retries} attempts")
