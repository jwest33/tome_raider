"""Dataset repair utilities for fixing validation errors."""

import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from loguru import logger
from copy import deepcopy

from ..sources.base import Sample
from .validator import DatasetValidator, ValidationResult


@dataclass
class RepairResult:
    """Result of a repair operation."""

    original_sample: Sample
    repaired_sample: Optional[Sample]
    success: bool
    changes: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "changes": self.changes,
            "error": self.error,
        }


class RepairStrategy:
    """Base class for repair strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with config."""
        self.config = config or {}

    def repair(self, sample: Sample, validation_result: ValidationResult) -> RepairResult:
        """
        Repair a sample.

        Args:
            sample: Sample to repair
            validation_result: Validation result with errors

        Returns:
            RepairResult
        """
        raise NotImplementedError


class TruncateStrategy(RepairStrategy):
    """Truncate text to meet length requirements."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize truncation strategy."""
        super().__init__(config)
        self.max_instruction_length = self.config.get("max_instruction_length", 5000)
        self.max_response_length = self.config.get("max_response_length", 10000)
        self.preserve_sentences = self.config.get("preserve_sentences", True)

    def repair(self, sample: Sample, validation_result: ValidationResult) -> RepairResult:
        """Repair by truncating text intelligently."""
        repaired = deepcopy(sample)
        changes = []

        # Check if instruction needs truncating
        if sample.instruction and len(sample.instruction) > self.max_instruction_length:
            original_len = len(sample.instruction)
            repaired.instruction = self._truncate_text(
                sample.instruction,
                self.max_instruction_length
            )
            new_len = len(repaired.instruction)
            changes.append(
                f"Truncated instruction from {original_len} to {new_len} chars"
            )

        # Check if response needs truncating
        if sample.response and len(sample.response) > self.max_response_length:
            original_len = len(sample.response)
            repaired.response = self._truncate_text(
                sample.response,
                self.max_response_length
            )
            new_len = len(repaired.response)
            changes.append(
                f"Truncated response from {original_len} to {new_len} chars"
            )

        return RepairResult(
            original_sample=sample,
            repaired_sample=repaired,
            success=bool(changes),
            changes=changes
        )

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text intelligently at sentence boundaries.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        if not self.preserve_sentences:
            return text[:max_length].rstrip()

        # Try to truncate at sentence boundaries
        # Look for sentence endings: . ! ? followed by space or newline
        sentence_endings = re.finditer(r'[.!?][\s\n]', text[:max_length])

        # Find the last sentence ending before max_length
        last_ending = None
        for match in sentence_endings:
            last_ending = match.end()

        if last_ending and last_ending > max_length * 0.7:  # At least 70% of max
            return text[:last_ending].rstrip()

        # If no good sentence boundary, try paragraph breaks
        paragraph_breaks = [m.end() for m in re.finditer(r'\n\n', text[:max_length])]
        if paragraph_breaks:
            last_break = paragraph_breaks[-1]
            if last_break > max_length * 0.7:
                return text[:last_break].rstrip()

        # If no good break point, truncate at word boundary
        truncated = text[:max_length].rstrip()
        last_space = truncated.rfind(' ')

        if last_space > max_length * 0.9:  # At least 90% of max
            return truncated[:last_space].rstrip()

        return truncated


class SummarizeStrategy(RepairStrategy):
    """Use LLM to intelligently condense text while preserving meaning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize summarization strategy."""
        super().__init__(config)
        self.max_instruction_length = self.config.get("max_instruction_length", 5000)
        self.max_response_length = self.config.get("max_response_length", 10000)
        self.model_path = self.config.get("model_path")
        self.llm_manager = None

        if not self.model_path:
            raise ValueError("model_path is required for SummarizeStrategy")

    def _ensure_model_loaded(self):
        """Ensure LLM model is loaded."""
        if self.llm_manager is None:
            from ..generation.llm_manager import LlamaServerManager

            logger.info(f"Loading model for summarization: {self.model_path}")
            self.llm_manager = LlamaServerManager()

            success = self.llm_manager.load_model(
                model_path=self.model_path,
                context_size=8192,
                n_gpu_layers=-1  # Use GPU if available
            )

            if not success:
                raise RuntimeError(f"Failed to load model: {self.model_path}")

    def _summarize_text(self, text: str, max_length: int, text_type: str = "instruction") -> str:
        """
        Use LLM to condense text while preserving meaning.

        Args:
            text: Text to condense
            max_length: Maximum character length
            text_type: Type of text (instruction or response)

        Returns:
            Condensed text
        """
        self._ensure_model_loaded()

        # Calculate target length (aim for 90% of max to leave margin)
        target_length = int(max_length * 0.9)

        if text_type == "instruction":
            prompt = f"""You are a text condensing assistant. Your task is to shorten the following instruction while preserving ALL key requirements, constraints, and intent.

Original instruction ({len(text)} characters):
{text}

Create a condensed version that:
1. Preserves all essential requirements and constraints
2. Maintains clarity and specificity
3. Keeps the same intent and expected output
4. Is approximately {target_length} characters or less
5. Remains grammatically correct and coherent

Condensed instruction:"""
        else:  # response
            prompt = f"""You are a text condensing assistant. Your task is to shorten the following response while preserving the core answer and key information.

Original response ({len(text)} characters):
{text}

Create a condensed version that:
1. Preserves the main answer and conclusion
2. Keeps essential details and explanations
3. Maintains logical flow and coherence
4. Is approximately {target_length} characters or less
5. Remains grammatically correct

Condensed response:"""

        logger.debug(f"Summarizing {text_type} from {len(text)} to ~{target_length} chars")

        try:
            condensed = self.llm_manager.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused output
                max_tokens=2048,
                top_p=0.9
            )

            if not condensed:
                logger.warning("LLM returned empty response, falling back to truncation")
                truncate = TruncateStrategy(self.config)
                return truncate._truncate_text(text, max_length)

            # Verify it's actually shorter
            if len(condensed) > max_length:
                logger.warning(f"LLM output still too long ({len(condensed)} > {max_length}), truncating")
                truncate = TruncateStrategy(self.config)
                return truncate._truncate_text(condensed, max_length)

            return condensed.strip()

        except Exception as e:
            logger.error(f"Summarization failed: {e}, falling back to truncation")
            truncate = TruncateStrategy(self.config)
            return truncate._truncate_text(text, max_length)

    def repair(self, sample: Sample, validation_result: ValidationResult) -> RepairResult:
        """Repair by using LLM to condense text."""
        repaired = deepcopy(sample)
        changes = []

        try:
            # Check if instruction needs condensing
            if sample.instruction and len(sample.instruction) > self.max_instruction_length:
                original_len = len(sample.instruction)
                repaired.instruction = self._summarize_text(
                    sample.instruction,
                    self.max_instruction_length,
                    "instruction"
                )
                new_len = len(repaired.instruction)
                changes.append(
                    f"Condensed instruction from {original_len} to {new_len} chars using LLM"
                )

            # Check if response needs condensing
            if sample.response and len(sample.response) > self.max_response_length:
                original_len = len(sample.response)
                repaired.response = self._summarize_text(
                    sample.response,
                    self.max_response_length,
                    "response"
                )
                new_len = len(repaired.response)
                changes.append(
                    f"Condensed response from {original_len} to {new_len} chars using LLM"
                )

            return RepairResult(
                original_sample=sample,
                repaired_sample=repaired,
                success=bool(changes),
                changes=changes
            )

        except Exception as e:
            logger.error(f"Summarization repair failed: {e}")
            return RepairResult(
                original_sample=sample,
                repaired_sample=None,
                success=False,
                changes=[],
                error=str(e)
            )

    def cleanup(self):
        """Unload model and cleanup resources."""
        if self.llm_manager:
            logger.info("Unloading summarization model")
            self.llm_manager.unload_model()
            self.llm_manager = None


class SplitStrategy(RepairStrategy):
    """Split long samples into multiple shorter samples."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize split strategy."""
        super().__init__(config)
        self.max_instruction_length = self.config.get("max_instruction_length", 5000)
        self.max_response_length = self.config.get("max_response_length", 10000)

    def repair(self, sample: Sample, validation_result: ValidationResult) -> RepairResult:
        """
        Repair by splitting into multiple samples.

        Note: This returns the first split sample. Full implementation would
        need to return multiple samples.
        """
        # For now, just use truncation as a fallback
        # Full split implementation would require returning List[Sample]
        truncate = TruncateStrategy(self.config)
        return truncate.repair(sample, validation_result)


class DatasetRepairer:
    """Repair datasets with validation errors."""

    def __init__(
        self,
        validator: Optional[DatasetValidator] = None,
        strategy: str = "truncate",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dataset repairer.

        Args:
            validator: Dataset validator to use
            strategy: Repair strategy name (truncate, summarize, split)
            config: Configuration dict
        """
        self.config = config or {}
        self.validator = validator or DatasetValidator(self.config.get("validation", {}))

        # Initialize strategy
        strategy_config = {
            "max_instruction_length": self.validator.max_instruction_length,
            "max_response_length": self.validator.max_response_length,
            **self.config.get("repair", {})
        }

        if strategy == "truncate":
            self.strategy = TruncateStrategy(strategy_config)
        elif strategy == "summarize":
            self.strategy = SummarizeStrategy(strategy_config)
        elif strategy == "split":
            self.strategy = SplitStrategy(strategy_config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy_name = strategy

    def repair_sample(self, sample: Sample) -> RepairResult:
        """
        Repair a single sample.

        Args:
            sample: Sample to repair

        Returns:
            RepairResult
        """
        # Validate first
        validation_result = self.validator.validate_sample(sample)

        if validation_result.valid:
            return RepairResult(
                original_sample=sample,
                repaired_sample=sample,
                success=True,
                changes=["No repairs needed - sample is valid"]
            )

        # Apply repair strategy
        try:
            repair_result = self.strategy.repair(sample, validation_result)

            # Validate repaired sample
            if repair_result.repaired_sample:
                repaired_validation = self.validator.validate_sample(
                    repair_result.repaired_sample
                )

                if not repaired_validation.valid:
                    # Repair didn't fix all issues
                    return RepairResult(
                        original_sample=sample,
                        repaired_sample=repair_result.repaired_sample,
                        success=False,
                        changes=repair_result.changes,
                        error=f"Repair incomplete: {', '.join(repaired_validation.errors)}"
                    )

            return repair_result

        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return RepairResult(
                original_sample=sample,
                repaired_sample=None,
                success=False,
                changes=[],
                error=str(e)
            )

    def repair_dataset(
        self,
        dataset: List[Sample],
        skip_valid: bool = True
    ) -> Dict[str, Any]:
        """
        Repair entire dataset.

        Args:
            dataset: Dataset to repair
            skip_valid: Skip samples that are already valid

        Returns:
            Dict with repaired dataset and statistics
        """
        logger.info(f"Repairing dataset with {len(dataset)} samples using {self.strategy_name} strategy")

        repaired_samples = []
        repair_stats = {
            "total": len(dataset),
            "already_valid": 0,
            "repaired": 0,
            "failed": 0,
            "changes": []
        }

        for idx, sample in enumerate(dataset):
            # Validate first
            validation_result = self.validator.validate_sample(sample)

            if validation_result.valid and skip_valid:
                repaired_samples.append(sample)
                repair_stats["already_valid"] += 1
                continue

            # Attempt repair
            repair_result = self.repair_sample(sample)

            if repair_result.success and repair_result.repaired_sample:
                repaired_samples.append(repair_result.repaired_sample)
                repair_stats["repaired"] += 1
                repair_stats["changes"].append({
                    "sample_index": idx,
                    "changes": repair_result.changes
                })
            else:
                # Keep original sample even if repair failed
                repaired_samples.append(sample)
                repair_stats["failed"] += 1
                if repair_result.error:
                    logger.warning(f"Sample {idx} repair failed: {repair_result.error}")

        logger.info(
            f"Repair complete: {repair_stats['already_valid']} valid, "
            f"{repair_stats['repaired']} repaired, {repair_stats['failed']} failed"
        )

        return {
            "dataset": repaired_samples,
            "statistics": repair_stats
        }

    def repair_and_validate(self, dataset: List[Sample]) -> Dict[str, Any]:
        """
        Repair dataset and provide before/after validation.

        Args:
            dataset: Dataset to repair

        Returns:
            Dict with repaired dataset and validation results
        """
        # Validate before
        before_validation = self.validator.validate_all(dataset)

        # Repair
        repair_result = self.repair_dataset(dataset)

        # Validate after
        after_validation = self.validator.validate_all(repair_result["dataset"])

        return {
            "dataset": repair_result["dataset"],
            "repair_statistics": repair_result["statistics"],
            "before_validation": before_validation,
            "after_validation": after_validation,
            "improvement": {
                "valid_count": after_validation["valid"] - before_validation["valid"],
                "invalid_count": before_validation["invalid"] - after_validation["invalid"],
            }
        }

    def cleanup(self):
        """Cleanup strategy resources (e.g., unload models)."""
        if hasattr(self.strategy, 'cleanup'):
            self.strategy.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
