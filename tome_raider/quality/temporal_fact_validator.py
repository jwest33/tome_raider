"""Validator for temporal fact datasets."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from loguru import logger

from ..sources.base import Sample
from .validator import ValidationResult


class TemporalFactValidator:
    """Validates temporal fact datasets for embedding experiments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temporal fact validator.

        Args:
            config: Optional validation configuration
        """
        self.config = config or {}
        self.min_fact_length = self.config.get("min_fact_length", 20)
        self.max_fact_length = self.config.get("max_fact_length", 500)

    def validate_sample(self, sample: Sample) -> ValidationResult:
        """
        Validate a single temporal fact sample.

        Args:
            sample: Sample to validate

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(valid=True)

        # Get errors from existing validation logic
        errors = self._validate_sample(sample, idx=0)

        # Add errors to result
        for error in errors:
            result.add_error(error)

        return result

    def validate_all(self, dataset: List[Sample]) -> Dict[str, Any]:
        """
        Validate entire temporal fact dataset.

        Args:
            dataset: List of temporal fact samples

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {len(dataset)} temporal fact samples")

        errors = []
        warnings = []
        valid_count = 0
        invalid_count = 0

        # Track groups for group-level validation
        groups = defaultdict(list)

        for idx, sample in enumerate(dataset):
            sample_errors = self._validate_sample(sample, idx)

            if sample_errors:
                invalid_count += 1
                errors.extend(sample_errors)
            else:
                valid_count += 1

            # Collect samples by group for group-level validation
            group_id = sample.metadata.custom.get("group_id")
            if group_id:
                groups[group_id].append((idx, sample))

        # Validate groups
        group_warnings = self._validate_groups(groups)
        warnings.extend(group_warnings)

        logger.info(f"Validation complete: {valid_count}/{len(dataset)} valid, {invalid_count} invalid")

        return {
            "total": len(dataset),
            "valid": valid_count,
            "invalid": invalid_count,
            "errors": errors,
            "warnings": warnings,
            "group_count": len(groups),
        }

    def _validate_sample(self, sample: Sample, idx: int) -> List[str]:
        """
        Validate a single temporal fact sample.

        Args:
            sample: Sample to validate
            idx: Sample index

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check instruction (the fact text)
        if not sample.instruction or not sample.instruction.strip():
            errors.append(f"Sample {idx}: Missing or empty fact text (instruction)")
        elif len(sample.instruction) < self.min_fact_length:
            errors.append(f"Sample {idx}: Fact too short ({len(sample.instruction)} chars, min {self.min_fact_length})")
        elif len(sample.instruction) > self.max_fact_length:
            errors.append(f"Sample {idx}: Fact too long ({len(sample.instruction)} chars, max {self.max_fact_length})")

        # Response should be empty for temporal facts (embedding experiments)
        # This is valid - don't report as error

        # Check required metadata fields
        if not hasattr(sample.metadata, 'custom') or not sample.metadata.custom:
            errors.append(f"Sample {idx}: Missing custom metadata")
            return errors

        custom = sample.metadata.custom

        # Check timestamp
        timestamp = custom.get("timestamp")
        if not timestamp:
            errors.append(f"Sample {idx}: Missing timestamp in metadata")
        else:
            try:
                datetime.fromisoformat(timestamp)
            except (ValueError, TypeError) as e:
                errors.append(f"Sample {idx}: Invalid timestamp format: {timestamp}")

        # Check group_id
        if not custom.get("group_id"):
            errors.append(f"Sample {idx}: Missing group_id in metadata")

        # Check variation_id
        variation_id = custom.get("variation_id")
        if variation_id is None:
            errors.append(f"Sample {idx}: Missing variation_id in metadata")
        elif not isinstance(variation_id, int) or variation_id < 1:
            errors.append(f"Sample {idx}: Invalid variation_id: {variation_id}")

        return errors

    def _validate_groups(self, groups: Dict[str, List[tuple]]) -> List[str]:
        """
        Validate fact groups for consistency.

        Args:
            groups: Dictionary mapping group_id to list of (index, sample) tuples

        Returns:
            List of warning messages
        """
        warnings = []

        for group_id, samples in groups.items():
            # Check variation IDs are sequential
            variation_ids = [s[1].metadata.custom.get("variation_id") for s in samples]
            expected_ids = list(range(1, len(samples) + 1))

            if sorted(variation_ids) != expected_ids:
                warnings.append(
                    f"Group {group_id}: Variation IDs not sequential. "
                    f"Expected {expected_ids}, got {sorted(variation_ids)}"
                )

            # Check for duplicate facts within group
            facts = [s[1].instruction.strip().lower() for s in samples]
            if len(facts) != len(set(facts)):
                duplicates = [f for f in facts if facts.count(f) > 1]
                warnings.append(
                    f"Group {group_id}: Contains {len(duplicates)} duplicate facts"
                )

            # Check timestamps are sequential
            timestamps = []
            for idx, sample in samples:
                ts_str = sample.metadata.custom.get("timestamp")
                if ts_str:
                    try:
                        timestamps.append(datetime.fromisoformat(ts_str))
                    except:
                        pass

            if len(timestamps) == len(samples):
                if timestamps != sorted(timestamps):
                    warnings.append(f"Group {group_id}: Timestamps not in chronological order")

        return warnings

    def get_statistics(self, dataset: List[Sample]) -> Dict[str, Any]:
        """
        Get dataset statistics for temporal facts.

        Args:
            dataset: List of temporal fact samples

        Returns:
            Dictionary with statistics
        """
        if not dataset:
            return {}

        # Collect timestamps and groups
        timestamps = []
        groups = set()
        facts_per_group = defaultdict(int)

        for sample in dataset:
            if hasattr(sample.metadata, 'custom') and sample.metadata.custom:
                custom = sample.metadata.custom

                # Collect timestamp
                ts_str = custom.get("timestamp")
                if ts_str:
                    try:
                        timestamps.append(datetime.fromisoformat(ts_str))
                    except:
                        pass

                # Collect group info
                group_id = custom.get("group_id")
                if group_id:
                    groups.add(group_id)
                    facts_per_group[group_id] += 1

        stats = {
            "total_facts": len(dataset),
            "num_groups": len(groups),
        }

        if facts_per_group:
            avg_facts_per_group = sum(facts_per_group.values()) / len(facts_per_group)
            stats["avg_facts_per_group"] = round(avg_facts_per_group, 1)

        if timestamps:
            stats["date_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "days": (max(timestamps) - min(timestamps)).days,
            }

            # Detect frequency
            if len(timestamps) > 1:
                deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                          for i in range(len(timestamps) - 1) if i < 10]  # Sample first 10
                avg_delta_hours = sum(deltas) / len(deltas) if deltas else 0

                if 0.9 <= avg_delta_hours <= 1.1:
                    stats["detected_frequency"] = "hourly"
                elif 23 <= avg_delta_hours <= 25:
                    stats["detected_frequency"] = "daily"
                elif 0.015 <= avg_delta_hours <= 0.025:  # ~1 minute
                    stats["detected_frequency"] = "minutes"
                else:
                    stats["detected_frequency"] = f"~{avg_delta_hours:.1f} hours"

        return stats
