"""Dataset validation with strict rules and task-specific validators."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import ast
from loguru import logger

from ..sources.base import Sample


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Add an error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)


class DatasetValidator:
    """Validates dataset samples with strict rules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration
        """
        config = config or {}

        self.strict_mode = config.get("strict_mode", True)
        self.required_fields = config.get("required_fields", ["instruction", "response"])
        self.min_instruction_length = config.get("min_instruction_length", 10)
        self.max_instruction_length = config.get("max_instruction_length", 5000)
        self.min_response_length = config.get("min_response_length", 20)
        self.max_response_length = config.get("max_response_length", 10000)

    def validate_sample(self, sample: Sample) -> ValidationResult:
        """
        Validate a single sample.

        Args:
            sample: Sample to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Required fields
        if not sample.instruction:
            result.add_error("Missing required field: instruction")

        if not sample.response:
            result.add_error("Missing required field: response")

        # Type validation
        if sample.instruction and not isinstance(sample.instruction, str):
            result.add_error("instruction must be string")

        if sample.response and not isinstance(sample.response, str):
            result.add_error("response must be string")

        # Length validation
        if sample.instruction:
            inst_len = len(sample.instruction)

            if inst_len < self.min_instruction_length:
                result.add_error(
                    f"instruction too short: {inst_len} chars "
                    f"(min: {self.min_instruction_length})"
                )

            if inst_len > self.max_instruction_length:
                result.add_error(
                    f"instruction too long: {inst_len} chars "
                    f"(max: {self.max_instruction_length})"
                )

        if sample.response:
            resp_len = len(sample.response)

            if resp_len < self.min_response_length:
                result.add_error(
                    f"response too short: {resp_len} chars "
                    f"(min: {self.min_response_length})"
                )

            if resp_len > self.max_response_length:
                result.add_error(
                    f"response too long: {resp_len} chars "
                    f"(max: {self.max_response_length})"
                )

        # Check for empty or whitespace-only content
        if sample.instruction and not sample.instruction.strip():
            result.add_error("instruction is empty or whitespace-only")

        if sample.response and not sample.response.strip():
            result.add_error("response is empty or whitespace-only")

        return result

    def validate_all(
        self,
        dataset: List[Sample],
        stop_on_first_error: bool = False
    ) -> Dict[str, Any]:
        """
        Validate entire dataset.

        Args:
            dataset: List of samples
            stop_on_first_error: Stop on first error

        Returns:
            Validation summary
        """
        logger.info(f"Validating {len(dataset)} samples")

        valid_count = 0
        invalid_count = 0
        all_errors = []
        all_warnings = []

        for idx, sample in enumerate(dataset):
            result = self.validate_sample(sample)

            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.extend([f"Sample {idx}: {err}" for err in result.errors])

                if stop_on_first_error:
                    break

            all_warnings.extend([f"Sample {idx}: {warn}" for warn in result.warnings])

        summary = {
            "total": len(dataset),
            "valid": valid_count,
            "invalid": invalid_count,
            "errors": all_errors,
            "warnings": all_warnings,
            "all_valid": invalid_count == 0,
        }

        logger.info(
            f"Validation complete: {valid_count}/{len(dataset)} valid, "
            f"{invalid_count} invalid"
        )

        return summary


class MathValidator:
    """Validator for math problems."""

    def validate(self, sample: Sample) -> ValidationResult:
        """Validate math problem sample."""
        result = ValidationResult(valid=True)

        response = sample.response

        # Check for common math answer patterns
        answer_patterns = ["answer:", "=", "solution:", "result:"]
        has_answer = any(pattern in response.lower() for pattern in answer_patterns)

        if not has_answer:
            result.add_warning("Response may not contain explicit answer")

        # Check for mathematical symbols
        math_symbols = ["+", "-", "*", "/", "=", "^"]
        has_math = any(symbol in response for symbol in math_symbols)

        if not has_math:
            result.add_warning("Response may not contain mathematical notation")

        return result


class CodeValidator:
    """Validator for code samples."""

    def validate(self, sample: Sample) -> ValidationResult:
        """Validate code sample."""
        result = ValidationResult(valid=True)

        response = sample.response

        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)

        if not code_blocks:
            result.add_warning("No code blocks found in response")
            return result

        # Try to parse code
        for idx, code in enumerate(code_blocks):
            # Try Python parsing (most common)
            try:
                ast.parse(code)
            except SyntaxError as e:
                result.add_warning(
                    f"Code block {idx + 1} may have syntax errors: {e}"
                )

        return result

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        code_blocks = []

        # Look for markdown code blocks
        if "```" in text:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                # Remove language identifier if present
                code = parts[i]
                if "\n" in code:
                    lines = code.split("\n", 1)
                    code = lines[1] if len(lines) > 1 else lines[0]
                code_blocks.append(code.strip())

        return code_blocks


class QAValidator:
    """Validator for Q&A samples."""

    def validate(self, sample: Sample) -> ValidationResult:
        """Validate Q&A sample."""
        result = ValidationResult(valid=True)

        instruction = sample.instruction
        response = sample.response

        # Check if instruction is a question
        if not self._is_question(instruction):
            result.add_warning("Instruction may not be a question")

        # Check if response answers the question
        # (This is a simple heuristic)
        if len(response) < len(instruction):
            result.add_warning("Response may be too short relative to question")

        return result

    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        question_words = ["what", "when", "where", "who", "why", "how", "which"]
        text_lower = text.lower()

        # Ends with question mark
        if text.strip().endswith("?"):
            return True

        # Starts with question word
        for word in question_words:
            if text_lower.startswith(word):
                return True

        return False
