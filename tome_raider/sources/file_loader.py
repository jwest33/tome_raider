"""File loader for JSON, JSONL, CSV, Parquet, TXT, and PDF files."""

import json
import csv
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import pandas as pd
from loguru import logger

from .base import DataSource, Sample, SourceMetadata


class FileLoader(DataSource):
    """Load data from various file formats."""

    SUPPORTED_FORMATS = {
        ".json",
        ".jsonl",
        ".csv",
        ".parquet",
        ".txt",
        ".pdf",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file loader.

        Args:
            config: Configuration with 'paths' or 'path' key
        """
        super().__init__(config)

        # Support both 'path' and 'paths' keys
        if "path" in config:
            self.paths = [config["path"]]
        elif "paths" in config:
            paths = config["paths"]
            self.paths = paths if isinstance(paths, list) else [paths]
        else:
            raise ValueError("Config must contain 'path' or 'paths' key")

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
        if not self.paths:
            raise ValueError("No file paths provided")

        # Expand glob patterns and validate files
        expanded_paths = []
        for path_pattern in self.paths:
            path_obj = Path(path_pattern)

            if "*" in path_pattern or "?" in path_pattern:
                # Handle glob patterns
                parent = path_obj.parent
                pattern = path_obj.name
                matches = list(parent.glob(pattern))

                if not matches:
                    logger.warning(f"No files match pattern: {path_pattern}")
                else:
                    expanded_paths.extend(matches)
            else:
                # Regular file path
                if not path_obj.exists():
                    raise FileNotFoundError(f"File not found: {path_pattern}")

                expanded_paths.append(path_obj)

        self.expanded_paths = expanded_paths

        if not self.expanded_paths:
            raise ValueError("No valid files found")

        # Validate file formats
        for path in self.expanded_paths:
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                logger.warning(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                )

        return True

    def load(self) -> Iterator[Sample]:
        """
        Load data from all files.

        Yields:
            Sample objects
        """
        for file_path in self.expanded_paths:
            logger.info(f"Loading file: {file_path}")

            try:
                yield from self._load_file(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

    def _load_file(self, file_path: Path) -> Iterator[Sample]:
        """
        Load single file based on format.

        Args:
            file_path: Path to file

        Yields:
            Sample objects
        """
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            yield from self._load_json(file_path)
        elif suffix == ".jsonl":
            yield from self._load_jsonl(file_path)
        elif suffix == ".csv":
            yield from self._load_csv(file_path)
        elif suffix == ".parquet":
            yield from self._load_parquet(file_path)
        elif suffix == ".txt":
            yield from self._load_txt(file_path)
        elif suffix == ".pdf":
            yield from self._load_pdf(file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")

    def _load_json(self, file_path: Path) -> Iterator[Sample]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Check for common keys that contain the data array
            if "data" in data:
                items = data["data"]
            elif "samples" in data:
                items = data["samples"]
            elif "examples" in data:
                items = data["examples"]
            else:
                # Treat single dict as single item
                items = [data]
        else:
            logger.warning(f"Unexpected JSON structure in {file_path}")
            return

        for idx, item in enumerate(items):
            try:
                sample = self._dict_to_sample(item, file_path, idx + 1)
                if sample:
                    yield sample
            except Exception as e:
                logger.warning(f"Error parsing item {idx + 1} in {file_path}: {e}")

    def _load_jsonl(self, file_path: Path) -> Iterator[Sample]:
        """Load JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = self._dict_to_sample(data, file_path, line_num)
                    if sample:
                        yield sample
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")

    def _load_csv(self, file_path: Path) -> Iterator[Sample]:
        """Load CSV file."""
        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            try:
                data = row.to_dict()
                sample = self._dict_to_sample(data, file_path, idx + 2)  # +2 for header
                if sample:
                    yield sample
            except Exception as e:
                logger.warning(f"Error parsing row {idx + 2} in {file_path}: {e}")

    def _load_parquet(self, file_path: Path) -> Iterator[Sample]:
        """Load Parquet file."""
        df = pd.read_parquet(file_path)

        for idx, row in df.iterrows():
            try:
                data = row.to_dict()
                sample = self._dict_to_sample(data, file_path, idx + 1)
                if sample:
                    yield sample
            except Exception as e:
                logger.warning(f"Error parsing row {idx + 1} in {file_path}: {e}")

    def _load_txt(self, file_path: Path) -> Iterator[Sample]:
        """
        Load plain text file.

        Assumes format:
        === INSTRUCTION ===
        instruction text
        === RESPONSE ===
        response text
        === END ===

        Or just plain paragraphs separated by blank lines.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try structured format first
        if "=== INSTRUCTION ===" in content and "=== RESPONSE ===" in content:
            samples = content.split("=== END ===")
            for idx, sample_text in enumerate(samples, 1):
                if not sample_text.strip():
                    continue

                parts = sample_text.split("=== RESPONSE ===")
                if len(parts) != 2:
                    continue

                instruction = parts[0].replace("=== INSTRUCTION ===", "").strip()
                response = parts[1].strip()

                if instruction and response:
                    metadata = self._create_metadata(
                        source_file=str(file_path),
                        source_line=None,
                    )
                    yield Sample(
                        instruction=instruction,
                        response=response,
                        metadata=metadata,
                    )
        else:
            # Plain text - treat as document to generate instructions from
            logger.info(f"Plain text file detected, treating as document: {file_path}")
            # For now, just skip - this would be handled by instruction generation

    def _load_pdf(self, file_path: Path) -> Iterator[Sample]:
        """Load PDF file."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
            full_text = ""

            for page in reader.pages:
                full_text += page.extract_text() + "\n\n"

            # Similar to txt - treat as document
            logger.info(
                f"PDF loaded with {len(reader.pages)} pages. "
                f"Treat as document for instruction generation."
            )

            # For now, skip - would be handled by instruction generation

        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")

    def _dict_to_sample(
        self,
        data: Dict[str, Any],
        file_path: Path,
        line_num: int
    ) -> Optional[Sample]:
        """
        Convert dictionary to Sample object.

        Args:
            data: Dictionary with sample data
            file_path: Source file path
            line_num: Line number in file

        Returns:
            Sample object or None if invalid
        """
        # Extract instruction and response with field mapping
        instruction = data.get(self.instruction_field)
        response = data.get(self.response_field)

        if not instruction or not response:
            # Try common alternative field names
            instruction = instruction or data.get("input") or data.get("question") or data.get("query")
            response = response or data.get("output") or data.get("answer") or data.get("completion")

        if not instruction or not response:
            logger.debug(f"Missing instruction or response in line {line_num}")
            return None

        # Check if metadata already exists in the data
        existing_metadata = data.get("metadata", {})

        # Create metadata
        metadata = self._create_metadata(
            source_file=str(file_path),
            source_line=line_num,
            tags=existing_metadata.get("tags", []),
            **{k: v for k, v in existing_metadata.items() if k != "tags"}
        )

        # Create sample
        sample = Sample(
            instruction=str(instruction),
            response=str(response),
            metadata=metadata,
            id=data.get("id"),
        )

        return sample

    def count(self) -> Optional[int]:
        """
        Count total samples.

        Returns:
            Number of samples or None
        """
        try:
            total = 0
            for sample in self.load():
                total += 1
            return total
        except Exception:
            return None
