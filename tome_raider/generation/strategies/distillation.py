"""Distillation strategy for teacher-student training."""

import json
from typing import Iterator, List
from pathlib import Path
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class DistillationStrategy(GenerationStrategy):
    """Distillation strategy using a teacher model."""

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        prompts = config.get("prompts")
        if isinstance(prompts, str):
            self.prompts = self._load_prompts(prompts)
        elif isinstance(prompts, list):
            self.prompts = prompts
        else:
            raise ValueError("prompts must be a file path or list")

        self.batch_size = config.get("batch_size", 8)
        self.student_format = config.get("student_format", True)

    def _load_prompts(self, path: str) -> List[str]:
        """Load prompts from file."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        prompts = []

        if file_path.suffix == ".jsonl":
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    prompt = data.get("prompt") or data.get("instruction") or data.get("text")
                    if prompt:
                        prompts.append(prompt)
        elif file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = [str(item) if not isinstance(item, dict) else item.get("prompt", "") for item in data]
        else:
            with open(file_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]

        return prompts

    def generate(self) -> Iterator[Sample]:
        """Generate samples using teacher model."""
        logger.info(f"Starting distillation with {len(self.prompts)} prompts")

        # Process in batches
        for i in range(0, len(self.prompts), self.batch_size):
            batch = self.prompts[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}")

            for prompt in batch:
                try:
                    # Generate response from teacher model
                    response = self._generate_with_retry(prompt)

                    if not response:
                        continue

                    # Create sample
                    sample = self._create_sample(
                        instruction=prompt,
                        response=response,
                        tags=["distillation"],
                        teacher_model=self.llm.current_model,
                    )

                    yield sample

                except Exception as e:
                    logger.warning(f"Error generating for prompt: {e}")
