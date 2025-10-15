"""Response generation strategy."""

import json
from typing import Iterator, List
from pathlib import Path
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class ResponseGenerationStrategy(GenerationStrategy):
    """Generate responses for existing instructions."""

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        instructions = config.get("instructions")
        if isinstance(instructions, str):
            self.instructions = self._load_instructions(instructions)
        elif isinstance(instructions, list):
            self.instructions = instructions
        else:
            raise ValueError("instructions must be a file path or list")

    def _load_instructions(self, path: str) -> List[str]:
        """Load instructions from file."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Instructions file not found: {path}")

        instructions = []

        if file_path.suffix == ".jsonl":
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    inst = data.get("instruction") or data.get("input") or data.get("text")
                    if inst:
                        instructions.append(inst)
        elif file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    instructions = [str(item) if not isinstance(item, dict) else item.get("instruction", "") for item in data]
        else:
            with open(file_path, 'r') as f:
                instructions = [line.strip() for line in f if line.strip()]

        return instructions

    def generate(self) -> Iterator[Sample]:
        """Generate responses for instructions."""
        logger.info(f"Generating responses for {len(self.instructions)} instructions")

        for idx, instruction in enumerate(self.instructions):
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(self.instructions)}")

            try:
                response = self._generate_with_retry(instruction)

                if not response:
                    continue

                sample = self._create_sample(
                    instruction=instruction,
                    response=response,
                    tags=["response_generation"],
                )

                yield sample

            except Exception as e:
                logger.warning(f"Error generating response for instruction {idx}: {e}")
