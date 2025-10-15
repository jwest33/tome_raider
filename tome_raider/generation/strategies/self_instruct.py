"""Self-Instruct generation strategy."""

from typing import Iterator, List
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class SelfInstructStrategy(GenerationStrategy):
    """Self-Instruct strategy for generating instruction-response pairs."""

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        self.seed_tasks = config.get("seed_tasks", [])
        self.generations_per_seed = config.get("generations_per_seed", 5)
        self.target_count = config.get("target_count", 100)

        # Quality filters
        self.min_instruction_length = config.get("min_instruction_length", 20)
        self.min_response_length = config.get("min_response_length", 50)

    def generate(self) -> Iterator[Sample]:
        """Generate samples using self-instruct."""
        logger.info("Starting self-instruct generation")

        generated_count = 0

        # If no seed tasks provided, use default
        if not self.seed_tasks:
            self.seed_tasks = self._get_default_seeds()

        for seed_idx, seed in enumerate(self.seed_tasks):
            if generated_count >= self.target_count:
                break

            logger.info(f"Processing seed {seed_idx + 1}/{len(self.seed_tasks)}")

            for gen_idx in range(self.generations_per_seed):
                if generated_count >= self.target_count:
                    break

                try:
                    # Generate new instruction
                    new_instruction = self._generate_instruction(seed, gen_idx)

                    if not new_instruction or len(new_instruction) < self.min_instruction_length:
                        continue

                    # Generate response
                    response = self._generate_response(new_instruction)

                    if not response or len(response) < self.min_response_length:
                        continue

                    # Create sample
                    sample = self._create_sample(
                        instruction=new_instruction,
                        response=response,
                        tags=["self_instruct"],
                        seed_task=seed,
                        generation_index=gen_idx,
                    )

                    yield sample
                    generated_count += 1

                except Exception as e:
                    logger.warning(f"Error generating from seed {seed_idx}: {e}")

        logger.info(f"Self-instruct generation complete: {generated_count} samples")

    def _get_default_seeds(self) -> List[str]:
        """Get default seed tasks."""
        return [
            "Write a function to calculate the factorial of a number",
            "Explain the concept of object-oriented programming",
            "Describe how to make a basic web server",
            "Write a poem about nature",
            "Explain the Pythagorean theorem",
            "Describe the water cycle",
            "Write a tutorial on binary search",
            "Explain quantum computing in simple terms",
            "Describe the process of photosynthesis",
            "Write a guide to basic data structures",
        ]

    def _generate_instruction(self, seed: str, variation: int) -> str:
        """Generate new instruction based on seed."""
        prompt = f"""Generate a new task instruction similar to but different from this example:

Example: {seed}

Create a new, unique task that:
1. Is related to the same domain or topic
2. Has a different specific goal or focus
3. Is clear and specific
4. Can be completed with a detailed response

New task instruction:"""

        result = self._generate_with_retry(prompt)
        return result.strip()

    def _generate_response(self, instruction: str) -> str:
        """Generate response for instruction."""
        prompt = f"""Task: {instruction}

Provide a detailed, high-quality response to this task. Be thorough, accurate, and helpful.

Response:"""

        result = self._generate_with_retry(prompt)
        return result.strip()
