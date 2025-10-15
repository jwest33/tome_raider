"""Evol-Instruct strategy for instruction evolution."""

import json
from typing import Iterator, List
from pathlib import Path
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class EvolInstructStrategy(GenerationStrategy):
    """Evol-Instruct strategy for evolving instructions."""

    EVOLUTION_TYPES = {
        "add_constraints": "Add constraints or requirements to make the task more specific",
        "deepen_reasoning": "Increase the depth of reasoning required",
        "increase_complexity": "Make the task more complex by adding steps or considerations",
        "concretize": "Make the task more concrete with specific examples or scenarios",
        "increase_breadth": "Broaden the scope to include more aspects",
    }

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        base_instructions = config.get("base_instructions")
        if isinstance(base_instructions, str):
            self.base_instructions = self._load_instructions(base_instructions)
        elif isinstance(base_instructions, list):
            self.base_instructions = base_instructions
        else:
            raise ValueError("base_instructions must be a file path or list")

        self.evolution_rounds = config.get("evolution_rounds", 3)
        self.evolution_types = config.get("evolution_types", list(self.EVOLUTION_TYPES.keys()))

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
                    for item in data:
                        if isinstance(item, str):
                            instructions.append(item)
                        elif isinstance(item, dict):
                            inst = item.get("instruction") or item.get("input")
                            if inst:
                                instructions.append(inst)
        else:
            # Plain text - one instruction per line
            with open(file_path, 'r') as f:
                instructions = [line.strip() for line in f if line.strip()]

        return instructions

    def generate(self) -> Iterator[Sample]:
        """Generate evolved instructions."""
        logger.info(f"Starting evol-instruct with {len(self.base_instructions)} base instructions")

        for idx, base_instruction in enumerate(self.base_instructions):
            logger.debug(f"Evolving instruction {idx + 1}/{len(self.base_instructions)}")

            current_instruction = base_instruction

            for round_idx in range(self.evolution_rounds):
                try:
                    # Evolve instruction
                    evolved = self._evolve_instruction(
                        current_instruction,
                        round_idx
                    )

                    if not evolved:
                        break

                    current_instruction = evolved

                except Exception as e:
                    logger.warning(f"Error evolving instruction {idx}: {e}")
                    break

            # Generate response for final evolved instruction
            try:
                response = self._generate_response(current_instruction)

                sample = self._create_sample(
                    instruction=current_instruction,
                    response=response,
                    tags=["evol_instruct"],
                    base_instruction=base_instruction,
                    evolution_rounds=self.evolution_rounds,
                )

                yield sample

            except Exception as e:
                logger.warning(f"Error generating response for instruction {idx}: {e}")

    def _evolve_instruction(self, instruction: str, round_idx: int) -> str:
        """Evolve an instruction."""
        # Select evolution type (cycle through types)
        evolution_type = self.evolution_types[round_idx % len(self.evolution_types)]
        evolution_desc = self.EVOLUTION_TYPES[evolution_type]

        prompt = f"""Evolve the following instruction by: {evolution_desc}

Original instruction:
{instruction}

Evolved instruction (keep it clear and achievable):"""

        result = self._generate_with_retry(prompt)
        return result.strip()

    def _generate_response(self, instruction: str) -> str:
        """Generate response for evolved instruction."""
        prompt = f"""Task: {instruction}

Provide a comprehensive, detailed response to this task.

Response:"""

        result = self._generate_with_retry(prompt)
        return result.strip()
