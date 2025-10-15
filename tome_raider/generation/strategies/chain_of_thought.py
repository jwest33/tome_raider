"""Chain-of-Thought synthesis strategy."""

import json
from typing import Iterator, List
from pathlib import Path
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class ChainOfThoughtStrategy(GenerationStrategy):
    """Generate responses with chain-of-thought reasoning."""

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        problems = config.get("problems")
        if isinstance(problems, str):
            self.problems = self._load_problems(problems)
        elif isinstance(problems, list):
            self.problems = problems
        else:
            raise ValueError("problems must be a file path or list")

        # CoT markers
        cot_markers = config.get("cot_markers", {})
        self.start_marker = cot_markers.get("start", "<start_working_out>")
        self.end_marker = cot_markers.get("end", "<end_working_out>")
        self.solution_start = cot_markers.get("solution_start", "<SOLUTION>")
        self.solution_end = cot_markers.get("solution_end", "</SOLUTION>")

    def _load_problems(self, path: str) -> List[str]:
        """Load problems from file."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Problems file not found: {path}")

        problems = []

        if file_path.suffix == ".jsonl":
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    problem = data.get("problem") or data.get("instruction") or data.get("text")
                    if problem:
                        problems.append(problem)
        elif file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    problems = [str(item) if not isinstance(item, dict) else item.get("problem", "") for item in data]
        else:
            with open(file_path, 'r') as f:
                problems = [line.strip() for line in f if line.strip()]

        return problems

    def generate(self) -> Iterator[Sample]:
        """Generate chain-of-thought samples."""
        logger.info(f"Generating CoT samples for {len(self.problems)} problems")

        for idx, problem in enumerate(self.problems):
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(self.problems)}")

            try:
                # Generate reasoning and solution
                reasoning, solution = self._generate_cot_response(problem)

                if not reasoning or not solution:
                    continue

                # Format response with CoT markers
                response = self._format_cot_response(reasoning, solution)

                sample = self._create_sample(
                    instruction=problem,
                    response=response,
                    tags=["chain_of_thought"],
                )

                yield sample

            except Exception as e:
                logger.warning(f"Error generating CoT for problem {idx}: {e}")

    def _generate_cot_response(self, problem: str) -> tuple:
        """Generate reasoning and solution."""
        prompt = f"""Problem: {problem}

Solve this problem step by step. Show your reasoning process clearly, then provide the final solution.

Step-by-step reasoning:"""

        # Generate full response
        full_response = self._generate_with_retry(prompt, max_tokens=2048)

        if not full_response:
            return None, None

        # Try to split reasoning and solution
        # Look for common solution indicators
        solution_indicators = [
            "Final answer:",
            "Therefore,",
            "The solution is",
            "Answer:",
            "Result:",
        ]

        reasoning = full_response
        solution = ""

        for indicator in solution_indicators:
            if indicator in full_response:
                parts = full_response.split(indicator, 1)
                reasoning = parts[0].strip()
                solution = (indicator + parts[1]).strip()
                break

        # If no clear split, use last paragraph as solution
        if not solution:
            paragraphs = full_response.strip().split("\n\n")
            if len(paragraphs) > 1:
                solution = paragraphs[-1]
                reasoning = "\n\n".join(paragraphs[:-1])
            else:
                # Fallback: treat everything as reasoning
                solution = "The solution follows from the reasoning above."

        return reasoning, solution

    def _format_cot_response(self, reasoning: str, solution: str) -> str:
        """Format response with CoT markers."""
        return f"""{self.start_marker}
{reasoning}
{self.end_marker}

{self.solution_start}
{solution}
{self.solution_end}"""
