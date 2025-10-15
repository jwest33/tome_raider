"""Main generation orchestrator."""

from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from loguru import logger

from .llm_manager import LlamaServerManager
from ..sources.base import Sample, SourceMetadata


class GenerationJob:
    """A generation job with configuration and state."""

    def __init__(self, config: Dict[str, Any], llm_manager: Optional[LlamaServerManager] = None):
        """
        Initialize generation job.

        Args:
            config: Generation configuration
            llm_manager: LLM manager instance (optional)
        """
        self.config = config
        self.llm_manager = llm_manager or LlamaServerManager(config.get("llm", {}))

        self.strategy_type = config.get("strategy")
        self.model_path = config.get("model")
        self.results: List[Sample] = []
        self.errors: List[Dict[str, Any]] = []

        # Load strategy
        self.strategy = self._load_strategy()

    def _load_strategy(self):
        """Load generation strategy based on config."""
        strategy_type = self.strategy_type

        if strategy_type == "self_instruct":
            from .strategies.self_instruct import SelfInstructStrategy
            return SelfInstructStrategy(self.config, self.llm_manager)

        elif strategy_type == "evol_instruct":
            from .strategies.evol_instruct import EvolInstructStrategy
            return EvolInstructStrategy(self.config, self.llm_manager)

        elif strategy_type == "distillation":
            from .strategies.distillation import DistillationStrategy
            return DistillationStrategy(self.config, self.llm_manager)

        elif strategy_type == "response_generation":
            from .strategies.response_gen import ResponseGenerationStrategy
            return ResponseGenerationStrategy(self.config, self.llm_manager)

        elif strategy_type == "instruction_generation":
            from .strategies.instruction_gen import InstructionGenerationStrategy
            return InstructionGenerationStrategy(self.config, self.llm_manager)

        elif strategy_type == "chain_of_thought":
            from .strategies.chain_of_thought import ChainOfThoughtStrategy
            return ChainOfThoughtStrategy(self.config, self.llm_manager)

        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")

    def run(self) -> List[Sample]:
        """
        Execute generation job.

        Returns:
            List of generated samples
        """
        logger.info(f"Starting generation job with strategy: {self.strategy_type}")

        try:
            # Load model if not already loaded
            if not self.llm_manager.is_ready:
                if not self.model_path:
                    raise ValueError("Model path is required")

                self.llm_manager.load_model(
                    self.model_path,
                    context_size=self.config.get("context_size"),
                    n_gpu_layers=self.config.get("n_gpu_layers", 0),
                    threads=self.config.get("threads"),
                )

            # Run strategy
            logger.info("Executing generation strategy")
            for sample in self.strategy.generate():
                self.results.append(sample)

                # Log progress
                if len(self.results) % 10 == 0:
                    logger.info(f"Generated {len(self.results)} samples")

            logger.info(f"Generation complete. Total samples: {len(self.results)}")

            return self.results

        except Exception as e:
            logger.error(f"Generation job failed: {e}")
            self.log_error(e)
            raise

    def log_error(self, error: Exception):
        """
        Log error with context.

        Args:
            error: Exception that occurred
        """
        import traceback
        from datetime import datetime

        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": {
                "strategy": self.strategy_type,
                "model": self.model_path,
                "samples_generated": len(self.results),
            }
        }

        self.errors.append(error_entry)
        logger.error(f"Error logged: {error_entry}")


class DataGenerator:
    """Main data generator interface."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data generator.

        Args:
            config: Generator configuration
        """
        self.config = config
        self.llm_manager = LlamaServerManager(config.get("llm", {}))

    def generate(
        self,
        strategy: str,
        model: str,
        **kwargs
    ) -> List[Sample]:
        """
        Generate data using specified strategy.

        Args:
            strategy: Generation strategy name
            model: Path to model file
            **kwargs: Strategy-specific parameters

        Returns:
            List of generated samples
        """
        # Build job config
        job_config = {
            "strategy": strategy,
            "model": model,
            **kwargs,
            "llm": self.config.get("llm", {}),
        }

        # Create and run job
        job = GenerationJob(job_config, self.llm_manager)
        return job.run()

    def close(self):
        """Clean up resources."""
        if self.llm_manager:
            self.llm_manager.unload_model()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
