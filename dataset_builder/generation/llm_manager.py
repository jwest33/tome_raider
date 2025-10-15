"""LLM Manager for llama-server lifecycle management."""

import subprocess
import time
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger


class LlamaServerManager:
    """Manages llama-server process and API communication."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM manager.

        Args:
            config: Configuration dictionary
        """
        config = config or {}

        self.server_command = config.get("server_command", "llama-server")
        self.base_url = config.get("base_url", "http://localhost:8080")
        self.default_context_size = config.get("default_context_size", 4096)
        self.health_check_interval = config.get("health_check_interval", 5)
        self.request_timeout = config.get("request_timeout", 300)

        self.current_model = None
        self.server_process = None
        self.is_ready = False

    def load_model(
        self,
        model_path: str,
        context_size: Optional[int] = None,
        n_gpu_layers: int = 0,
        threads: Optional[int] = None,
        batch_size: int = 512,
        additional_args: Optional[List[str]] = None,
    ) -> bool:
        """
        Load a model by starting llama-server.

        Args:
            model_path: Path to model file (.gguf)
            context_size: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            threads: Number of CPU threads
            batch_size: Batch size for processing
            additional_args: Additional command line arguments

        Returns:
            True if successful

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If server fails to start
        """
        # Check if model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Stop existing server if running
        if self.server_process:
            logger.info(f"Unloading current model: {self.current_model}")
            self.unload_model()

        # Build command
        ctx_size = context_size or self.default_context_size

        cmd = [
            self.server_command,
            "-m", str(model_path),
            "-c", str(ctx_size),
            "-b", str(batch_size),
        ]

        if n_gpu_layers > 0:
            cmd.extend(["-ngl", str(n_gpu_layers)])

        if threads:
            cmd.extend(["-t", str(threads)])

        if additional_args:
            cmd.extend(additional_args)

        logger.info(f"Starting llama-server with model: {model_path}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to be ready
            self.is_ready = self._wait_for_ready()

            if self.is_ready:
                self.current_model = model_path
                logger.info(f"Model loaded successfully: {model_path}")
                return True
            else:
                logger.error("Server failed to become ready")
                self.unload_model()
                return False

        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            self.unload_model()
            raise RuntimeError(f"Failed to start llama-server: {e}")

    def unload_model(self):
        """Gracefully shutdown llama-server."""
        if self.server_process:
            logger.info("Shutting down llama-server")

            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't terminate gracefully, killing")
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                logger.error(f"Error shutting down server: {e}")

            self.server_process = None
            self.current_model = None
            self.is_ready = False

            logger.info("Server shutdown complete")

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for server to be ready.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if server is ready
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=5
                )

                if response.status_code == 200:
                    logger.debug("Server health check passed")
                    return True

            except requests.exceptions.RequestException:
                pass

            time.sleep(self.health_check_interval)

        return False

    def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if healthy
        """
        if not self.server_process or self.server_process.poll() is not None:
            return False

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text or None if failed
        """
        if not self.is_ready:
            logger.error("Server not ready")
            return None

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n_predict": max_tokens,
            "stop": stop or [],
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=self.request_timeout
            )

            response.raise_for_status()
            data = response.json()

            # Extract generated text
            content = data.get("content", "")
            return content.strip()

        except requests.exceptions.Timeout:
            logger.error("Generation request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Generation request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return None

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Generation parameters

        Returns:
            List of generated texts
        """
        results = []

        for idx, prompt in enumerate(prompts):
            logger.debug(f"Generating {idx + 1}/{len(prompts)}")

            result = self.generate(prompt, **kwargs)
            results.append(result)

            # Check server health periodically
            if (idx + 1) % 10 == 0:
                if not self.health_check():
                    logger.error("Server health check failed")
                    # Try to recover
                    if self.current_model:
                        logger.info("Attempting to restart server")
                        self.load_model(self.current_model)

        return results

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about loaded model.

        Returns:
            Model information dict or None
        """
        if not self.is_ready:
            return None

        try:
            response = requests.get(
                f"{self.base_url}/props",
                timeout=5
            )

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")

        return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.unload_model()
