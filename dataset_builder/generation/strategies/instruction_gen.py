"""Instruction generation from documents strategy."""

import json
from typing import Iterator, List
from pathlib import Path
from loguru import logger

from .base_strategy import GenerationStrategy
from ...sources.base import Sample


class InstructionGenerationStrategy(GenerationStrategy):
    """Generate instructions from documents."""

    def __init__(self, config, llm_manager):
        super().__init__(config, llm_manager)

        documents_path = config.get("documents")
        if not documents_path:
            raise ValueError("documents path is required")

        self.documents = self._load_documents(documents_path)
        self.questions_per_document = config.get("questions_per_document", 5)
        self.question_types = config.get("question_types", ["factual", "analytical"])

    def _load_documents(self, path: str) -> List[str]:
        """Load documents from path."""
        documents = []
        path_obj = Path(path)

        if path_obj.is_file():
            # Single file
            with open(path_obj, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        elif path_obj.is_dir():
            # Directory of files
            for file_path in path_obj.glob("*"):
                if file_path.is_file() and file_path.suffix in [".txt", ".md"]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents.append(f.read())

        return documents

    def generate(self) -> Iterator[Sample]:
        """Generate instruction-response pairs from documents."""
        logger.info(f"Generating instructions from {len(self.documents)} documents")

        for doc_idx, document in enumerate(self.documents):
            logger.info(f"Processing document {doc_idx + 1}/{len(self.documents)}")

            # Generate questions for this document
            for q_idx in range(self.questions_per_document):
                try:
                    question_type = self.question_types[q_idx % len(self.question_types)]

                    # Generate question
                    instruction = self._generate_question(document, question_type)

                    if not instruction:
                        continue

                    # Generate answer from document
                    response = self._generate_answer(document, instruction)

                    if not response:
                        continue

                    sample = self._create_sample(
                        instruction=instruction,
                        response=response,
                        tags=["instruction_generation", question_type],
                        document_index=doc_idx,
                    )

                    yield sample

                except Exception as e:
                    logger.warning(f"Error generating from document {doc_idx}: {e}")

    def _generate_question(self, document: str, question_type: str) -> str:
        """Generate a question from document."""
        # Truncate document if too long
        doc_sample = document[:2000] if len(document) > 2000 else document

        prompt = f"""Based on the following document, generate a {question_type} question that can be answered using the information in the document.

Document:
{doc_sample}

Generate a clear, specific {question_type} question:"""

        result = self._generate_with_retry(prompt)
        return result.strip()

    def _generate_answer(self, document: str, question: str) -> str:
        """Generate answer to question from document."""
        doc_sample = document[:2000] if len(document) > 2000 else document

        prompt = f"""Document:
{doc_sample}

Question: {question}

Answer this question based on the information in the document. Provide a detailed, accurate answer.

Answer:"""

        result = self._generate_with_retry(prompt)
        return result.strip()
