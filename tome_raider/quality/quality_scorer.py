"""Quality scoring system for dataset samples."""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..sources.base import Sample


@dataclass
class QualityScore:
    """Quality score with component breakdown."""

    overall: float
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "components": self.components,
        }


class QualityScorer:
    """Calculate comprehensive quality scores for samples."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize quality scorer.

        Args:
            model_path: Optional path to ML-based quality model
        """
        self.model = None
        if model_path:
            self.model = self._load_quality_model(model_path)

    def _load_quality_model(self, model_path: str):
        """Load ML-based quality model (optional)."""
        # Placeholder for ML model loading
        logger.info(f"ML-based quality model would be loaded from: {model_path}")
        return None

    def score_sample(self, sample: Sample) -> QualityScore:
        """
        Calculate comprehensive quality score for a sample.

        Args:
            sample: Sample to score

        Returns:
            QualityScore object
        """
        components = {
            "instruction_clarity": self.score_instruction_clarity(sample),
            "instruction_complexity": self.score_instruction_complexity(sample),
            "response_completeness": self.score_response_completeness(sample),
            "response_coherence": self.score_response_coherence(sample),
            "alignment": self.score_alignment(sample),
            "diversity": self.score_diversity(sample),
        }

        # Add ML model score if available
        if self.model:
            components["model_score"] = self._predict_with_model(sample)

        # Calculate overall score (weighted average)
        overall = self._compute_overall_score(components)

        return QualityScore(overall=overall, components=components)

    def score_instruction_clarity(self, sample: Sample) -> float:
        """
        Score instruction clarity (0-1).

        Measures: grammar, readability, specificity
        """
        instruction = sample.instruction

        if not instruction:
            return 0.0

        score = 1.0

        # Check length (too short or too long reduces clarity)
        length = len(instruction)
        if length < 20:
            score *= 0.5
        elif length > 1000:
            score *= 0.8

        # Check for complete sentences (ends with punctuation)
        if not instruction.strip()[-1] in ".!?":
            score *= 0.9

        # Check for excessive capitalization
        if instruction.isupper():
            score *= 0.7

        # Check for question marks if it's a question
        question_words = ["what", "when", "where", "who", "why", "how", "which"]
        is_question = any(instruction.lower().startswith(w) for w in question_words)
        has_question_mark = "?" in instruction

        if is_question and not has_question_mark:
            score *= 0.9

        # Check for specific words (vs vague)
        vague_words = ["something", "anything", "somehow", "whatever", "stuff", "things"]
        vague_count = sum(1 for word in vague_words if word in instruction.lower())
        if vague_count > 2:
            score *= 0.8

        return max(0.0, min(1.0, score))

    def score_instruction_complexity(self, sample: Sample) -> float:
        """
        Score instruction complexity (0-1).

        Measures: vocabulary richness, concept depth
        """
        instruction = sample.instruction

        if not instruction:
            return 0.0

        score = 0.5  # Base score

        # Word count
        words = instruction.split()
        word_count = len(words)

        if word_count > 10:
            score += 0.1
        if word_count > 20:
            score += 0.1

        # Unique words (vocabulary diversity)
        unique_words = len(set(w.lower() for w in words))
        vocabulary_ratio = unique_words / max(word_count, 1)

        if vocabulary_ratio > 0.7:
            score += 0.1

        # Average word length (longer words = more complex)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        if avg_word_length > 5:
            score += 0.1
        if avg_word_length > 7:
            score += 0.1

        # Check for complex structures (multiple clauses)
        clause_indicators = [",", ";", ":", "because", "therefore", "however"]
        clause_count = sum(1 for ind in clause_indicators if ind in instruction)

        if clause_count > 2:
            score += 0.1

        return max(0.0, min(1.0, score))

    def score_response_completeness(self, sample: Sample) -> float:
        """
        Score response completeness (0-1).

        Measures: addresses all aspects of instruction
        """
        instruction = sample.instruction
        response = sample.response

        if not instruction or not response:
            return 0.0

        score = 0.5  # Base score

        # Length ratio (response should be substantial)
        inst_words = len(instruction.split())
        resp_words = len(response.split())

        # Expect response to be at least as long as instruction
        length_ratio = resp_words / max(inst_words, 1)

        if length_ratio >= 1.0:
            score += 0.2
        if length_ratio >= 2.0:
            score += 0.1

        # Check for structure (paragraphs, lists, etc.)
        has_paragraphs = "\n\n" in response
        has_lists = any(marker in response for marker in ["1.", "2.", "-", "*"])
        has_code = "```" in response or "    " in response

        if has_paragraphs:
            score += 0.1
        if has_lists:
            score += 0.05
        if has_code:
            score += 0.05

        # Check for conclusion/summary indicators
        conclusion_words = ["therefore", "thus", "in conclusion", "summary", "finally"]
        has_conclusion = any(word in response.lower() for word in conclusion_words)

        if has_conclusion:
            score += 0.1

        return max(0.0, min(1.0, score))

    def score_response_coherence(self, sample: Sample) -> float:
        """
        Score response coherence (0-1).

        Measures: logical flow, structure
        """
        response = sample.response

        if not response:
            return 0.0

        score = 0.6  # Base score

        # Check for transition words (indicates logical flow)
        transitions = [
            "first", "second", "next", "then", "finally",
            "however", "therefore", "thus", "additionally",
            "furthermore", "moreover", "consequently"
        ]

        transition_count = sum(1 for t in transitions if t in response.lower())

        if transition_count >= 2:
            score += 0.1
        if transition_count >= 4:
            score += 0.1

        # Check for consistent formatting
        sentences = response.split(".")
        if len(sentences) > 1:
            # Check if most sentences start with capital letters
            capital_starts = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            consistency = capital_starts / len(sentences)

            if consistency > 0.8:
                score += 0.1

        # Check for excessive repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)

            if unique_ratio < 0.5:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def score_alignment(self, sample: Sample) -> float:
        """
        Score alignment between instruction and response (0-1).

        Measures: response matches instruction intent
        """
        instruction = sample.instruction.lower()
        response = sample.response.lower()

        if not instruction or not response:
            return 0.0

        score = 0.5  # Base score

        # Extract key words from instruction (nouns, verbs)
        inst_words = set(re.findall(r'\b\w+\b', instruction))

        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        inst_keywords = inst_words - stop_words

        # Check how many instruction keywords appear in response
        if inst_keywords:
            keyword_overlap = sum(1 for kw in inst_keywords if kw in response)
            overlap_ratio = keyword_overlap / len(inst_keywords)

            score += overlap_ratio * 0.4

        # Check for task-specific alignment
        if "write" in instruction and "code" in instruction:
            if "def " in response or "function" in response or "```" in response:
                score += 0.1

        if "explain" in instruction:
            if "because" in response or "reason" in response:
                score += 0.1

        if "list" in instruction or "steps" in instruction:
            if any(marker in response for marker in ["1.", "2.", "3.", "-", "*"]):
                score += 0.1

        return max(0.0, min(1.0, score))

    def score_diversity(self, sample: Sample) -> float:
        """
        Score vocabulary diversity (0-1).

        Measures: vocabulary variety, n-gram uniqueness
        """
        text = sample.instruction + " " + sample.response

        if not text:
            return 0.0

        words = text.lower().split()

        if len(words) < 10:
            return 0.5

        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)

        # Penalize very low diversity
        if unique_ratio < 0.4:
            return 0.3

        # Reward high diversity
        score = min(1.0, unique_ratio * 1.5)

        return score

    def _predict_with_model(self, sample: Sample) -> float:
        """Predict quality using ML model (if loaded)."""
        if not self.model:
            return 0.5

        # Placeholder for ML model prediction
        return 0.5

    def _compute_overall_score(self, components: Dict[str, float]) -> float:
        """
        Compute overall score from components.

        Uses weighted average with higher weight on critical components.
        """
        weights = {
            "instruction_clarity": 0.20,
            "instruction_complexity": 0.15,
            "response_completeness": 0.25,
            "response_coherence": 0.20,
            "alignment": 0.15,
            "diversity": 0.05,
            "model_score": 0.0,  # Optional, not counted if not present
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for component, score in components.items():
            weight = weights.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def recommend_thresholds(
        self,
        dataset: List[Sample],
        scores: Optional[List[QualityScore]] = None
    ) -> Dict[str, float]:
        """
        Analyze dataset and recommend quality thresholds.

        Args:
            dataset: Dataset to analyze
            scores: Pre-computed scores (optional)

        Returns:
            Dictionary of recommended thresholds
        """
        if not scores:
            logger.info("Computing quality scores for threshold recommendation")
            scores = [self.score_sample(sample) for sample in dataset]

        overall_scores = [s.overall for s in scores]

        if not overall_scores:
            return {
                "strict": 0.8,
                "moderate": 0.6,
                "lenient": 0.4,
            }

        mean_score = np.mean(overall_scores)
        std_score = np.std(overall_scores)

        recommendations = {
            "strict": min(0.95, mean_score + std_score),       # Top ~16%
            "moderate": mean_score,                            # Top ~50%
            "lenient": max(0.3, mean_score - std_score),      # Top ~84%
            "statistics": {
                "mean": float(mean_score),
                "std": float(std_score),
                "min": float(np.min(overall_scores)),
                "max": float(np.max(overall_scores)),
                "median": float(np.median(overall_scores)),
            }
        }

        logger.info(f"Threshold recommendations: {recommendations}")

        return recommendations
