"""Stack Overflow Q&A scraper."""

from typing import Iterator, Dict, Any, List, Optional
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from loguru import logger

from .base_scraper import BaseScraper
from ..base import Sample


class StackOverflowScraper(BaseScraper):
    """Scraper for Stack Overflow questions and answers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Stack Overflow scraper.

        Args:
            config: Configuration with tags, max_samples, etc.
        """
        super().__init__(config)

        self.tags = config.get("tags", [])
        self.max_samples = config.get("max_samples", 100)
        self.min_score = config.get("min_score", 5)
        self.accepted_only = config.get("accepted_only", True)

        self.api_base = "https://api.stackexchange.com/2.3"

    def validate_config(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.tags:
            logger.warning("No tags specified, scraping may return random questions")

        return True

    def _scrape(self) -> Iterator[Sample]:
        """
        Scrape Stack Overflow Q&A pairs.

        Yields:
            Sample objects
        """
        logger.info(f"Scraping Stack Overflow with tags: {self.tags}")

        samples_collected = 0
        page = 1
        page_size = 100

        while samples_collected < self.max_samples:
            # Use Stack Exchange API
            params = {
                "page": page,
                "pagesize": min(page_size, self.max_samples - samples_collected),
                "order": "desc",
                "sort": "votes",
                "site": "stackoverflow",
                "filter": "withbody",
                "tagged": ";".join(self.tags) if self.tags else None,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            url = f"{self.api_base}/questions?{urlencode(params)}"

            logger.debug(f"Fetching page {page}")
            response = self._fetch_url(url)

            if not response:
                logger.warning(f"Failed to fetch page {page}, stopping")
                break

            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                break

            questions = data.get("items", [])

            if not questions:
                logger.info("No more questions available")
                break

            # Check if we've hit rate limit
            if "backoff" in data:
                backoff = data["backoff"]
                logger.warning(f"API backoff requested: {backoff}s")
                import time
                time.sleep(backoff)

            for question in questions:
                if samples_collected >= self.max_samples:
                    break

                # Filter by score
                if question.get("score", 0) < self.min_score:
                    continue

                # Get accepted answer
                if self.accepted_only and "accepted_answer_id" not in question:
                    continue

                try:
                    sample = self._question_to_sample(question)
                    if sample:
                        yield sample
                        samples_collected += 1
                except Exception as e:
                    logger.warning(f"Error processing question {question.get('question_id')}: {e}")

            # Check quota
            quota_remaining = data.get("quota_remaining", 0)
            if quota_remaining < 100:
                logger.warning(f"API quota low: {quota_remaining} remaining")

            if not data.get("has_more", False):
                break

            page += 1

        logger.info(f"Collected {samples_collected} samples from Stack Overflow")

    def _question_to_sample(self, question: Dict[str, Any]) -> Optional[Sample]:
        """
        Convert Stack Overflow question to Sample.

        Args:
            question: Question data from API

        Returns:
            Sample object or None
        """
        question_id = question.get("question_id")
        title = question.get("title", "")
        body = question.get("body", "")

        # Clean HTML from body
        body_text = self._clean_html(body)

        # Combine title and body for instruction
        instruction = f"{title}\n\n{body_text}".strip()

        # Get accepted answer
        accepted_answer_id = question.get("accepted_answer_id")

        if not accepted_answer_id:
            return None

        # Fetch answer
        answer_url = f"{self.api_base}/answers/{accepted_answer_id}?site=stackoverflow&filter=withbody"
        answer_response = self._fetch_url(answer_url)

        if not answer_response:
            return None

        try:
            answer_data = answer_response.json()
            answers = answer_data.get("items", [])

            if not answers:
                return None

            answer = answers[0]
            answer_body = answer.get("body", "")
            answer_text = self._clean_html(answer_body)

        except Exception as e:
            logger.warning(f"Error fetching answer for question {question_id}: {e}")
            return None

        # Create metadata
        metadata = self._create_metadata(
            source_url=question.get("link"),
            license="CC BY-SA",
            author=question.get("owner", {}).get("display_name"),
            tags=question.get("tags", []),
            score=question.get("score"),
            answer_score=answer.get("score"),
        )

        # Create sample
        sample = Sample(
            instruction=instruction,
            response=answer_text,
            metadata=metadata,
            id=f"stackoverflow_{question_id}",
        )

        return sample

    def _clean_html(self, html: str) -> str:
        """
        Clean HTML to plain text.

        Args:
            html: HTML string

        Returns:
            Plain text
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Handle code blocks specially
            for code in soup.find_all("code"):
                code.string = f"\n```\n{code.get_text()}\n```\n"

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)

            return text

        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
            return html
