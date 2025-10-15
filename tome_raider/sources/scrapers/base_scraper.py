"""Abstract base class for web scrapers with rate limiting and error handling."""

import time
import requests
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from loguru import logger

from ..base import DataSource, Sample


class BaseScraper(DataSource, ABC):
    """Base class for web scrapers with rate limiting and robots.txt compliance."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scraper.

        Args:
            config: Scraper configuration
        """
        super().__init__(config)

        self.rate_limit = config.get("rate_limit", 1.0)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        self.respect_robots_txt = config.get("respect_robots_txt", True)
        self.user_agent = config.get("user_agent", "GRPO-tome-raider/0.1.0")

        self.last_request_time = 0
        self.session = self._create_session()
        self.robots_parsers: Dict[str, RobotFileParser] = {}

    def _create_session(self) -> requests.Session:
        """
        Create requests session with headers.

        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        return session

    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _check_robots_txt(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            url: URL to check

        Returns:
            True if allowed, False otherwise
        """
        if not self.respect_robots_txt:
            return True

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self.robots_parsers:
            robots_url = urljoin(base_url, "/robots.txt")
            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
                self.robots_parsers[base_url] = parser
                logger.debug(f"Loaded robots.txt from {robots_url}")
            except Exception as e:
                logger.warning(f"Failed to load robots.txt from {robots_url}: {e}")
                # Assume allowed if we can't fetch robots.txt
                self.robots_parsers[base_url] = None
                return True

        parser = self.robots_parsers[base_url]
        if parser is None:
            return True

        allowed = parser.can_fetch(self.user_agent, url)
        if not allowed:
            logger.warning(f"URL blocked by robots.txt: {url}")

        return allowed

    def _fetch_url(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[requests.Response]:
        """
        Fetch URL with rate limiting and retry logic.

        Args:
            url: URL to fetch
            method: HTTP method (GET, POST)
            data: POST data
            headers: Additional headers

        Returns:
            Response object or None if failed
        """
        if not self._check_robots_txt(url):
            return None

        request_headers = {}
        if headers:
            request_headers.update(headers)

        for attempt in range(self.max_retries):
            try:
                self._apply_rate_limit()

                if method.upper() == "GET":
                    response = self.session.get(
                        url,
                        headers=request_headers,
                        timeout=self.timeout,
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url,
                        data=data,
                        headers=request_headers,
                        timeout=self.timeout,
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                logger.debug(f"Successfully fetched: {url}")
                return response

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                if status_code == 429:  # Too Many Requests
                    wait_time = min(2 ** attempt * self.rate_limit, 60)
                    logger.warning(f"Rate limited (429), waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                elif status_code in (403, 404):
                    logger.warning(f"HTTP {status_code} for {url}, skipping")
                    return None
                elif status_code >= 500:
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error fetching {url}: {e}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    @abstractmethod
    def _scrape(self) -> Iterator[Sample]:
        """
        Scrape data from source.

        Yields:
            Sample objects

        This method should be implemented by subclasses to perform
        the actual scraping logic.
        """
        pass

    def load(self) -> Iterator[Sample]:
        """
        Load data by scraping.

        Yields:
            Sample objects
        """
        try:
            yield from self._scrape()
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources (e.g., close session)."""
        if hasattr(self, "session"):
            self.session.close()
            logger.debug("Closed scraper session")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
