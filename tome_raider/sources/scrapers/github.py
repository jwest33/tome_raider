"""GitHub issues and discussions scraper."""

from typing import Iterator, Dict, Any, Optional
from urllib.parse import urljoin
from loguru import logger

from .base_scraper import BaseScraper
from ..base import Sample


class GitHubScraper(BaseScraper):
    """Scraper for GitHub issues, discussions, and README files."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GitHub scraper.

        Args:
            config: Configuration with repositories, content_types, etc.
        """
        super().__init__(config)

        self.repositories = config.get("repositories", [])
        self.content_types = config.get("content_types", ["issues"])
        self.max_samples = config.get("max_samples", 100)
        self.github_token = config.get("github_token")

        self.api_base = "https://api.github.com"

        # Add GitHub token to session if provided
        if self.github_token:
            self.session.headers.update({
                "Authorization": f"token {self.github_token}"
            })

    def validate_config(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.repositories:
            raise ValueError("At least one repository is required")

        valid_types = {"issues", "discussions", "readme"}
        for content_type in self.content_types:
            if content_type not in valid_types:
                raise ValueError(
                    f"Invalid content type: {content_type}. "
                    f"Valid types: {valid_types}"
                )

        return True

    def _scrape(self) -> Iterator[Sample]:
        """
        Scrape GitHub content.

        Yields:
            Sample objects
        """
        logger.info(f"Scraping GitHub repositories: {self.repositories}")

        samples_collected = 0

        for repo in self.repositories:
            if samples_collected >= self.max_samples:
                break

            logger.info(f"Processing repository: {repo}")

            if "issues" in self.content_types:
                for sample in self._scrape_issues(repo):
                    if samples_collected >= self.max_samples:
                        break
                    yield sample
                    samples_collected += 1

            if "discussions" in self.content_types:
                for sample in self._scrape_discussions(repo):
                    if samples_collected >= self.max_samples:
                        break
                    yield sample
                    samples_collected += 1

            if "readme" in self.content_types:
                sample = self._scrape_readme(repo)
                if sample and samples_collected < self.max_samples:
                    yield sample
                    samples_collected += 1

        logger.info(f"Collected {samples_collected} samples from GitHub")

    def _scrape_issues(self, repo: str) -> Iterator[Sample]:
        """
        Scrape issues from repository.

        Args:
            repo: Repository in format "owner/repo"

        Yields:
            Sample objects
        """
        page = 1
        per_page = 100

        while True:
            url = f"{self.api_base}/repos/{repo}/issues"
            params = {
                "state": "closed",
                "per_page": per_page,
                "page": page,
            }

            response = self._fetch_url(f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}")

            if not response:
                break

            try:
                issues = response.json()
            except Exception as e:
                logger.error(f"Failed to parse issues response: {e}")
                break

            if not issues:
                break

            for issue in issues:
                # Skip pull requests
                if "pull_request" in issue:
                    continue

                # Must have comments (answers)
                if issue.get("comments", 0) == 0:
                    continue

                try:
                    sample = self._issue_to_sample(issue, repo)
                    if sample:
                        yield sample
                except Exception as e:
                    logger.warning(f"Error processing issue {issue.get('number')}: {e}")

            # Check for more pages
            link_header = response.headers.get("Link", "")
            if 'rel="next"' not in link_header:
                break

            page += 1

    def _issue_to_sample(self, issue: Dict[str, Any], repo: str) -> Optional[Sample]:
        """
        Convert GitHub issue to Sample.

        Args:
            issue: Issue data from API
            repo: Repository name

        Returns:
            Sample object or None
        """
        title = issue.get("title", "")
        body = issue.get("body", "")

        # Combine title and body for instruction
        instruction = f"{title}\n\n{body}".strip()

        if not instruction:
            return None

        # Fetch comments
        comments_url = issue.get("comments_url")
        if not comments_url:
            return None

        response = self._fetch_url(comments_url)
        if not response:
            return None

        try:
            comments = response.json()
        except Exception:
            return None

        if not comments:
            return None

        # Use first comment as response (or combine top comments)
        # For issues, we could use the accepted solution or top-voted comment
        top_comment = comments[0]
        response_text = top_comment.get("body", "")

        if not response_text:
            return None

        # Create metadata
        metadata = self._create_metadata(
            source_url=issue.get("html_url"),
            license=None,  # Depends on repo license
            author=issue.get("user", {}).get("login"),
            tags=[label.get("name") for label in issue.get("labels", [])],
            repository=repo,
        )

        # Create sample
        sample = Sample(
            instruction=instruction,
            response=response_text,
            metadata=metadata,
            id=f"github_{repo.replace('/', '_')}_{issue.get('number')}",
        )

        return sample

    def _scrape_discussions(self, repo: str) -> Iterator[Sample]:
        """
        Scrape discussions from repository.

        Args:
            repo: Repository in format "owner/repo"

        Yields:
            Sample objects
        """
        # GitHub Discussions require GraphQL API
        # For simplicity, we'll skip implementation here
        # In production, you would use GraphQL to fetch discussions
        logger.info(f"Discussions scraping not yet implemented for {repo}")
        return
        yield  # Make this a generator

    def _scrape_readme(self, repo: str) -> Optional[Sample]:
        """
        Scrape README from repository.

        Args:
            repo: Repository in format "owner/repo"

        Returns:
            Sample object or None
        """
        url = f"{self.api_base}/repos/{repo}/readme"

        response = self._fetch_url(url)
        if not response:
            return None

        try:
            data = response.json()
            content = data.get("content", "")

            # Decode base64 content
            import base64
            readme_text = base64.b64decode(content).decode("utf-8")

            # Create sample (README as document for instruction generation)
            # For now, we'll treat it as a sample where instruction is a prompt
            # and response is the README content
            instruction = f"Provide documentation for the {repo} repository"
            response_text = readme_text

            # Create metadata
            metadata = self._create_metadata(
                source_url=f"https://github.com/{repo}",
                license=None,
                author=repo.split("/")[0],
                tags=["readme", "documentation"],
                repository=repo,
            )

            # Create sample
            sample = Sample(
                instruction=instruction,
                response=response_text,
                metadata=metadata,
                id=f"github_{repo.replace('/', '_')}_readme",
            )

            return sample

        except Exception as e:
            logger.error(f"Error scraping README for {repo}: {e}")
            return None
