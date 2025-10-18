"""Temporal fact generator using LLM for embedding temporal retrieval experiments."""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .llm_manager import LlamaServerManager


@dataclass
class TemporalFactConfig:
    """Configuration for temporal fact generation."""

    # Generation parameters
    num_fact_groups: int = 20
    variations_per_group: int = 7

    # Temporal parameters
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-31"
    frequency: str = "hourly"  # hourly, daily, minutes

    # LLM parameters
    model_path: str = None
    temperature: float = 0.8
    max_tokens: int = 150

    # Retry parameters
    max_retries: int = 3
    retry_delay: float = 2.0
    use_exponential_backoff: bool = True

    # Review parameters
    enable_review: bool = True
    review_temperature: float = 0.3  # Lower temperature for more consistent review
    auto_correct: bool = True  # Automatically correct issues found during review

    # Content parameters
    fact_domain: str = "news"  # news, science, business, sports, social, entertainment, technology
    seed: Optional[int] = None

    # Evolution mode - generate connected story events instead of semantic variations
    evolution_mode: bool = False


DOMAIN_PROMPTS = {
    "news": """Generate ONE newsworthy factual event with specific numbers.

Example: "The Dow Jones Industrial Average closed at 34,500 points on March 15th, up 250 points from the previous day."

Your fact (be specific with numbers):""",

    "business": """Generate ONE business fact with specific numbers.

Example: "Apple Inc. reported quarterly revenue of $89.5 billion, exceeding analyst expectations by 12%."

Your fact (include company name and numbers):""",

    "science": """Generate ONE scientific measurement or observation.

Example: "The James Webb Space Telescope detected a galaxy 13.2 billion light-years away, setting a new distance record."

Your fact (include specific measurements):""",

    "sports": """Generate ONE sports fact with specific statistics.

Example: "LeBron James scored 45 points in last night's game against the Boston Celtics, bringing his career total to 38,652 points."

Your fact (include player/team names and numbers):""",

    "social": """Generate ONE human social event or gathering with specific details.

Example: "The annual neighborhood block party on Maple Street attracted 127 residents, including 35 children, who participated in activities from 2:00 PM to 8:00 PM."

Your fact (include specific numbers, location, and attendees):""",

    "entertainment": """Generate ONE entertainment industry fact with specific figures.

Example: "The movie 'Cosmic Adventure' earned $127 million in its opening weekend, setting a new box office record for sci-fi films."

Your fact (include titles and specific numbers):""",

    "technology": """Generate ONE technology news fact with specific data.

Example: "OpenAI's GPT-5 model achieved 94.2% accuracy on the MMLU benchmark, surpassing the previous record by 8 percentage points."

Your fact (include product/company name and metrics):""",
}


# Evolution mode prompts - generate connected story events
EVOLUTION_SEED_PROMPTS = {
    "news": """Generate the FIRST event in a news story. This should be an initial newsworthy event that can naturally lead to follow-up developments.

Example: "Breaking news reported that a major earthquake with magnitude 6.8 struck the Pacific coast at 3:45 AM local time."

Your seed event (be specific with details):""",

    "business": """Generate the FIRST event in a business story. This should be an initial business development that can naturally lead to consequences and reactions.

Example: "TechCorp announced a surprise acquisition of StartupAI for $2.3 billion in an early morning press release."

Your seed event (include company names and specific details):""",

    "science": """Generate the FIRST event in a scientific discovery story. This should be an initial observation or finding that can lead to further analysis and implications.

Example: "Researchers at the Lunar Observatory detected an unusual radio signal originating from the Andromeda galaxy at precisely 11:23 PM GMT."

Your seed event (include specific measurements and details):""",

    "sports": """Generate the FIRST event in a sports story. This should be an initial game moment or announcement that can lead to reactions and consequences.

Example: "Star quarterback Marcus Williams suffered a knee injury in the first quarter of Sunday's playoff game against the Ravens."

Your seed event (include player/team names and specific details):""",

    "social": """Generate the FIRST event in a human social story. This should be an initial gathering, meeting, or social interaction that can lead to reactions and further developments.

Example: "At 7:15 AM, community organizer Maria Rodriguez set up tables and chairs in the community center lobby for the first-ever neighborhood breakfast meeting, expecting 20-30 residents."

Your seed event (include specific details, location, and people):""",

    "entertainment": """Generate the FIRST event in an entertainment story. This should be an initial release, announcement, or event that can lead to reactions and box office/streaming developments.

Example: "The blockbuster film 'Stellar Odyssey' premiered simultaneously in 4,200 theaters worldwide at midnight on Friday."

Your seed event (include titles and specific details):""",

    "technology": """Generate the FIRST event in a technology story. This should be an initial product launch or technical achievement that can lead to reactions and market developments.

Example: "QuantumTech unveiled their first consumer quantum processor, the Q-Core 1000, at a press event in San Francisco at 10:00 AM PST."

Your seed event (include product/company names and specific details):""",
}


EVOLUTION_OUTLINE_PROMPT = """Create a {num_events}-event story outline for the {domain} domain. Write {num_events} bullet points, each describing ONE specific event in sequence.

Requirements:
- Each bullet should be 50-80 characters (concise but specific)
- Events should be connected and build on each other
- Include specific details (names, numbers, actions)
- Show clear progression over time
- NO meta-commentary or labels (like "Event 1:" or "Hour 2:")

Example for social domain (5 events):
1. Maria sets up breakfast meeting at community center, expecting 20-30 residents
2. 47 residents arrive, share concerns about local park maintenance
3. Volunteers form committee, draft proposal for park improvements
4. Committee presents 12-page proposal to city council meeting
5. Council approves $15,000 budget for park renovation project

Now write {num_events} connected events for {domain} domain:"""

EVOLUTION_EXPAND_BULLET_PROMPT = """Expand this story event into a detailed description (target: 600-1100 characters).

Event to expand: {bullet}

Context - Previous events in this story:
{previous_context}

Requirements:
- Expand into 600-1100 characters (aim for rich detail around 800-1000 chars)
- Include specific details, actions, dialogue, and character thoughts/reactions
- Show connection to previous events (use "after", "following", "then")
- Keep it as a single coherent narrative moment or scene
- NO meta-commentary or labels (like "Story Beat 2")

Expanded event:"""

# Legacy prompt - kept for reference but not used in outline-first approach
EVOLUTION_NEXT_EVENT_PROMPT_LEGACY = """Continue this story with the next event that naturally follows.

Story so far:
{previous_events}

Write the next event that connects to what happened before. Use words like "after", "following", "then", or "because" to show the connection.

Next event:"""


VARIATION_PROMPT = """Rewrite this fact {num_variations} different ways. Keep ALL numbers and names EXACTLY the same.

Original: "{fact}"

Examples of good variations:
1. Rearrange sentence structure
2. Use synonyms (but keep numbers/names exact)
3. Add context words like "reportedly", "according to reports"
4. Change active/passive voice

Write {num_variations} variations (one per line, numbered):"""


REVIEW_PROMPT = """Analyze these variations of a factual statement. They should express the SAME information with different wording.

Original seed fact: "{seed}"

Variations:
{variations_list}

Check for issues:
1. Are any variations IDENTICAL or near-duplicates (>90% similar)?
2. Do ANY variations have DIFFERENT numbers/names than the seed?
3. Are any variations expressing a COMPLETELY DIFFERENT fact?

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "all_consistent": true or false,
  "duplicate_indices": [list of indices that are duplicates, 0-based],
  "inconsistent_indices": [list of indices with wrong numbers/facts, 0-based],
  "issues_description": "brief description of problems found"
}}"""


CORRECTION_PROMPT = """The following fact variation has issues. Generate a CORRECTED version that:
- Expresses the SAME fact as the seed
- Uses the EXACT SAME numbers/names as the seed
- Has DIFFERENT wording than other variations

Seed fact: "{seed}"

Problematic variation: "{bad_variation}"

Other existing variations (avoid duplicating these):
{other_variations}

Issue: {issue_description}

Generate ONE corrected variation:"""


class FactGroupReviewer:
    """Reviews and corrects fact groups for consistency."""

    def __init__(self, llm_manager: LlamaServerManager, config: TemporalFactConfig):
        """
        Initialize fact group reviewer.

        Args:
            llm_manager: LLM manager for review/correction
            config: Configuration including review settings
        """
        self.llm_manager = llm_manager
        self.config = config
        self.stats = {
            "groups_reviewed": 0,
            "groups_passed": 0,
            "groups_corrected": 0,
            "total_corrections": 0,
        }

    def review_and_correct(self, seed_fact: str, variations: List[str]) -> List[str]:
        """
        Review variations for consistency and correct if needed.

        Args:
            seed_fact: Original seed fact (or seed event in evolution mode)
            variations: List of variation strings (or events in evolution mode)

        Returns:
            Corrected list of variations/events
        """
        self.stats["groups_reviewed"] += 1

        if not self.config.enable_review:
            return variations

        # Skip review for evolution mode (events should be different and connected)
        if self.config.evolution_mode:
            logger.debug("Skipping variation review for evolution mode")
            self.stats["groups_passed"] += 1
            return variations

        # Run LLM review for standard variation mode
        review_result = self._review_group(seed_fact, variations)

        if review_result is None:
            logger.warning("Review failed, returning original variations")
            return variations

        # Check if all consistent
        if review_result.get("all_consistent", False):
            self.stats["groups_passed"] += 1
            logger.debug("Fact group passed review")
            return variations

        # Auto-correct if enabled
        if self.config.auto_correct:
            corrected = self._correct_issues(seed_fact, variations, review_result)
            self.stats["groups_corrected"] += 1
            return corrected
        else:
            logger.warning(f"Issues found but auto_correct disabled: {review_result.get('issues_description')}")
            return variations

    def _review_group(self, seed_fact: str, variations: List[str]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to review a fact group.

        Args:
            seed_fact: Original seed fact
            variations: List of variations

        Returns:
            Review result dict or None if failed
        """
        # Format variations list
        variations_list = "\n".join([f"{i+1}. {var}" for i, var in enumerate(variations)])

        prompt = REVIEW_PROMPT.format(
            seed=seed_fact,
            variations_list=variations_list
        )

        try:
            result = self.llm_manager.generate(
                prompt=prompt,
                temperature=self.config.review_temperature,
                max_tokens=300,
                stop=["Seed fact:", "Variations:"]
            )

            if not result:
                return None

            # Parse JSON response
            result = result.strip()

            # Remove markdown code blocks if present
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join([l for l in lines if not l.startswith("```")])

            # Remove "json" tag if present
            result = result.replace("```json", "").replace("```", "").strip()

            review_data = json.loads(result)
            return review_data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse review JSON: {e}")
            logger.debug(f"Raw response: {result}")
            return None
        except Exception as e:
            logger.warning(f"Review failed: {e}")
            return None

    def _correct_issues(
        self,
        seed_fact: str,
        variations: List[str],
        review_result: Dict[str, Any]
    ) -> List[str]:
        """
        Correct variations based on review results.

        Args:
            seed_fact: Original seed fact
            variations: Current variations
            review_result: Review result with issue indices

        Returns:
            Corrected variations list
        """
        corrected = list(variations)
        issues_to_fix = set()

        # Collect all problematic indices
        duplicates = review_result.get("duplicate_indices", [])
        inconsistent = review_result.get("inconsistent_indices", [])
        issues_to_fix.update(duplicates)
        issues_to_fix.update(inconsistent)

        if not issues_to_fix:
            return corrected

        logger.info(f"Correcting {len(issues_to_fix)} variations in fact group")

        # Fix each problematic variation
        for idx in issues_to_fix:
            if idx >= len(variations):
                continue

            issue_type = "duplicate" if idx in duplicates else "inconsistent"
            corrected_var = self._correct_single_variation(
                seed_fact,
                variations[idx],
                corrected,
                issue_type
            )

            if corrected_var:
                corrected[idx] = corrected_var
                self.stats["total_corrections"] += 1
                logger.debug(f"Corrected variation {idx}: {corrected_var[:60]}...")

        return corrected

    def _correct_single_variation(
        self,
        seed_fact: str,
        bad_variation: str,
        other_variations: List[str],
        issue_type: str
    ) -> Optional[str]:
        """
        Correct a single problematic variation.

        Args:
            seed_fact: Original seed fact
            bad_variation: The problematic variation
            other_variations: Other variations to avoid duplicating
            issue_type: Type of issue (duplicate/inconsistent)

        Returns:
            Corrected variation or None if failed
        """
        # Format other variations
        others_text = "\n".join([f"- {v}" for v in other_variations if v != bad_variation])

        issue_desc = (
            "This variation is too similar to others"
            if issue_type == "duplicate"
            else "This variation has different numbers/facts than the seed"
        )

        prompt = CORRECTION_PROMPT.format(
            seed=seed_fact,
            bad_variation=bad_variation,
            other_variations=others_text,
            issue_description=issue_desc
        )

        try:
            result = self.llm_manager.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=["Seed fact:", "Issue:"]
            )

            if result:
                corrected = result.strip()

                # Clean up
                if ":" in corrected:
                    parts = corrected.split(":", 1)
                    if len(parts) > 1:
                        corrected = parts[1].strip()

                corrected = corrected.strip('"\'')

                return corrected if corrected and len(corrected) > 15 else None

        except Exception as e:
            logger.warning(f"Correction failed: {e}")

        return None

    def get_stats(self) -> Dict[str, int]:
        """Get review statistics."""
        return dict(self.stats)


class TemporalFactGenerator:
    """Generate temporal factual statements using LLM."""

    def __init__(
        self,
        config: TemporalFactConfig,
        llm_manager: Optional[LlamaServerManager] = None
    ):
        """
        Initialize temporal fact generator.

        Args:
            config: Generation configuration
            llm_manager: Optional pre-configured LLM manager
        """
        self.config = config
        self.llm_manager = llm_manager
        self._owns_llm = False

        # Validate domain
        if config.fact_domain not in DOMAIN_PROMPTS:
            available_domains = list(DOMAIN_PROMPTS.keys())
            logger.warning(
                f"Domain '{config.fact_domain}' not found in DOMAIN_PROMPTS. "
                f"Available domains: {available_domains}. "
                f"Will fall back to 'news' domain."
            )

        if self.llm_manager is None and config.model_path:
            logger.info(f"Initializing LLM with model: {config.model_path}")
            self.llm_manager = LlamaServerManager()
            self.llm_manager.load_model(config.model_path)
            self._owns_llm = True

        # Initialize reviewer if enabled
        self.reviewer = None
        if config.enable_review and self.llm_manager:
            self.reviewer = FactGroupReviewer(self.llm_manager, config)
            logger.info("Fact group reviewer enabled")

    def generate(self) -> List[Dict[str, Any]]:
        """
        Generate temporal fact dataset.

        Returns:
            List of dictionaries with timestamp, fact, group_id, variation_id
        """
        if not self.llm_manager or not self.llm_manager.is_ready:
            raise RuntimeError("LLM not ready. Provide model_path or llm_manager.")

        logger.info(
            f"Generating {self.config.num_fact_groups} fact groups with "
            f"{self.config.variations_per_group} variations each"
        )

        all_facts = []
        successful_groups = 0
        failed_groups = 0

        # Generate timestamp ranges for each group
        timestamp_ranges = self._generate_timestamp_ranges()

        for group_idx in range(self.config.num_fact_groups):
            logger.info(f"Generating fact group {group_idx + 1}/{self.config.num_fact_groups}")

            # Assign timestamps to this group
            timestamps = timestamp_ranges[group_idx]

            # Branch based on evolution mode
            if self.config.evolution_mode:
                # Evolution mode: generate connected story events
                events = self._generate_evolved_group(group_idx, timestamps)
                if not events:
                    logger.error(f"Failed to generate evolved group {group_idx} after all retries")
                    failed_groups += 1
                    continue

                # Ensure we have the requested number of events
                events = events[:self.config.variations_per_group]

                # If we didn't get enough events, pad with fallback
                while len(events) < self.config.variations_per_group:
                    events.append("Additional developments continued to unfold in the ongoing situation.")

                # Create fact entries for evolution mode
                group_id = f"group_{group_idx + 1:03d}"
                for event_idx, (event, timestamp) in enumerate(zip(events, timestamps)):
                    all_facts.append({
                        "timestamp": timestamp.isoformat(),
                        "fact": event.strip(),
                        "group_id": group_id,
                        "variation_id": event_idx + 1,
                    })

            else:
                # Standard variation mode: generate semantic variations of one fact
                # Generate seed fact with retries
                seed_fact = self._generate_seed_fact_with_retry()
                if not seed_fact:
                    logger.error(f"Failed to generate seed fact for group {group_idx} after all retries")
                    failed_groups += 1
                    continue

                logger.debug(f"Seed fact: {seed_fact}")

                # Generate variations with retries
                variations = self._generate_variations_with_retry(seed_fact)
                if not variations:
                    logger.error(f"Failed to generate variations for group {group_idx} after all retries")
                    failed_groups += 1
                    continue

                # Ensure we have the requested number of variations
                variations = variations[:self.config.variations_per_group]

                # If we didn't get enough variations, pad with the seed fact
                while len(variations) < self.config.variations_per_group:
                    variations.append(seed_fact)

                # Review and correct if enabled
                if self.reviewer:
                    variations = self.reviewer.review_and_correct(seed_fact, variations)

                # Create fact entries
                group_id = f"group_{group_idx + 1:03d}"
                for var_idx, (fact, timestamp) in enumerate(zip(variations, timestamps)):
                    all_facts.append({
                        "timestamp": timestamp.isoformat(),
                        "fact": fact.strip(),
                        "group_id": group_id,
                        "variation_id": var_idx + 1,
                    })

            successful_groups += 1

        # Log review statistics if reviewer was used
        if self.reviewer:
            review_stats = self.reviewer.get_stats()
            logger.info(
                f"Review stats: {review_stats['groups_passed']} passed, "
                f"{review_stats['groups_corrected']} corrected, "
                f"{review_stats['total_corrections']} individual corrections"
            )

        logger.info(
            f"Generation complete: {successful_groups}/{self.config.num_fact_groups} "
            f"groups successful, {failed_groups} failed, {len(all_facts)} total facts"
        )
        return all_facts

    def _generate_timestamp_ranges(self) -> List[List[datetime]]:
        """
        Generate timestamp ranges for each fact group.

        Returns:
            List of timestamp lists, one per group
        """
        start = datetime.fromisoformat(self.config.start_date)
        end = datetime.fromisoformat(self.config.end_date)

        # Calculate time delta based on frequency
        if self.config.frequency == "hourly":
            delta = timedelta(hours=1)
        elif self.config.frequency == "daily":
            delta = timedelta(days=1)
        elif self.config.frequency == "minutes":
            delta = timedelta(minutes=1)
        else:
            raise ValueError(f"Unsupported frequency: {self.config.frequency}")

        # Calculate total time span
        total_span = end - start

        # Divide time span across groups
        group_span = total_span / self.config.num_fact_groups

        timestamp_ranges = []
        for group_idx in range(self.config.num_fact_groups):
            group_start = start + group_span * group_idx

            # Generate timestamps for this group's variations
            timestamps = []
            current = group_start
            for _ in range(self.config.variations_per_group):
                timestamps.append(current)
                current += delta

            timestamp_ranges.append(timestamps)

        return timestamp_ranges

    def _generate_seed_fact_with_retry(self) -> Optional[str]:
        """
        Generate a seed fact with automatic retries.

        Returns:
            Generated fact string or None if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.use_exponential_backoff else 1)
                    logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} after {delay:.1f}s delay")
                    time.sleep(delay)

                result = self._generate_seed_fact()
                if result and len(result) > 20:  # Ensure we got a reasonable fact
                    return result
                else:
                    logger.warning(f"Seed fact generation attempt {attempt + 1} returned invalid/empty result")

            except Exception as e:
                logger.warning(f"Seed fact generation attempt {attempt + 1} failed: {e}")

        return None

    def _generate_seed_fact(self) -> Optional[str]:
        """
        Generate a seed fact using LLM (single attempt).

        Returns:
            Generated fact string or None if failed
        """
        # Get domain prompt with fallback
        if self.config.fact_domain in DOMAIN_PROMPTS:
            domain_prompt = DOMAIN_PROMPTS[self.config.fact_domain]
        else:
            logger.warning(
                f"Domain '{self.config.fact_domain}' not in DOMAIN_PROMPTS, using 'news' as fallback"
            )
            domain_prompt = DOMAIN_PROMPTS["news"]

        prompt = domain_prompt

        result = self.llm_manager.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=["\n\n", "Example:", "Generate"]
        )

        if result:
            # Clean up the result
            fact = result.strip()

            # Remove "Your fact:" prefix if present
            if ":" in fact:
                parts = fact.split(":", 1)
                if len(parts) > 1:
                    fact = parts[1].strip()

            # Remove numbering if present
            if fact and fact[0].isdigit():
                parts = fact.split('.', 1)
                if len(parts) > 1:
                    fact = parts[1].strip()

            # Remove quotes if present
            fact = fact.strip('"\'')

            # Remove any trailing examples or instructions
            for marker in ["Example:", "Your fact", "Generate"]:
                if marker in fact:
                    fact = fact.split(marker)[0].strip()

            return fact if fact else None

        return None

    def _generate_variations_with_retry(self, seed_fact: str) -> List[str]:
        """
        Generate variations with automatic retries and fallback.

        Args:
            seed_fact: Original fact to vary

        Returns:
            List of fact variations
        """
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.use_exponential_backoff else 1)
                    logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} for variations after {delay:.1f}s delay")
                    time.sleep(delay)

                variations = self._generate_variations(seed_fact)
                if len(variations) >= 3:  # At least 3 variations to be useful
                    return variations
                else:
                    logger.warning(f"Variation generation attempt {attempt + 1} returned {len(variations)} variations (need >= 3)")

            except Exception as e:
                logger.warning(f"Variation generation attempt {attempt + 1} failed: {e}")

        # Fallback: Create simple variations manually
        logger.warning(f"All variation generation attempts failed, using fallback variations")
        return self._create_fallback_variations(seed_fact)

    def _generate_variations(self, seed_fact: str) -> List[str]:
        """
        Generate variations of a seed fact (single attempt).

        Args:
            seed_fact: Original fact to vary

        Returns:
            List of fact variations
        """
        prompt = VARIATION_PROMPT.format(
            fact=seed_fact,
            num_variations=self.config.variations_per_group
        )

        result = self.llm_manager.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens * 3,
            stop=["Original:", "Example:"]
        )

        if not result:
            return []

        # Parse variations from result
        variations = []
        lines = result.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that look like instructions
            if any(keyword in line.lower() for keyword in ["example", "variation", "write", "rewrite", "original"]):
                continue

            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit():
                parts = line.split('.', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    continue

            # Remove bullet points
            line = line.lstrip('-*•').strip()

            # Remove quotes
            line = line.strip('"\'')

            # Must be a reasonable length and not the original
            if line and len(line) > 15 and line.lower() != seed_fact.lower():
                variations.append(line)

        return variations

    def _create_fallback_variations(self, seed_fact: str) -> List[str]:
        """
        Create simple manual variations as fallback.

        Args:
            seed_fact: Original fact

        Returns:
            List of simple variations
        """
        variations = [seed_fact]  # Include the original

        # Add simple prefix variations
        prefixes = [
            "According to reports, ",
            "It was reported that ",
            "Sources indicate that ",
            "Officials confirmed that ",
            "Data shows that ",
        ]

        for prefix in prefixes[:self.config.variations_per_group - 1]:
            # Lowercase first letter if adding prefix
            modified = seed_fact[0].lower() + seed_fact[1:] if seed_fact else seed_fact
            variations.append(prefix + modified)

        return variations[:self.config.variations_per_group]

    def _get_context_aware_fallback(self, seed_event: str, event_number: int) -> str:
        """
        Generate a context-aware fallback event when LLM generation fails.

        Args:
            seed_event: The initial seed event for context
            event_number: Which event number this is in the sequence

        Returns:
            A fallback event string that references the story context
        """
        # Extract key terms from seed event for context
        # Simple heuristic: get domain-relevant words
        domain_templates = {
            "news": [
                "Following the initial reports, authorities provided additional updates on the developing situation.",
                "As the story developed, new details emerged about the circumstances surrounding the incident.",
                "In the hours that followed, officials released more information to the public.",
            ],
            "business": [
                "Following the announcement, market analysts weighed in on the potential implications.",
                "As the news spread, stakeholders began assessing the impact on operations.",
                "In subsequent trading sessions, further market reactions developed.",
            ],
            "social": [
                "After the initial gathering began, more people arrived and conversations developed.",
                "As word spread through the community, additional residents joined the event.",
                "Following the initial meeting, participants began organizing follow-up activities.",
            ],
            "sports": [
                "Following the game, coaches and players addressed the media about the outcome.",
                "As news spread, fans and analysts reacted to the development.",
                "In the aftermath, team officials discussed next steps.",
            ],
            "science": [
                "Following the initial observation, researchers began detailed analysis of the data.",
                "As the findings emerged, the scientific community took notice.",
                "Subsequent analysis revealed additional insights about the phenomenon.",
            ],
            "entertainment": [
                "Following the release, critics and audiences shared their reactions.",
                "As word spread, box office tracking showed continued interest.",
                "In the days that followed, industry insiders discussed the implications.",
            ],
            "technology": [
                "Following the launch, tech analysts evaluated the new capabilities.",
                "As users began testing the features, early reviews started appearing.",
                "In subsequent announcements, additional details were revealed.",
            ],
        }

        # Get templates for current domain or use generic
        templates = domain_templates.get(self.config.fact_domain, domain_templates["news"])

        # Rotate through templates based on event number
        template_idx = event_number % len(templates)

        return templates[template_idx]

    def _generate_story_outline(self, num_events: int) -> Optional[List[str]]:
        """
        Generate a story outline with bullet-point events.

        Args:
            num_events: Number of events to include in outline

        Returns:
            List of bullet-point event descriptions, or None if failed
        """
        prompt = EVOLUTION_OUTLINE_PROMPT.format(
            num_events=num_events,
            domain=self.config.fact_domain
        )

        try:
            result = self.llm_manager.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=num_events * 30,  # ~30 tokens per bullet
                stop=["\n\n\n", "Example:", "Now write"]
            )

            if not result:
                logger.warning("Outline generation returned empty result")
                return None

            # Parse bullets from result
            bullets = []
            lines = result.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove numbering (1., 2., etc.)
                if line and line[0].isdigit():
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                    else:
                        continue
                elif line.startswith('-') or line.startswith('*'):
                    line = line[1:].strip()

                # Skip lines that look like instructions or examples
                if any(keyword in line.lower() for keyword in ["example", "requirement", "write", "event:", "bullet"]):
                    continue

                # Must be reasonable length (30-150 chars for a bullet)
                if line and 30 <= len(line) <= 150:
                    bullets.append(line)

            if len(bullets) >= num_events * 0.6:  # Got at least 60% of requested bullets
                logger.debug(f"Generated outline with {len(bullets)} bullets")
                return bullets[:num_events]  # Take only what we need
            else:
                logger.warning(f"Outline generation produced only {len(bullets)}/{num_events} bullets")
                return None

        except Exception as e:
            logger.error(f"Outline generation failed: {e}")
            return None

    def _expand_outline_bullet(
        self,
        bullet: str,
        previous_events: List[str],
        target_length_min: int = 600,
        target_length_max: int = 1100
    ) -> Optional[str]:
        """
        Expand a bullet-point event into a full description.

        Args:
            bullet: Bullet point to expand
            previous_events: Previously expanded events for context
            target_length_min: Minimum character length
            target_length_max: Maximum character length

        Returns:
            Expanded event description, or None if failed
        """
        # Build context from previous events (last 2 for brevity)
        if previous_events:
            context_events = previous_events[-2:] if len(previous_events) > 2 else previous_events
            previous_context = "\n".join([f"- {event[:100]}..." if len(event) > 100 else f"- {event}" for event in context_events])
        else:
            previous_context = "(This is the first event)"

        prompt = EVOLUTION_EXPAND_BULLET_PROMPT.format(
            bullet=bullet,
            previous_context=previous_context
        )

        try:
            result = self.llm_manager.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=200,  # Enough for 250-400 chars
                stop=["\n\n\n", "Event to expand:", "Context:"]
            )

            if not result:
                logger.warning(f"Expansion returned empty for bullet: {bullet[:50]}")
                return None

            # Clean up the result
            event = result.strip()

            # Remove "Expanded event:" prefix if present
            if event.lower().startswith("expanded event:"):
                event = event[15:].strip()
            elif ":" in event[:40]:
                parts = event.split(":", 1)
                if len(parts) > 1 and any(keyword in parts[0].lower() for keyword in ["expanded", "event"]):
                    event = parts[1].strip()

            # Remove leading numbering
            if event and len(event) > 3 and event[0].isdigit() and event[1:3] in ['. ', ') ']:
                event = event[3:].strip()

            # Remove quotes
            event = event.strip('"\'')

            # Check length
            if not (target_length_min <= len(event) <= target_length_max):
                logger.warning(f"Expanded event length {len(event)} outside target range {target_length_min}-{target_length_max}")
                # If it's too long, truncate intelligently
                if len(event) > target_length_max:
                    # Try to truncate at sentence boundary
                    truncate_pos = event.rfind('.', 0, target_length_max)
                    if truncate_pos > target_length_max * 0.8:
                        event = event[:truncate_pos + 1].strip()
                    else:
                        event = event[:target_length_max].strip()

            logger.debug(f"Expanded bullet to {len(event)} chars: {event[:80]}...")
            return event if len(event) >= 50 else None  # Minimum 50 chars

        except Exception as e:
            logger.error(f"Bullet expansion failed: {e}")
            return None

    def _generate_evolved_group(self, group_idx: int, timestamps: List[datetime]) -> List[str]:
        """
        Generate an evolved story group with connected events using outline-first approach.

        Args:
            group_idx: Index of the current group
            timestamps: List of timestamps for this group's events

        Returns:
            List of connected event strings
        """
        logger.info(f"Generating evolved story for group {group_idx + 1} (outline-first approach)")

        # Step 1: Generate story outline with retries
        outline = None
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                delay = self.config.retry_delay * (2 ** attempt if self.config.use_exponential_backoff else 1)
                logger.info(f"Retry outline generation attempt {attempt + 1}/{self.config.max_retries} after {delay:.1f}s")
                time.sleep(delay)

            outline = self._generate_story_outline(self.config.variations_per_group)
            if outline and len(outline) >= self.config.variations_per_group * 0.6:
                break

        if not outline:
            logger.error(f"Failed to generate outline for group {group_idx} after all retries")
            return []

        logger.info(f"Generated outline with {len(outline)} bullets for group {group_idx + 1}")
        for i, bullet in enumerate(outline, 1):
            logger.debug(f"  Bullet {i}: {bullet}")

        # Ensure we have enough bullets (pad if needed)
        while len(outline) < self.config.variations_per_group:
            outline.append("Additional developments continued in the ongoing situation.")

        # Step 2: Expand each bullet into a full event
        events = []
        for event_idx, bullet in enumerate(outline[:self.config.variations_per_group]):
            logger.debug(f"Expanding event {event_idx + 1}/{self.config.variations_per_group}: {bullet[:60]}...")

            # Try to expand with retries
            expanded = None
            for attempt in range(self.config.max_retries):
                if attempt > 0:
                    delay = self.config.retry_delay
                    logger.debug(f"Retry expansion attempt {attempt + 1}/{self.config.max_retries}")
                    time.sleep(delay)

                expanded = self._expand_outline_bullet(
                    bullet=bullet,
                    previous_events=events,
                    target_length_min=600,
                    target_length_max=1100  # Target ~1000 chars for rich story detail
                )

                if expanded:
                    break

            if not expanded:
                logger.warning(f"Failed to expand bullet {event_idx + 1}, using bullet as fallback")
                # Use the bullet itself with context prefix
                if events:
                    expanded = f"Following the previous developments, {bullet.lower()}"
                else:
                    expanded = bullet
                # Ensure minimum length
                if len(expanded) < 50:
                    expanded = f"At this stage of the unfolding events, {expanded}"

            events.append(expanded)
            logger.debug(f"Event {event_idx + 1} ({len(expanded)} chars): {expanded[:80]}...")

        return events

    def _generate_seed_event_with_retry(self) -> Optional[str]:
        """
        Generate a seed event for evolution mode with automatic retries.

        Returns:
            Generated seed event string or None if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.use_exponential_backoff else 1)
                    logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} after {delay:.1f}s delay")
                    time.sleep(delay)

                result = self._generate_seed_event()
                if result and len(result) > 20:
                    return result
                else:
                    logger.warning(f"Seed event generation attempt {attempt + 1} returned invalid/empty result")

            except Exception as e:
                logger.warning(f"Seed event generation attempt {attempt + 1} failed: {e}")

        return None

    def _generate_seed_event(self) -> Optional[str]:
        """
        Generate a seed event for evolution mode using LLM (single attempt).

        Returns:
            Generated seed event string or None if failed
        """
        # Get evolution seed prompt with fallback
        if self.config.fact_domain in EVOLUTION_SEED_PROMPTS:
            domain_prompt = EVOLUTION_SEED_PROMPTS[self.config.fact_domain]
        else:
            logger.warning(
                f"Domain '{self.config.fact_domain}' not in EVOLUTION_SEED_PROMPTS, using 'news' as fallback"
            )
            domain_prompt = EVOLUTION_SEED_PROMPTS["news"]

        # Use more tokens for seed events
        max_tokens_for_seed = max(200, self.config.max_tokens)

        result = self.llm_manager.generate(
            prompt=domain_prompt,
            temperature=self.config.temperature,
            max_tokens=max_tokens_for_seed,
            stop=["\n\n\n", "Example:", "Your seed event"]  # Minimal stop sequences
        )

        if result:
            # Log raw output for debugging
            logger.debug(f"Raw seed event output ({len(result)} chars): {result[:100]}...")

            # Clean up the result - be less aggressive
            event = result.strip()

            # Only remove prefix if it's clearly at the start
            if event.lower().startswith("your seed event:"):
                event = event[16:].strip()
            elif ":" in event[:40]:  # Only check first 40 chars for prefix
                parts = event.split(":", 1)
                if len(parts) > 1 and any(keyword in parts[0].lower() for keyword in ["event", "seed", "your", "fact"]):
                    event = parts[1].strip()

            # Remove leading numbering only if it's clearly a list number
            if event and len(event) > 3 and event[0].isdigit() and event[1:3] in ['. ', ') ']:
                event = event[3:].strip()

            # Remove quotes only from start and end
            event = event.strip('"\'')

            # Only remove trailing markers, not ones in middle of text
            for marker in ["Example:", "Your seed event"]:
                if marker in event:
                    marker_pos = event.find(marker)
                    # Only cut if marker is in last 20% of text
                    if marker_pos > len(event) * 0.8:
                        event = event[:marker_pos].strip()

            logger.debug(f"Cleaned seed event ({len(event)} chars): {event[:100]}...")

            return event if event and len(event) > 10 else None

        return None

    def _generate_next_event_with_retry(self, seed_event: str, previous_events: List[str]) -> Optional[str]:
        """
        Generate the next event in an evolved story with automatic retries.

        Args:
            seed_event: The initial seed event
            previous_events: List of all events generated so far (including seed)

        Returns:
            Generated next event string or None if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (2 ** attempt if self.config.use_exponential_backoff else 1)
                    logger.debug(f"Retry attempt {attempt + 1}/{self.config.max_retries} for next event after {delay:.1f}s delay")
                    time.sleep(delay)

                result = self._generate_next_event(seed_event, previous_events)
                if result and len(result) > 20:
                    return result
                else:
                    logger.warning(f"Next event generation attempt {attempt + 1} returned invalid/empty result")

            except Exception as e:
                logger.warning(f"Next event generation attempt {attempt + 1} failed: {e}")

        return None

    def _generate_next_event(self, seed_event: str, previous_events: List[str]) -> Optional[str]:
        """
        Generate the next event in an evolved story using LLM (single attempt).

        Args:
            seed_event: The initial seed event
            previous_events: List of all events generated so far

        Returns:
            Generated next event string or None if failed
        """
        # Implement context windowing - show seed + last 3 events for long stories
        if len(previous_events) > 4:
            context_events = [previous_events[0]] + previous_events[-3:]
            events_display = [
                f"1. {context_events[0]}",
                "...",
            ] + [f"{len(previous_events)-3+i}. {event}" for i, event in enumerate(context_events[1:])]
            previous_events_text = "\n".join(events_display)
        else:
            previous_events_text = "\n".join([f"{i+1}. {event}" for i, event in enumerate(previous_events)])

        prompt = EVOLUTION_NEXT_EVENT_PROMPT.format(
            previous_events=previous_events_text
        )

        # Use more tokens for evolution events (200 instead of default 150)
        max_tokens_for_event = max(200, self.config.max_tokens)

        result = self.llm_manager.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=max_tokens_for_event,
            stop=["\n\n\n", "Story so far:", "Next event:"]  # Minimal stop sequences
        )

        if result:
            # Log raw output for debugging
            logger.debug(f"Raw LLM output ({len(result)} chars): {result[:100]}...")

            # Clean up the result - be less aggressive
            event = result.strip()

            # Only remove prefix if it's clearly at the start
            if event.lower().startswith("next event:"):
                event = event[11:].strip()
            elif ":" in event[:30]:  # Only check first 30 chars for prefix
                parts = event.split(":", 1)
                if len(parts) > 1 and any(keyword in parts[0].lower() for keyword in ["event", "next"]):
                    event = parts[1].strip()

            # Remove leading numbering only if it's clearly a list number (e.g., "1. " or "5. ")
            if event and len(event) > 3 and event[0].isdigit() and event[1:3] in ['. ', ') ']:
                event = event[3:].strip()

            # Remove quotes only from start and end
            event = event.strip('"\'')

            # Only remove markers if they appear at the end (not in middle of text)
            for marker in ["Story so far:", "Next event:"]:
                if event.endswith(marker):
                    event = event[:-len(marker)].strip()

            logger.debug(f"Cleaned event ({len(event)} chars): {event[:100]}...")

            return event if event and len(event) > 10 else None

        return None

    def cleanup(self):
        """Clean up resources."""
        if self._owns_llm and self.llm_manager:
            logger.info("Cleaning up LLM resources")
            self.llm_manager.unload_model()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def generate_temporal_facts(
    model_path: str,
    num_fact_groups: int = 20,
    variations_per_group: int = 7,
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    frequency: str = "hourly",
    fact_domain: str = "news",
    temperature: float = 0.8,
    max_tokens: int = 150,
    seed: Optional[int] = None,
    evolution_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate temporal factual statements for embedding retrieval experiments.

    Args:
        model_path: Path to LLM model file (.gguf)
        num_fact_groups: Number of distinct fact groups
        variations_per_group: Number of variations per fact (or events in evolution mode)
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        frequency: Time frequency ('hourly', 'daily', 'minutes')
        fact_domain: Domain for facts ('news', 'business', 'science', 'sports')
        temperature: LLM sampling temperature
        max_tokens: Max tokens per generation
        seed: Random seed (not currently used)
        evolution_mode: If True, generate connected story events instead of variations

    Returns:
        List of fact dictionaries with timestamp, fact, group_id, variation_id

    Example:
        >>> facts = generate_temporal_facts(
        ...     model_path="models/llama-7b.gguf",
        ...     num_fact_groups=10,
        ...     variations_per_group=5,
        ...     frequency="hourly"
        ... )
        >>> # Evolution mode example
        >>> stories = generate_temporal_facts(
        ...     model_path="models/llama-7b.gguf",
        ...     num_fact_groups=5,
        ...     variations_per_group=7,
        ...     evolution_mode=True
        ... )
    """
    config = TemporalFactConfig(
        num_fact_groups=num_fact_groups,
        variations_per_group=variations_per_group,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        fact_domain=fact_domain,
        seed=seed,
        evolution_mode=evolution_mode,
    )

    with TemporalFactGenerator(config) as generator:
        return generator.generate()


def convert_to_samples(
    facts: List[Dict[str, Any]],
    evolution_mode: bool = False
) -> List:
    """
    Convert temporal facts to Sample objects for DatasetStore.

    Args:
        facts: List of temporal fact dictionaries
        evolution_mode: Whether facts were generated in evolution mode

    Returns:
        List of Sample objects
    """
    from ..sources.base import Sample, SourceMetadata

    samples = []
    for fact in facts:
        # Determine appropriate tags based on mode
        if evolution_mode:
            tags = ["temporal", "embedding-experiment", "evolution", fact['group_id']]
        else:
            tags = ["temporal", "embedding-experiment", "variation", fact['group_id']]

        metadata = SourceMetadata(
            source_type="generated_temporal_facts",
            tags=tags,
            custom={
                "timestamp": fact['timestamp'],
                "group_id": fact['group_id'],
                "variation_id": fact['variation_id'],
                "evolution_mode": evolution_mode,
            }
        )

        # Use fact as instruction, empty response for embedding experiments
        sample = Sample(
            instruction=fact['fact'],
            response="",
            metadata=metadata
        )
        samples.append(sample)

    return samples


def save_to_dataset_store(
    facts: List[Dict[str, Any]],
    dataset_name: str,
    store_path: str = "./datasets",
    generation_params: Optional[Dict[str, Any]] = None,
    evolution_mode: bool = False
) -> str:
    """
    Save temporal facts to DatasetStore with indexing.

    Args:
        facts: Temporal fact data
        dataset_name: Name for the dataset
        store_path: Path to dataset store directory
        generation_params: Parameters used for generation (saved as metadata)
        evolution_mode: Whether facts were generated in evolution mode

    Returns:
        Path to saved dataset
    """
    from ..storage.dataset_store import DatasetStore

    store = DatasetStore(base_path=store_path)

    # Convert to Sample objects
    samples = convert_to_samples(facts, evolution_mode=evolution_mode)

    # Prepare metadata
    metadata = {
        "type": "temporal_facts",
        "total_facts": len(facts),
        "num_groups": len(set(f['group_id'] for f in facts)),
        "generation_timestamp": datetime.now().isoformat(),
        "evolution_mode": evolution_mode,
    }

    # Add generation parameters if provided
    if generation_params:
        metadata["generation_params"] = generation_params

    # Add temporal statistics
    timestamps = [datetime.fromisoformat(f['timestamp']) for f in facts]
    metadata["temporal_stats"] = {
        "start_date": min(timestamps).isoformat(),
        "end_date": max(timestamps).isoformat(),
        "date_range_days": (max(timestamps) - min(timestamps)).days,
    }

    # Save to store
    filepath = store.save(
        dataset=samples,
        name=dataset_name,
        format="jsonl",
        metadata=metadata
    )

    logger.info(f"Saved {len(samples)} facts to dataset store: {dataset_name}")
    return filepath


def save_temporal_facts(
    data: List[Dict[str, Any]],
    output_path: str,
    format: Literal["jsonl", "json", "csv"] = "jsonl"
):
    """
    Save temporal fact data to file (standalone, without DatasetStore).

    Args:
        data: Temporal fact data
        output_path: Output file path
        format: Output format (jsonl, json, csv)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

    elif format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif format == "csv":
        import csv

        if not data:
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved {len(data)} facts to {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python temporal_fact_generator.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    logger.info("Generating temporal facts...")

    facts = generate_temporal_facts(
        model_path=model_path,
        num_fact_groups=5,
        variations_per_group=5,
        start_date="2024-01-01",
        end_date="2024-01-07",
        frequency="hourly",
        fact_domain="business"
    )

    # Save to file
    save_temporal_facts(facts, "temporal_facts.jsonl", format="jsonl")

    print(f"\nGenerated {len(facts)} facts")
    print(f"\nSample facts from first group:")
    for fact in facts[:5]:
        print(f"  [{fact['timestamp']}] {fact['fact']}")
