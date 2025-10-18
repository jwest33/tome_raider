"""Time series data generator."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from loguru import logger


@dataclass
class TimeSeriesConfig:
    """Configuration for time series generation."""

    num_series: int = 5
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-31"
    frequency: str = "hourly"  # hourly, daily, minutes

    # Pattern parameters
    base_value: float = 100.0
    trend: float = 0.0  # per hour
    noise_std: float = 5.0

    # Seasonality
    daily_seasonality: bool = True
    daily_amplitude: float = 10.0
    weekly_seasonality: bool = False
    weekly_amplitude: float = 15.0

    # Cross-series variation
    series_variation: float = 0.1  # 10% variation in base values
    seed: Optional[int] = None


class TimeSeriesGenerator:
    """Generate synthetic time series data with similar patterns."""

    def __init__(self, config: TimeSeriesConfig):
        """
        Initialize time series generator.

        Args:
            config: Time series configuration
        """
        self.config = config

        if config.seed is not None:
            np.random.seed(config.seed)

    def generate(self) -> List[Dict[str, Any]]:
        """
        Generate time series data.

        Returns:
            List of dictionaries with timestamp, value, and series_id
        """
        logger.info(f"Generating {self.config.num_series} time series from {self.config.start_date} to {self.config.end_date}")

        # Generate timestamps
        timestamps = self._generate_timestamps()
        logger.info(f"Generated {len(timestamps)} timestamps")

        # Generate multiple series
        all_data = []
        for series_idx in range(self.config.num_series):
            series_id = f"series_{series_idx + 1}"
            series_data = self._generate_series(timestamps, series_id)
            all_data.extend(series_data)

        logger.info(f"Generated {len(all_data)} total data points")
        return all_data

    def _generate_timestamps(self) -> List[datetime]:
        """Generate timestamp sequence."""
        start = datetime.fromisoformat(self.config.start_date)
        end = datetime.fromisoformat(self.config.end_date)

        if self.config.frequency == "hourly":
            delta = timedelta(hours=1)
        elif self.config.frequency == "daily":
            delta = timedelta(days=1)
        elif self.config.frequency == "minutes":
            delta = timedelta(minutes=1)
        else:
            raise ValueError(f"Unsupported frequency: {self.config.frequency}")

        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current)
            current += delta

        return timestamps

    def _generate_series(
        self,
        timestamps: List[datetime],
        series_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate a single time series.

        Args:
            timestamps: List of timestamps
            series_id: Identifier for this series

        Returns:
            List of data points
        """
        # Apply variation to base parameters for this series
        series_base = self.config.base_value * (1 + np.random.uniform(
            -self.config.series_variation,
            self.config.series_variation
        ))

        series_trend = self.config.trend * (1 + np.random.uniform(-0.2, 0.2))

        # Generate values
        values = []
        start_time = timestamps[0]

        for i, ts in enumerate(timestamps):
            # Base value with trend
            value = series_base + series_trend * i

            # Add daily seasonality
            if self.config.daily_seasonality:
                hour_of_day = ts.hour + ts.minute / 60.0
                daily_component = self.config.daily_amplitude * np.sin(
                    2 * np.pi * hour_of_day / 24.0
                )
                value += daily_component

            # Add weekly seasonality
            if self.config.weekly_seasonality:
                day_of_week = ts.weekday()
                weekly_component = self.config.weekly_amplitude * np.sin(
                    2 * np.pi * day_of_week / 7.0
                )
                value += weekly_component

            # Add random noise
            noise = np.random.normal(0, self.config.noise_std)
            value += noise

            values.append(value)

        # Create data points
        data = []
        for ts, val in zip(timestamps, values):
            data.append({
                "timestamp": ts.isoformat(),
                "value": round(float(val), 2),
                "series_id": series_id,
            })

        return data


def generate_time_series(
    num_series: int = 5,
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    frequency: str = "hourly",
    base_value: float = 100.0,
    trend: float = 0.0,
    noise_std: float = 5.0,
    daily_seasonality: bool = True,
    daily_amplitude: float = 10.0,
    weekly_seasonality: bool = False,
    weekly_amplitude: float = 15.0,
    series_variation: float = 0.1,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate time series data with similar patterns but different timestamps.

    Args:
        num_series: Number of time series to generate
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        frequency: Time frequency ('hourly', 'daily', 'minutes')
        base_value: Base value for the series
        trend: Trend component (change per time step)
        noise_std: Standard deviation of random noise
        daily_seasonality: Whether to add daily seasonal pattern
        daily_amplitude: Amplitude of daily seasonality
        weekly_seasonality: Whether to add weekly seasonal pattern
        weekly_amplitude: Amplitude of weekly seasonality
        series_variation: Variation between series (0-1)
        seed: Random seed for reproducibility

    Returns:
        List of data points with timestamp, value, and series_id

    Example:
        >>> data = generate_time_series(
        ...     num_series=3,
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-07",
        ...     frequency="hourly",
        ...     base_value=100.0,
        ...     noise_std=5.0
        ... )
        >>> len(data)
        504  # 3 series * 7 days * 24 hours
    """
    config = TimeSeriesConfig(
        num_series=num_series,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        base_value=base_value,
        trend=trend,
        noise_std=noise_std,
        daily_seasonality=daily_seasonality,
        daily_amplitude=daily_amplitude,
        weekly_seasonality=weekly_seasonality,
        weekly_amplitude=weekly_amplitude,
        series_variation=series_variation,
        seed=seed,
    )

    generator = TimeSeriesGenerator(config)
    return generator.generate()


def save_time_series(
    data: List[Dict[str, Any]],
    output_path: str,
    format: Literal["jsonl", "json", "csv"] = "jsonl"
):
    """
    Save time series data to file.

    Args:
        data: Time series data
        output_path: Output file path
        format: Output format (jsonl, json, csv)
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        import csv

        if not data:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved {len(data)} data points to {output_path}")


if __name__ == "__main__":
    # Example usage
    data = generate_time_series(
        num_series=5,
        start_date="2024-01-01",
        end_date="2024-01-31",
        frequency="hourly",
        base_value=100.0,
        trend=0.05,  # slight upward trend
        noise_std=5.0,
        daily_seasonality=True,
        daily_amplitude=10.0,
        seed=42
    )

    # Save to file
    save_time_series(data, "time_series_data.jsonl", format="jsonl")

    print(f"Generated {len(data)} data points")
    print(f"Sample data points:")
    for i in range(min(5, len(data))):
        print(f"  {data[i]}")
