"""
Size factor implementation (log market capitalization, inverted).

This module implements the size anomaly: smaller companies tend to outperform
larger companies over time. This factor uses log market cap, inverted so that
smaller companies receive higher scores.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.factors.base import BaseFactor
from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class SizeFactor(BaseFactor):
    """
    Size factor based on market capitalization (inverted).

    The size anomaly suggests that small cap stocks tend to outperform
    large cap stocks. This factor computes log(market_cap) and inverts it
    (negative sign) so that smaller companies receive higher scores.

    Formula:
        size_score = -log(market_cap)

    Note: Using log transformation reduces the impact of extreme values
    and makes the distribution more normal.

    Attributes:
        use_log: Whether to use log transformation (default True)

    Example:
        >>> config = {
        ...     'use_log': True,
        ...     'enabled': True
        ... }
        >>> size = SizeFactor(config)
        >>> scores = size.compute(prices, fundamentals, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 smallest companies
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize size factor from configuration.

        Args:
            config: Configuration dict with keys:
                - use_log: Use log transformation (default True)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)

        Example:
            >>> config = {
            ...     'use_log': True,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = SizeFactor(config)
        """
        # Extract configuration
        self.use_log = config.get('use_log', True)
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)

        # Initialize base class
        super().__init__(name="size", enabled=enabled, weight=weight)

        logger.info(
            f"Initialized {self.name}: use_log={self.use_log}"
        )

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute size scores for all tickers (inverted).

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Dict with 'market_cap' DataFrame
            as_of_date: Calculation date

        Returns:
            Series with size scores (higher = smaller company)

        Raises:
            InsufficientDataError: If required fundamental data missing
            DataError: If data format invalid
        """
        # Validate inputs
        self._validate_prices(prices)
        as_of_date = self._validate_date_in_range(prices, as_of_date)

        # Validate fundamentals
        self._validate_fundamentals(fundamentals)

        logger.info(
            f"Computing {self.name} for {len(prices.columns)} tickers "
            f"as of {as_of_date.date()}"
        )

        try:
            # Get current market prices for all tickers
            current_prices = prices.loc[as_of_date]
            tickers = current_prices.index.tolist()

            # Get market cap values
            market_caps = self._get_fundamental_values(
                fundamentals['market_cap'], tickers, as_of_date
            )

            # Compute size scores
            if self.use_log:
                # Log transformation (handle zeros and negatives)
                size_scores = np.log(market_caps)
                size_scores = size_scores.replace([np.inf, -np.inf], np.nan)
            else:
                # Raw market cap
                size_scores = market_caps.copy()

            # Invert: smaller companies = higher score
            size_scores = -size_scores

            # Handle edge cases
            size_scores = self._handle_edge_cases(size_scores, market_caps)

            # Log statistics
            valid_scores = size_scores.dropna()
            if len(valid_scores) > 0:
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(size_scores)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")

            return size_scores

        except InsufficientDataError:
            # Re-raise InsufficientDataError as-is (don't wrap it)
            raise
        except Exception as e:
            logger.error(f"{self.name}: Error computing size: {e}")
            raise DataError(f"Size computation failed: {e}")

    def _validate_fundamentals(self, fundamentals: Dict[str, pd.DataFrame]) -> None:
        """
        Validate fundamental data format and required fields.

        Args:
            fundamentals: Dict of fundamental DataFrames

        Raises:
            DataError: If validation fails
        """
        required_fields = ['market_cap']

        if not isinstance(fundamentals, dict):
            raise DataError(f"Fundamentals must be dict, got {type(fundamentals)}")

        for field in required_fields:
            if field not in fundamentals:
                raise DataError(
                    f"Missing required fundamental field: '{field}'. "
                    f"Available: {list(fundamentals.keys())}"
                )

            if not isinstance(fundamentals[field], pd.DataFrame):
                raise DataError(
                    f"Fundamental field '{field}' must be DataFrame, "
                    f"got {type(fundamentals[field])}"
                )

            if fundamentals[field].empty:
                raise DataError(f"Fundamental field '{field}' is empty")

    def _get_fundamental_values(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Extract fundamental values for specific tickers and date.

        Args:
            data: DataFrame with tickers as columns and dates as index
            tickers: List of ticker symbols to extract
            as_of_date: Date to extract data for

        Returns:
            Series with values for each ticker
        """
        # Ensure DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataError(
                f"Fundamental data must have DatetimeIndex, got {type(data.index)}"
            )

        # Get values at or before as_of_date (point-in-time)
        historical = data.loc[:as_of_date]

        if len(historical) == 0:
            raise InsufficientDataError(
                f"No fundamental data available before {as_of_date.date()}"
            )

        # Use most recent available data
        latest_values = historical.iloc[-1]

        # Extract only requested tickers (fill missing with NaN)
        result = pd.Series(index=tickers, dtype=float)
        for ticker in tickers:
            if ticker in latest_values.index:
                result[ticker] = latest_values[ticker]
            else:
                result[ticker] = np.nan

        return result

    def _handle_edge_cases(
        self,
        scores: pd.Series,
        market_caps: pd.Series,
    ) -> pd.Series:
        """
        Handle edge cases in size calculation.

        Args:
            scores: Raw size scores (inverted log market cap)
            market_caps: Original market cap values

        Returns:
            Cleaned size scores
        """
        # Handle zero or negative market cap (invalid)
        invalid_mask = (market_caps <= 0) | market_caps.isna()
        if invalid_mask.any():
            invalid_tickers = scores.index[invalid_mask].tolist()
            logger.warning(
                f"{self.name}: {len(invalid_tickers)} tickers have zero/negative/NaN "
                f"market cap, setting to NaN: {invalid_tickers[:5]}"
            )
            scores[invalid_mask] = np.nan

        return scores

    def required_data(self) -> Dict[str, List[str]]:
        """
        Specify required data fields.

        Returns:
            Dict with 'prices' and 'fundamentals' requirements
        """
        return {
            'prices': ['Close'],
            'fundamentals': ['market_cap']
        }

    def get_required_history_days(self) -> int:
        """
        Get minimum number of trading days required for this factor.

        Returns:
            Minimum trading days needed
        """
        # Size factor only needs current market cap data
        return 1


def compute_size(
    prices: pd.DataFrame,
    fundamentals: Dict[str, pd.DataFrame],
    as_of_date: datetime,
    use_log: bool = True,
) -> pd.Series:
    """
    Standalone function to compute size scores.

    This is a convenience function that can be used without instantiating
    the SizeFactor class.

    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        fundamentals: Dict with 'market_cap' DataFrame
        as_of_date: Calculation date
        use_log: Use log transformation (default True)

    Returns:
        Series with size scores (higher = smaller company)

    Example:
        >>> size_scores = compute_size(
        ...     prices,
        ...     fundamentals,
        ...     datetime(2024, 1, 31),
        ...     use_log=True
        ... )
    """
    # Create temporary config
    config = {
        'use_log': use_log,
        'enabled': True,
        'weight': 1.0
    }

    # Use class implementation
    factor = SizeFactor(config)
    return factor.compute(prices, fundamentals, as_of_date)
