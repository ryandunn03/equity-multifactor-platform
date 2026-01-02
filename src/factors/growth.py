"""
Growth factor implementation (revenue + earnings growth).

This module implements the growth anomaly: companies with strong revenue and
earnings growth tend to outperform over time.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.factors.base import BaseFactor
from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class GrowthFactor(BaseFactor):
    """
    Growth factor combining revenue growth and earnings growth.

    The growth anomaly suggests that companies with strong fundamental
    growth outperform over time. This combines:
    - Revenue Growth: (revenue[t] / revenue[t-lookback]) - 1
    - Earnings Growth: (earnings[t] / earnings[t-lookback]) - 1

    Formula:
        revenue_growth = (revenue[t] / revenue[t-lookback]) - 1
        earnings_growth = (earnings[t] / earnings[t-lookback]) - 1
        growth_score = w_rev * rev_growth_z + w_earn * earn_growth_z

    Attributes:
        weight_revenue_growth: Weight for revenue growth component (0-1)
        weight_earnings_growth: Weight for earnings growth component (0-1)
        lookback_quarters: Number of quarters to look back for growth calculation

    Example:
        >>> config = {
        ...     'weight_revenue_growth': 0.5,
        ...     'weight_earnings_growth': 0.5,
        ...     'lookback_quarters': 4,
        ...     'enabled': True
        ... }
        >>> growth = GrowthFactor(config)
        >>> scores = growth.compute(prices, fundamentals, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 growth stocks
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize growth factor from configuration.

        Args:
            config: Configuration dict with keys:
                - weight_revenue_growth: Weight for revenue growth (default 0.5)
                - weight_earnings_growth: Weight for earnings growth (default 0.5)
                - lookback_quarters: Quarters to look back (default 4, i.e., YoY)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)

        Example:
            >>> config = {
            ...     'weight_revenue_growth': 0.5,
            ...     'weight_earnings_growth': 0.5,
            ...     'lookback_quarters': 4,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = GrowthFactor(config)
        """
        # Extract configuration
        self.weight_revenue_growth = config.get('weight_revenue_growth', 0.5)
        self.weight_earnings_growth = config.get('weight_earnings_growth', 0.5)
        self.lookback_quarters = config.get('lookback_quarters', 4)  # YoY default
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)

        # Validate weights
        if not (0 <= self.weight_revenue_growth <= 1):
            raise ValueError(f"weight_revenue_growth must be in [0,1], got {self.weight_revenue_growth}")
        if not (0 <= self.weight_earnings_growth <= 1):
            raise ValueError(f"weight_earnings_growth must be in [0,1], got {self.weight_earnings_growth}")

        # Validate lookback
        if self.lookback_quarters < 1:
            raise ValueError(f"lookback_quarters must be >= 1, got {self.lookback_quarters}")

        # Normalize weights to sum to 1
        total_weight = self.weight_revenue_growth + self.weight_earnings_growth
        if total_weight == 0:
            raise ValueError("At least one component weight must be non-zero")

        self.weight_revenue_growth /= total_weight
        self.weight_earnings_growth /= total_weight

        # Initialize base class
        super().__init__(name="growth", enabled=enabled, weight=weight)

        logger.info(
            f"Initialized {self.name}: Revenue weight={self.weight_revenue_growth:.2f}, "
            f"Earnings weight={self.weight_earnings_growth:.2f}, "
            f"Lookback={self.lookback_quarters} quarters"
        )

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute growth scores for all tickers.

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Dict with 'revenue', 'earnings' DataFrames
            as_of_date: Calculation date

        Returns:
            Series with growth scores (higher = stronger growth)

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

            # Compute Revenue Growth
            revenue_growth = self._compute_growth_rate(
                fundamentals['revenue'], tickers, as_of_date, 'revenue'
            )

            # Compute Earnings Growth
            earnings_growth = self._compute_growth_rate(
                fundamentals['earnings'], tickers, as_of_date, 'earnings'
            )

            # Combine signals using z-scores
            growth_scores = self._combine_signals(revenue_growth, earnings_growth)

            # Log statistics
            valid_scores = growth_scores.dropna()
            if len(valid_scores) > 0:
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(growth_scores)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")

            return growth_scores

        except InsufficientDataError:
            # Re-raise InsufficientDataError as-is (don't wrap it)
            raise
        except Exception as e:
            logger.error(f"{self.name}: Error computing growth: {e}")
            raise DataError(f"Growth computation failed: {e}")

    def _validate_fundamentals(self, fundamentals: Dict[str, pd.DataFrame]) -> None:
        """
        Validate fundamental data format and required fields.

        Args:
            fundamentals: Dict of fundamental DataFrames

        Raises:
            DataError: If validation fails
        """
        required_fields = ['revenue', 'earnings']

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

    def _compute_growth_rate(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        as_of_date: datetime,
        field_name: str,
    ) -> pd.Series:
        """
        Compute growth rate for a given fundamental field.

        Args:
            data: DataFrame with tickers as columns and dates as index
            tickers: List of ticker symbols
            as_of_date: Calculation date
            field_name: Name of field (for logging)

        Returns:
            Series of growth rates
        """
        # Ensure DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataError(
                f"Fundamental data must have DatetimeIndex, got {type(data.index)}"
            )

        # Get values at current date
        current_values = self._get_fundamental_values(data, tickers, as_of_date)

        # Get values at lookback date (approximately quarters * 90 days)
        lookback_days = self.lookback_quarters * 90
        lookback_values = self._get_fundamental_values_at_offset(
            data, tickers, as_of_date, lookback_days
        )

        # Compute growth: (current / past) - 1
        growth = (current_values / lookback_values) - 1.0
        growth = growth.replace([np.inf, -np.inf], np.nan)

        # Handle edge cases
        growth = self._handle_growth_edge_cases(
            growth, current_values, lookback_values, field_name
        )

        valid_count = growth.notna().sum()
        logger.debug(
            f"{self.name}: {field_name} growth computed for {valid_count}/{len(tickers)} tickers"
        )

        return growth

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

    def _get_fundamental_values_at_offset(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        as_of_date: datetime,
        lookback_days: int,
    ) -> pd.Series:
        """
        Get fundamental values at approximately lookback_days before as_of_date.

        Args:
            data: DataFrame with tickers as columns and dates as index
            tickers: List of ticker symbols
            as_of_date: Reference date
            lookback_days: Approximate days to look back

        Returns:
            Series with values for each ticker
        """
        # Get all data up to as_of_date
        historical = data.loc[:as_of_date]

        if len(historical) < 2:
            # Not enough history for growth calculation
            return pd.Series(np.nan, index=tickers, dtype=float)

        # Find date approximately lookback_days ago
        # Since fundamental data is sparse (quarterly), find closest available date
        target_date = as_of_date - pd.Timedelta(days=lookback_days)

        # Get all dates before as_of_date
        available_dates = historical.index

        # Find closest date to target (but before as_of_date)
        dates_before_target = available_dates[available_dates <= as_of_date]

        if len(dates_before_target) < 2:
            # Not enough history
            return pd.Series(np.nan, index=tickers, dtype=float)

        # Use second-to-last available date (simple approach for quarterly data)
        # For more accuracy, could search for closest to target_date
        # But for quarterly data, just going back by index works well
        offset_idx = min(self.lookback_quarters, len(dates_before_target) - 1)
        offset_values = historical.iloc[-(offset_idx + 1)]

        # Extract only requested tickers
        result = pd.Series(index=tickers, dtype=float)
        for ticker in tickers:
            if ticker in offset_values.index:
                result[ticker] = offset_values[ticker]
            else:
                result[ticker] = np.nan

        return result

    def _handle_growth_edge_cases(
        self,
        growth: pd.Series,
        current_values: pd.Series,
        lookback_values: pd.Series,
        field_name: str,
    ) -> pd.Series:
        """
        Handle edge cases in growth calculation.

        Args:
            growth: Raw growth rates
            current_values: Current fundamental values
            lookback_values: Historical fundamental values
            field_name: Name of field (for logging)

        Returns:
            Cleaned growth rates
        """
        # Handle division by zero or negative base values
        invalid_base_mask = (lookback_values <= 0) | lookback_values.isna()
        if invalid_base_mask.any():
            invalid_tickers = growth.index[invalid_base_mask].tolist()
            logger.debug(
                f"{self.name}: {len(invalid_tickers)} tickers have zero/negative/NaN "
                f"{field_name} at lookback date: {invalid_tickers[:5]}"
            )
            growth[invalid_base_mask] = np.nan

        # Handle negative current values (losses for earnings)
        # For revenue, negative is invalid. For earnings, negative is valid but growth is tricky
        if field_name == 'revenue':
            invalid_current_mask = (current_values <= 0) | current_values.isna()
            if invalid_current_mask.any():
                invalid_tickers = growth.index[invalid_current_mask].tolist()
                logger.debug(
                    f"{self.name}: {len(invalid_tickers)} tickers have zero/negative "
                    f"revenue (invalid): {invalid_tickers[:5]}"
                )
                growth[invalid_current_mask] = np.nan

        # Cap extreme growth rates (likely data errors)
        # Growth > 1000% or < -100% is suspicious
        extreme_mask = (growth > 10.0) | (growth < -1.0)
        if extreme_mask.any():
            extreme_tickers = growth.index[extreme_mask].tolist()
            logger.warning(
                f"{self.name}: {len(extreme_tickers)} tickers have extreme {field_name} "
                f"growth (>1000% or <-100%), capping: {extreme_tickers[:5]}"
            )
            growth = growth.clip(-1.0, 10.0)

        return growth

    def _combine_signals(
        self,
        revenue_growth: pd.Series,
        earnings_growth: pd.Series,
    ) -> pd.Series:
        """
        Combine revenue and earnings growth using z-scores.

        Args:
            revenue_growth: Revenue growth rates
            earnings_growth: Earnings growth rates

        Returns:
            Combined growth scores
        """
        # Compute z-scores for each component
        rev_zscore = self._compute_zscore(revenue_growth)
        earn_zscore = self._compute_zscore(earnings_growth)

        # Weighted combination
        combined = (
            self.weight_revenue_growth * rev_zscore +
            self.weight_earnings_growth * earn_zscore
        )

        return combined

    def _compute_zscore(self, series: pd.Series) -> pd.Series:
        """
        Compute z-scores (standardized values).

        Args:
            series: Input series

        Returns:
            Z-scored series
        """
        valid = series.dropna()

        if len(valid) == 0:
            return pd.Series(np.nan, index=series.index)

        mean = valid.mean()
        std = valid.std()

        if pd.isna(std) or std < 1e-10:
            # All values identical or std too small - return zeros
            return pd.Series(0.0, index=series.index)

        zscore = (series - mean) / std

        return zscore

    def required_data(self) -> Dict[str, List[str]]:
        """
        Specify required data fields.

        Returns:
            Dict with 'prices' and 'fundamentals' requirements
        """
        return {
            'prices': ['Close'],
            'fundamentals': ['revenue', 'earnings']
        }

    def get_required_history_days(self) -> int:
        """
        Get minimum number of trading days required for this factor.

        Returns:
            Minimum trading days needed
        """
        # Need at least lookback_quarters worth of fundamental data
        # Approximate as quarters * 90 days
        return self.lookback_quarters * 90 + 30  # Buffer


def compute_growth(
    prices: pd.DataFrame,
    fundamentals: Dict[str, pd.DataFrame],
    as_of_date: datetime,
    weight_revenue_growth: float = 0.5,
    weight_earnings_growth: float = 0.5,
    lookback_quarters: int = 4,
) -> pd.Series:
    """
    Standalone function to compute growth scores.

    This is a convenience function that can be used without instantiating
    the GrowthFactor class.

    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        fundamentals: Dict with 'revenue', 'earnings' DataFrames
        as_of_date: Calculation date
        weight_revenue_growth: Weight for revenue growth (default 0.5)
        weight_earnings_growth: Weight for earnings growth (default 0.5)
        lookback_quarters: Quarters to look back (default 4, YoY)

    Returns:
        Series with growth scores

    Example:
        >>> growth_scores = compute_growth(
        ...     prices,
        ...     fundamentals,
        ...     datetime(2024, 1, 31),
        ...     weight_revenue_growth=0.5,
        ...     weight_earnings_growth=0.5,
        ...     lookback_quarters=4
        ... )
    """
    # Create temporary config
    config = {
        'weight_revenue_growth': weight_revenue_growth,
        'weight_earnings_growth': weight_earnings_growth,
        'lookback_quarters': lookback_quarters,
        'enabled': True,
        'weight': 1.0
    }

    # Use class implementation
    factor = GrowthFactor(config)
    return factor.compute(prices, fundamentals, as_of_date)
