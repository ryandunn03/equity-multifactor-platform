"""
Value factor implementation (Book-to-Market + Earnings Yield).

This module implements the value anomaly: stocks trading at low valuations
relative to fundamentals tend to outperform.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.factors.base import BaseFactor
from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class ValueFactor(BaseFactor):
    """
    Value factor combining Book-to-Market ratio and Earnings Yield.

    The value anomaly suggests that stocks with high book-to-market ratios
    and earnings yields outperform over the long term. This implementation
    combines both signals.

    Formula:
        book_to_market = book_value / market_cap
        earnings_yield = earnings / market_cap
        value_score = weight_bm * bm_zscore + weight_ey * ey_zscore

    Attributes:
        weight_book_to_market: Weight for book-to-market component (0-1)
        weight_earnings_yield: Weight for earnings yield component (0-1)

    Example:
        >>> config = {
        ...     'weight_book_to_market': 0.5,
        ...     'weight_earnings_yield': 0.5,
        ...     'enabled': True
        ... }
        >>> value = ValueFactor(config)
        >>> scores = value.compute(prices, fundamentals, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 value stocks
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize value factor from configuration.

        Args:
            config: Configuration dict with keys:
                - weight_book_to_market: Weight for B/M ratio (default 0.5)
                - weight_earnings_yield: Weight for E/P ratio (default 0.5)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)

        Example:
            >>> config = {
            ...     'weight_book_to_market': 0.5,
            ...     'weight_earnings_yield': 0.5,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = ValueFactor(config)
        """
        # Extract configuration
        self.weight_book_to_market = config.get('weight_book_to_market', 0.5)
        self.weight_earnings_yield = config.get('weight_earnings_yield', 0.5)
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)

        # Validate weights
        if not (0 <= self.weight_book_to_market <= 1):
            raise ValueError(f"weight_book_to_market must be in [0,1], got {self.weight_book_to_market}")
        if not (0 <= self.weight_earnings_yield <= 1):
            raise ValueError(f"weight_earnings_yield must be in [0,1], got {self.weight_earnings_yield}")

        # Normalize weights to sum to 1
        total_weight = self.weight_book_to_market + self.weight_earnings_yield
        if total_weight == 0:
            raise ValueError("At least one component weight must be non-zero")

        self.weight_book_to_market /= total_weight
        self.weight_earnings_yield /= total_weight

        # Initialize base class
        super().__init__(name="value", enabled=enabled, weight=weight)

        logger.info(
            f"Initialized {self.name}: B/M weight={self.weight_book_to_market:.2f}, "
            f"E/Y weight={self.weight_earnings_yield:.2f}"
        )

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute value scores for all tickers.

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Dict with 'book_value', 'market_cap', 'earnings' DataFrames
            as_of_date: Calculation date

        Returns:
            Series with value scores (higher = more attractive value)

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

            # Compute Book-to-Market
            book_to_market = self._compute_book_to_market(
                fundamentals, tickers, as_of_date
            )

            # Compute Earnings Yield
            earnings_yield = self._compute_earnings_yield(
                fundamentals, tickers, as_of_date
            )

            # Combine signals using z-scores
            value_scores = self._combine_signals(book_to_market, earnings_yield)

            # Log statistics
            valid_scores = value_scores.dropna()
            if len(valid_scores) > 0:
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(value_scores)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")

            return value_scores

        except InsufficientDataError:
            # Re-raise InsufficientDataError as-is (don't wrap it)
            raise
        except Exception as e:
            logger.error(f"{self.name}: Error computing value: {e}")
            raise DataError(f"Value computation failed: {e}")

    def _validate_fundamentals(self, fundamentals: Dict[str, pd.DataFrame]) -> None:
        """
        Validate fundamental data format and required fields.

        Args:
            fundamentals: Dict of fundamental DataFrames

        Raises:
            DataError: If validation fails
        """
        required_fields = ['book_value', 'market_cap', 'earnings']

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

    def _compute_book_to_market(
        self,
        fundamentals: Dict[str, pd.DataFrame],
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute book-to-market ratio for all tickers.

        Args:
            fundamentals: Dict of fundamental DataFrames
            tickers: List of ticker symbols
            as_of_date: Calculation date

        Returns:
            Series of book-to-market ratios
        """
        book_values = self._get_fundamental_values(
            fundamentals['book_value'], tickers, as_of_date
        )
        market_caps = self._get_fundamental_values(
            fundamentals['market_cap'], tickers, as_of_date
        )

        # B/M = Book Value / Market Cap
        # Handle division by zero
        book_to_market = book_values / market_caps
        book_to_market = book_to_market.replace([np.inf, -np.inf], np.nan)

        # Filter out negative book values (problematic companies)
        book_to_market[book_values <= 0] = np.nan

        # Cap extreme values (likely data errors)
        book_to_market = book_to_market.clip(0, 10)

        valid_count = book_to_market.notna().sum()
        logger.debug(
            f"{self.name}: Book-to-Market computed for {valid_count}/{len(tickers)} tickers"
        )

        return book_to_market

    def _compute_earnings_yield(
        self,
        fundamentals: Dict[str, pd.DataFrame],
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute earnings yield (E/P ratio) for all tickers.

        Args:
            fundamentals: Dict of fundamental DataFrames
            tickers: List of ticker symbols
            as_of_date: Calculation date

        Returns:
            Series of earnings yields
        """
        earnings = self._get_fundamental_values(
            fundamentals['earnings'], tickers, as_of_date
        )
        market_caps = self._get_fundamental_values(
            fundamentals['market_cap'], tickers, as_of_date
        )

        # E/P = Earnings / Market Cap
        # Handle division by zero
        earnings_yield = earnings / market_caps
        earnings_yield = earnings_yield.replace([np.inf, -np.inf], np.nan)

        # Filter out negative earnings (losses)
        # Note: Some strategies keep losses, but for simplicity we'll exclude
        earnings_yield[earnings <= 0] = np.nan

        # Cap extreme values
        earnings_yield = earnings_yield.clip(0, 1)

        valid_count = earnings_yield.notna().sum()
        logger.debug(
            f"{self.name}: Earnings Yield computed for {valid_count}/{len(tickers)} tickers"
        )

        return earnings_yield

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

    def _combine_signals(
        self,
        book_to_market: pd.Series,
        earnings_yield: pd.Series,
    ) -> pd.Series:
        """
        Combine book-to-market and earnings yield using z-scores.

        Args:
            book_to_market: Book-to-market ratios
            earnings_yield: Earnings yields

        Returns:
            Combined value scores
        """
        # Compute z-scores for each component
        bm_zscore = self._compute_zscore(book_to_market)
        ey_zscore = self._compute_zscore(earnings_yield)

        # Weighted combination
        combined = (
            self.weight_book_to_market * bm_zscore +
            self.weight_earnings_yield * ey_zscore
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

        if std == 0 or pd.isna(std):
            # All values identical - return zeros
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
            'fundamentals': ['book_value', 'market_cap', 'earnings']
        }

    def get_required_history_days(self) -> int:
        """
        Get minimum number of trading days required for this factor.

        Returns:
            Minimum trading days needed
        """
        # Value factors only need current fundamental data
        return 1


def compute_value(
    prices: pd.DataFrame,
    fundamentals: Dict[str, pd.DataFrame],
    as_of_date: datetime,
    weight_book_to_market: float = 0.5,
    weight_earnings_yield: float = 0.5,
) -> pd.Series:
    """
    Standalone function to compute value scores.

    This is a convenience function that can be used without instantiating
    the ValueFactor class.

    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        fundamentals: Dict with 'book_value', 'market_cap', 'earnings' DataFrames
        as_of_date: Calculation date
        weight_book_to_market: Weight for B/M component (default 0.5)
        weight_earnings_yield: Weight for E/Y component (default 0.5)

    Returns:
        Series with value scores

    Example:
        >>> value_scores = compute_value(
        ...     prices,
        ...     fundamentals,
        ...     datetime(2024, 1, 31),
        ...     weight_book_to_market=0.5,
        ...     weight_earnings_yield=0.5
        ... )
    """
    # Create temporary config
    config = {
        'weight_book_to_market': weight_book_to_market,
        'weight_earnings_yield': weight_earnings_yield,
        'enabled': True,
        'weight': 1.0
    }

    # Use class implementation
    factor = ValueFactor(config)
    return factor.compute(prices, fundamentals, as_of_date)
