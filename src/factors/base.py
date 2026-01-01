"""
Abstract base class for all factor implementations.

This module defines the interface that all concrete factors must implement,
along with common validation and helper methods.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class BaseFactor(ABC):
    """
    Abstract base class for all equity factors.
    
    All concrete factor implementations must inherit from this class and
    implement the compute() and required_data() methods.
    
    Attributes:
        name: Factor identifier (e.g., "momentum", "value")
        enabled: Whether this factor is active in the strategy
        weight: Base weight for equal-weighted combination
    
    Example:
        >>> class MyFactor(BaseFactor):
        ...     def __init__(self, config):
        ...         super().__init__(name="my_factor", enabled=config.get('enabled', True))
        ...         
        ...     def compute(self, prices, fundamentals, as_of_date):
        ...         # Implementation here
        ...         return scores
        ...     
        ...     def required_data(self):
        ...         return {'prices': ['Close'], 'fundamentals': []}
    """
    
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize base factor.
        
        Args:
            name: Factor identifier
            enabled: Whether factor is active
            weight: Base weight for combination (default 1.0)
        """
        self.name = name
        self.enabled = enabled
        self.weight = weight
        
        logger.debug(f"Initialized {self.name} factor (enabled={enabled}, weight={weight})")
    
    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute factor scores for all tickers as of a specific date.
        
        This method must be implemented by all concrete factor classes.
        It should return raw factor scores (not normalized).
        
        Args:
            prices: DataFrame with DatetimeIndex (rows) and ticker symbols (columns)
                   containing adjusted close prices
            fundamentals: Dict mapping ticker symbols to DataFrames containing
                         fundamental data (income statement, balance sheet, etc.)
            as_of_date: Point-in-time date for calculation (must respect data availability)
        
        Returns:
            Series with ticker symbols as index and raw factor scores as values.
            Higher scores should indicate more attractive stocks (consistent orientation).
        
        Raises:
            InsufficientDataError: If not enough historical data available
            DataError: If data format is invalid or required fields missing
        
        Example:
            >>> prices = get_prices(['AAPL', 'MSFT'], start, end)
            >>> factor = MomentumFactor(config)
            >>> scores = factor.compute(prices, {}, datetime(2024, 1, 31))
            >>> print(scores.head())
            AAPL    0.234
            MSFT    0.156
            dtype: float64
        """
        pass
    
    @abstractmethod
    def required_data(self) -> Dict[str, List[str]]:
        """
        Specify required data fields for this factor.
        
        Returns:
            Dictionary with 'prices' and 'fundamentals' keys, each containing
            a list of required field names.
        
        Example:
            >>> factor = ValueFactor(config)
            >>> factor.required_data()
            {
                'prices': ['Close'],
                'fundamentals': ['totalStockholdersEquity', 'marketCap']
            }
        """
        pass
    
    def _validate_prices(self, prices: pd.DataFrame) -> None:
        """
        Validate price DataFrame format and quality.
        
        Args:
            prices: Price DataFrame to validate
        
        Raises:
            DataError: If validation fails
        """
        if not isinstance(prices, pd.DataFrame):
            raise DataError(f"Prices must be DataFrame, got {type(prices)}")
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise DataError(
                f"Price index must be DatetimeIndex, got {type(prices.index)}. "
                "Ensure prices have dates as index."
            )
        
        if not prices.index.is_monotonic_increasing:
            logger.warning(f"{self.name}: Price index not sorted, sorting now")
            prices.sort_index(inplace=True)
        
        if prices.empty:
            raise DataError(f"{self.name}: Price DataFrame is empty")
        
        # Check for columns with all NaN
        all_nan_cols = prices.columns[prices.isna().all()].tolist()
        if all_nan_cols:
            logger.warning(
                f"{self.name}: Columns with all NaN values: {all_nan_cols}. "
                "These tickers will be excluded."
            )
        
        logger.debug(
            f"{self.name}: Validated prices - "
            f"{len(prices)} dates, {len(prices.columns)} tickers"
        )
    
    def _validate_date_in_range(
        self,
        prices: pd.DataFrame,
        as_of_date: datetime,
        tolerance_days: int = 5,
    ) -> datetime:
        """
        Check if as_of_date exists in price index and find closest if needed.
        
        Args:
            prices: Price DataFrame
            as_of_date: Requested calculation date
            tolerance_days: Maximum days to search for nearest date
        
        Returns:
            Actual date to use (may be adjusted to nearest business day)
        
        Raises:
            DataError: If date not found within tolerance
        """
        if as_of_date in prices.index:
            return as_of_date
        
        # Try to find nearest business day
        min_date = prices.index.min()
        max_date = prices.index.max()
        
        if as_of_date < min_date:
            raise DataError(
                f"{self.name}: as_of_date {as_of_date.date()} is before "
                f"available data start {min_date.date()}"
            )
        
        if as_of_date > max_date:
            raise DataError(
                f"{self.name}: as_of_date {as_of_date.date()} is after "
                f"available data end {max_date.date()}"
            )
        
        # Find nearest date within tolerance
        date_diff = abs(prices.index - as_of_date)
        nearest_idx = date_diff.argmin()
        nearest_date = prices.index[nearest_idx]
        days_away = date_diff.min().days
        
        if days_away > tolerance_days:
            raise DataError(
                f"{self.name}: Nearest date to {as_of_date.date()} is "
                f"{nearest_date.date()} ({days_away} days away), "
                f"exceeds tolerance of {tolerance_days} days"
            )
        
        if days_away > 0:
            logger.debug(
                f"{self.name}: Using {nearest_date.date()} instead of "
                f"{as_of_date.date()} ({days_away} days difference)"
            )
        
        return nearest_date
    
    def _check_sufficient_history(
        self,
        prices: pd.DataFrame,
        as_of_date: datetime,
        required_days: int,
    ) -> None:
        """
        Verify sufficient historical data exists before as_of_date.
        
        Args:
            prices: Price DataFrame
            as_of_date: Calculation date
            required_days: Minimum number of trading days required
        
        Raises:
            InsufficientDataError: If insufficient history available
        """
        # Get all dates up to and including as_of_date
        historical_data = prices.loc[:as_of_date]
        available_days = len(historical_data)
        
        if available_days < required_days:
            raise InsufficientDataError(
                f"{self.name}: Requires {required_days} trading days of history, "
                f"but only {available_days} days available before {as_of_date.date()}. "
                f"Earliest usable date would be approximately "
                f"{prices.index[0] + timedelta(days=required_days * 1.4)}"
            )
        
        logger.debug(
            f"{self.name}: Sufficient history - {available_days} days available "
            f"(required: {required_days})"
        )
    
    def _get_price_at_offset(
        self,
        prices: pd.DataFrame,
        as_of_date: datetime,
        offset_days: int,
    ) -> pd.Series:
        """
        Get prices at a specific number of trading days before as_of_date.
        
        Args:
            prices: Price DataFrame
            as_of_date: Reference date
            offset_days: Number of trading days to look back (positive number)
        
        Returns:
            Series of prices at the offset date
        
        Raises:
            DataError: If offset date not available
        """
        historical = prices.loc[:as_of_date]
        
        if len(historical) < offset_days + 1:
            raise DataError(
                f"{self.name}: Cannot get prices {offset_days} days back from "
                f"{as_of_date.date()}, insufficient data"
            )
        
        # Get price at offset (iloc is 0-indexed, so subtract 1)
        offset_price = historical.iloc[-(offset_days + 1)]
        
        return offset_price
    
    def __repr__(self) -> str:
        """String representation of factor."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"enabled={self.enabled}, weight={self.weight})"
        )