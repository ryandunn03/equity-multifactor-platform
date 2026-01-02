"""
Quality factor implementation (ROE + Profit Margin + Leverage).

This module implements the quality anomaly: companies with strong profitability,
efficient operations, and conservative capital structure tend to outperform.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.factors.base import BaseFactor
from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class QualityFactor(BaseFactor):
    """
    Quality factor combining ROE, Profit Margin, and Leverage.

    The quality anomaly suggests that companies with high profitability,
    efficient operations, and low debt outperform over time. This combines:
    - ROE (Return on Equity): net_income / equity
    - Profit Margin: net_income / revenue
    - Leverage: debt / equity (inverted - lower is better)

    Formula:
        roe = net_income / equity
        profit_margin = net_income / revenue
        leverage = debt / equity (inverted in scoring)
        quality_score = w_roe * roe_z + w_pm * pm_z - w_lev * lev_z

    Attributes:
        weight_roe: Weight for ROE component (0-1)
        weight_profit_margin: Weight for profit margin component (0-1)
        weight_leverage: Weight for leverage component (0-1)

    Example:
        >>> config = {
        ...     'weight_roe': 0.4,
        ...     'weight_profit_margin': 0.3,
        ...     'weight_leverage': 0.3,
        ...     'enabled': True
        ... }
        >>> quality = QualityFactor(config)
        >>> scores = quality.compute(prices, fundamentals, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 quality stocks
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize quality factor from configuration.

        Args:
            config: Configuration dict with keys:
                - weight_roe: Weight for ROE (default 0.4)
                - weight_profit_margin: Weight for profit margin (default 0.3)
                - weight_leverage: Weight for leverage (default 0.3)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)

        Example:
            >>> config = {
            ...     'weight_roe': 0.4,
            ...     'weight_profit_margin': 0.3,
            ...     'weight_leverage': 0.3,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = QualityFactor(config)
        """
        # Extract configuration
        self.weight_roe = config.get('weight_roe', 0.4)
        self.weight_profit_margin = config.get('weight_profit_margin', 0.3)
        self.weight_leverage = config.get('weight_leverage', 0.3)
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)

        # Validate weights
        if not (0 <= self.weight_roe <= 1):
            raise ValueError(f"weight_roe must be in [0,1], got {self.weight_roe}")
        if not (0 <= self.weight_profit_margin <= 1):
            raise ValueError(f"weight_profit_margin must be in [0,1], got {self.weight_profit_margin}")
        if not (0 <= self.weight_leverage <= 1):
            raise ValueError(f"weight_leverage must be in [0,1], got {self.weight_leverage}")

        # Normalize weights to sum to 1
        total_weight = self.weight_roe + self.weight_profit_margin + self.weight_leverage
        if total_weight == 0:
            raise ValueError("At least one component weight must be non-zero")

        self.weight_roe /= total_weight
        self.weight_profit_margin /= total_weight
        self.weight_leverage /= total_weight

        # Initialize base class
        super().__init__(name="quality", enabled=enabled, weight=weight)

        logger.info(
            f"Initialized {self.name}: ROE weight={self.weight_roe:.2f}, "
            f"PM weight={self.weight_profit_margin:.2f}, "
            f"Leverage weight={self.weight_leverage:.2f}"
        )

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute quality scores for all tickers.

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Dict with 'net_income', 'equity', 'revenue', 'debt' DataFrames
            as_of_date: Calculation date

        Returns:
            Series with quality scores (higher = higher quality)

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

            # Compute ROE
            roe = self._compute_roe(fundamentals, tickers, as_of_date)

            # Compute Profit Margin
            profit_margin = self._compute_profit_margin(fundamentals, tickers, as_of_date)

            # Compute Leverage
            leverage = self._compute_leverage(fundamentals, tickers, as_of_date)

            # Combine signals using z-scores
            quality_scores = self._combine_signals(roe, profit_margin, leverage)

            # Log statistics
            valid_scores = quality_scores.dropna()
            if len(valid_scores) > 0:
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(quality_scores)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")

            return quality_scores

        except InsufficientDataError:
            # Re-raise InsufficientDataError as-is (don't wrap it)
            raise
        except Exception as e:
            logger.error(f"{self.name}: Error computing quality: {e}")
            raise DataError(f"Quality computation failed: {e}")

    def _validate_fundamentals(self, fundamentals: Dict[str, pd.DataFrame]) -> None:
        """
        Validate fundamental data format and required fields.

        Args:
            fundamentals: Dict of fundamental DataFrames

        Raises:
            DataError: If validation fails
        """
        required_fields = ['net_income', 'equity', 'revenue', 'debt']

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

    def _compute_roe(
        self,
        fundamentals: Dict[str, pd.DataFrame],
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute Return on Equity (ROE) for all tickers.

        Args:
            fundamentals: Dict of fundamental DataFrames
            tickers: List of ticker symbols
            as_of_date: Calculation date

        Returns:
            Series of ROE values
        """
        net_income = self._get_fundamental_values(
            fundamentals['net_income'], tickers, as_of_date
        )
        equity = self._get_fundamental_values(
            fundamentals['equity'], tickers, as_of_date
        )

        # ROE = Net Income / Equity
        # Handle division by zero
        roe = net_income / equity
        roe = roe.replace([np.inf, -np.inf], np.nan)

        # Filter out negative or zero equity (problematic)
        roe[equity <= 0] = np.nan

        # Cap extreme values (likely data errors)
        # ROE typically ranges from -50% to +50%, but can be higher
        roe = roe.clip(-1.0, 2.0)

        valid_count = roe.notna().sum()
        logger.debug(
            f"{self.name}: ROE computed for {valid_count}/{len(tickers)} tickers"
        )

        return roe

    def _compute_profit_margin(
        self,
        fundamentals: Dict[str, pd.DataFrame],
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute profit margin for all tickers.

        Args:
            fundamentals: Dict of fundamental DataFrames
            tickers: List of ticker symbols
            as_of_date: Calculation date

        Returns:
            Series of profit margin values
        """
        net_income = self._get_fundamental_values(
            fundamentals['net_income'], tickers, as_of_date
        )
        revenue = self._get_fundamental_values(
            fundamentals['revenue'], tickers, as_of_date
        )

        # Profit Margin = Net Income / Revenue
        # Handle division by zero
        profit_margin = net_income / revenue
        profit_margin = profit_margin.replace([np.inf, -np.inf], np.nan)

        # Filter out zero or negative revenue (invalid)
        profit_margin[revenue <= 0] = np.nan

        # Cap extreme values
        # Profit margins typically range from -50% to +50%
        profit_margin = profit_margin.clip(-1.0, 1.0)

        valid_count = profit_margin.notna().sum()
        logger.debug(
            f"{self.name}: Profit Margin computed for {valid_count}/{len(tickers)} tickers"
        )

        return profit_margin

    def _compute_leverage(
        self,
        fundamentals: Dict[str, pd.DataFrame],
        tickers: List[str],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute leverage (Debt/Equity) for all tickers.

        Note: Higher leverage is worse for quality, so this will be
        subtracted in the final score (inverted).

        Args:
            fundamentals: Dict of fundamental DataFrames
            tickers: List of ticker symbols
            as_of_date: Calculation date

        Returns:
            Series of leverage values
        """
        debt = self._get_fundamental_values(
            fundamentals['debt'], tickers, as_of_date
        )
        equity = self._get_fundamental_values(
            fundamentals['equity'], tickers, as_of_date
        )

        # Leverage = Debt / Equity
        # Handle division by zero
        leverage = debt / equity
        leverage = leverage.replace([np.inf, -np.inf], np.nan)

        # Filter out negative equity
        leverage[equity <= 0] = np.nan

        # Cap extreme values
        # Leverage typically ranges from 0 to 5x
        leverage = leverage.clip(0, 10.0)

        valid_count = leverage.notna().sum()
        logger.debug(
            f"{self.name}: Leverage computed for {valid_count}/{len(tickers)} tickers"
        )

        return leverage

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
        roe: pd.Series,
        profit_margin: pd.Series,
        leverage: pd.Series,
    ) -> pd.Series:
        """
        Combine ROE, profit margin, and leverage using z-scores.

        Note: Leverage is subtracted (inverted) since lower is better.

        Args:
            roe: Return on equity values
            profit_margin: Profit margin values
            leverage: Leverage (debt/equity) values

        Returns:
            Combined quality scores
        """
        # Compute z-scores for each component
        roe_zscore = self._compute_zscore(roe)
        pm_zscore = self._compute_zscore(profit_margin)
        lev_zscore = self._compute_zscore(leverage)

        # Weighted combination (leverage is subtracted - lower is better)
        combined = (
            self.weight_roe * roe_zscore +
            self.weight_profit_margin * pm_zscore -
            self.weight_leverage * lev_zscore  # Inverted
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
            'fundamentals': ['net_income', 'equity', 'revenue', 'debt']
        }

    def get_required_history_days(self) -> int:
        """
        Get minimum number of trading days required for this factor.

        Returns:
            Minimum trading days needed
        """
        # Quality factors only need current fundamental data
        return 1


def compute_quality(
    prices: pd.DataFrame,
    fundamentals: Dict[str, pd.DataFrame],
    as_of_date: datetime,
    weight_roe: float = 0.4,
    weight_profit_margin: float = 0.3,
    weight_leverage: float = 0.3,
) -> pd.Series:
    """
    Standalone function to compute quality scores.

    This is a convenience function that can be used without instantiating
    the QualityFactor class.

    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        fundamentals: Dict with 'net_income', 'equity', 'revenue', 'debt' DataFrames
        as_of_date: Calculation date
        weight_roe: Weight for ROE component (default 0.4)
        weight_profit_margin: Weight for profit margin component (default 0.3)
        weight_leverage: Weight for leverage component (default 0.3)

    Returns:
        Series with quality scores

    Example:
        >>> quality_scores = compute_quality(
        ...     prices,
        ...     fundamentals,
        ...     datetime(2024, 1, 31),
        ...     weight_roe=0.4,
        ...     weight_profit_margin=0.3,
        ...     weight_leverage=0.3
        ... )
    """
    # Create temporary config
    config = {
        'weight_roe': weight_roe,
        'weight_profit_margin': weight_profit_margin,
        'weight_leverage': weight_leverage,
        'enabled': True,
        'weight': 1.0
    }

    # Use class implementation
    factor = QualityFactor(config)
    return factor.compute(prices, fundamentals, as_of_date)
