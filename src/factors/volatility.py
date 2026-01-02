"""
Volatility factor implementation (60-day realized volatility, inverted).

This module implements the low volatility anomaly: stocks with lower historical
volatility tend to outperform high volatility stocks on a risk-adjusted basis.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.factors.base import BaseFactor
from src.utils.logging_config import get_logger
from src.utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class VolatilityFactor(BaseFactor):
    """
    Volatility factor based on realized historical volatility (inverted).

    The low volatility anomaly suggests that lower risk stocks tend to
    outperform higher risk stocks. This factor computes historical
    volatility and inverts it (negative sign) so that lower volatility
    stocks receive higher scores.

    Formula:
        returns = log(price[t] / price[t-1])
        volatility = std(returns) * sqrt(252)  # Annualized
        score = -volatility  # Inverted (lower vol = higher score)

    Attributes:
        lookback_days: Number of trading days for volatility calculation
        annualize: Whether to annualize the volatility

    Example:
        >>> config = {
        ...     'lookback_days': 60,
        ...     'annualize': True,
        ...     'enabled': True
        ... }
        >>> volatility = VolatilityFactor(config)
        >>> scores = volatility.compute(prices, {}, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 low-volatility stocks
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize volatility factor from configuration.

        Args:
            config: Configuration dict with keys:
                - lookback_days: Trading days for vol calculation (default 60)
                - annualize: Whether to annualize volatility (default True)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)

        Example:
            >>> config = {
            ...     'lookback_days': 60,
            ...     'annualize': True,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = VolatilityFactor(config)
        """
        # Extract configuration
        self.lookback_days = config.get('lookback_days', 60)
        self.annualize = config.get('annualize', True)
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)

        # Validate lookback
        if self.lookback_days < 5:
            raise ValueError(f"lookback_days must be >= 5, got {self.lookback_days}")

        # Initialize base class
        super().__init__(name="volatility", enabled=enabled, weight=weight)

        logger.info(
            f"Initialized {self.name}: lookback={self.lookback_days} days, "
            f"annualize={self.annualize}"
        )

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute volatility scores for all tickers (inverted).

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Not used for volatility (price-based only)
            as_of_date: Calculation date

        Returns:
            Series with volatility scores (higher = lower volatility)

        Raises:
            InsufficientDataError: If less than required history
            DataError: If data format invalid
        """
        # Validate inputs
        self._validate_prices(prices)
        as_of_date = self._validate_date_in_range(prices, as_of_date)

        # Check sufficient history (need lookback + 1 for returns)
        required_days = self.lookback_days + 10  # Extra buffer
        self._check_sufficient_history(prices, as_of_date, required_days)

        logger.info(
            f"Computing {self.name} for {len(prices.columns)} tickers "
            f"as of {as_of_date.date()}"
        )

        try:
            # Get historical prices for volatility calculation
            historical = prices.loc[:as_of_date]
            window_prices = historical.iloc[-(self.lookback_days + 1):]

            # Compute log returns
            returns = np.log(window_prices / window_prices.shift(1))
            returns = returns.iloc[1:]  # Drop first NaN row

            # Compute volatility (standard deviation of returns)
            volatility = returns.std()

            # Annualize if requested
            if self.annualize:
                volatility = volatility * np.sqrt(252)

            # Invert: lower volatility = higher score
            scores = -volatility

            # Handle edge cases
            scores = self._handle_edge_cases(scores, returns)

            # Log statistics
            valid_scores = scores.dropna()
            if len(valid_scores) > 0:
                # Note: scores are negative, so "max" is least negative (lowest vol)
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(scores)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")

            return scores

        except Exception as e:
            logger.error(f"{self.name}: Error computing volatility: {e}")
            raise DataError(f"Volatility computation failed: {e}")

    def _handle_edge_cases(self, scores: pd.Series, returns: pd.DataFrame) -> pd.Series:
        """
        Handle edge cases in volatility calculation.

        Args:
            scores: Raw volatility scores (inverted)
            returns: Return series used for calculation

        Returns:
            Cleaned volatility scores
        """
        # Handle tickers with insufficient data
        valid_returns = returns.notna().sum()
        min_required = max(5, self.lookback_days // 2)

        insufficient_mask = valid_returns < min_required
        if insufficient_mask.any():
            insufficient_tickers = scores.index[insufficient_mask].tolist()
            logger.warning(
                f"{self.name}: {len(insufficient_tickers)} tickers have insufficient "
                f"returns data (<{min_required} days), setting to NaN: "
                f"{insufficient_tickers[:5]}"
            )
            scores[insufficient_mask] = np.nan

        # Handle zero volatility (constant prices - suspicious)
        zero_vol_mask = (scores == 0) | scores.isna()
        if zero_vol_mask.any():
            zero_tickers = scores.index[zero_vol_mask].tolist()
            if len(zero_tickers) > 0:
                logger.debug(
                    f"{self.name}: {len(zero_tickers)} tickers have zero/NaN volatility: "
                    f"{zero_tickers[:5]}"
                )

        return scores

    def required_data(self) -> Dict[str, List[str]]:
        """
        Specify required data fields.

        Returns:
            Dict with 'prices' requiring adjusted close, no fundamentals needed
        """
        return {
            'prices': ['Close'],  # Adjusted close
            'fundamentals': []    # No fundamental data needed
        }

    def get_required_history_days(self) -> int:
        """
        Get minimum number of trading days required for this factor.

        Returns:
            Minimum trading days needed
        """
        return self.lookback_days + 10  # Buffer for robustness


def compute_volatility(
    prices: pd.DataFrame,
    as_of_date: datetime,
    lookback_days: int = 60,
    annualize: bool = True,
) -> pd.Series:
    """
    Standalone function to compute volatility scores.

    This is a convenience function that can be used without instantiating
    the VolatilityFactor class.

    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        as_of_date: Calculation date
        lookback_days: Trading days for volatility (default 60)
        annualize: Whether to annualize (default True)

    Returns:
        Series with volatility scores (higher = lower volatility)

    Example:
        >>> volatility_scores = compute_volatility(
        ...     prices,
        ...     datetime(2024, 1, 31),
        ...     lookback_days=60,
        ...     annualize=True
        ... )
    """
    # Create temporary config
    config = {
        'lookback_days': lookback_days,
        'annualize': annualize,
        'enabled': True,
        'weight': 1.0
    }

    # Use class implementation
    factor = VolatilityFactor(config)
    return factor.compute(prices, {}, as_of_date)
