"""
Momentum factor implementation (12-1 month returns).

This module implements the momentum anomaly: stocks with strong past performance
tend to continue outperforming in the near term.
"""

from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from .base import BaseFactor
from ..utils.logging_config import get_logger
from ..utils.exceptions import DataError, InsufficientDataError

logger = get_logger(__name__)


class MomentumFactor(BaseFactor):
    """
    Momentum factor based on trailing returns, skipping the most recent month.
    
    The classic momentum signal is defined as the cumulative return from
    t-252 trading days to t-21 trading days (12-1 month momentum). This
    skip avoids short-term reversal effects.
    
    Formula:
        momentum = (price[t-21] / price[t-252]) - 1
    
    Attributes:
        lookback_months: Number of months to look back (default 12)
        skip_months: Number of recent months to skip (default 1)
        lookback_days: Trading days equivalent of lookback_months
        skip_days: Trading days equivalent of skip_months
    
    Example:
        >>> config = {'lookback_months': 12, 'skip_months': 1, 'enabled': True}
        >>> momentum = MomentumFactor(config)
        >>> scores = momentum.compute(prices, {}, datetime(2024, 1, 31))
        >>> print(scores.nlargest(5))  # Top 5 momentum stocks
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize momentum factor from configuration.
        
        Args:
            config: Configuration dict with keys:
                - lookback_months: Months to look back (default 12)
                - skip_months: Recent months to skip (default 1)
                - enabled: Whether factor is active (default True)
                - weight: Factor weight for combination (default 1.0)
        
        Example:
            >>> config = {
            ...     'lookback_months': 12,
            ...     'skip_months': 1,
            ...     'enabled': True,
            ...     'weight': 1.0
            ... }
            >>> factor = MomentumFactor(config)
        """
        # Extract configuration
        self.lookback_months = config.get('lookback_months', 12)
        self.skip_months = config.get('skip_months', 1)
        enabled = config.get('enabled', True)
        weight = config.get('weight', 1.0)
        
        # Convert months to approximate trading days (21 per month)
        self.lookback_days = self.lookback_months * 21
        self.skip_days = self.skip_months * 21
        
        # Initialize base class
        super().__init__(name="momentum", enabled=enabled, weight=weight)
        
        logger.info(
            f"Initialized {self.name}: lookback={self.lookback_months}m "
            f"({self.lookback_days}d), skip={self.skip_months}m ({self.skip_days}d)"
        )
    
    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.Series:
        """
        Compute momentum scores for all tickers.
        
        Args:
            prices: DataFrame with DatetimeIndex and ticker columns
            fundamentals: Not used for momentum (price-based only)
            as_of_date: Calculation date
        
        Returns:
            Series with momentum scores (higher = stronger momentum)
        
        Raises:
            InsufficientDataError: If less than required history
            DataError: If data format invalid
        """
        # Validate inputs
        self._validate_prices(prices)
        as_of_date = self._validate_date_in_range(prices, as_of_date)
        
        # Check sufficient history (need lookback + skip + buffer)
        required_days = self.lookback_days + 30  # 30-day buffer for safety
        self._check_sufficient_history(prices, as_of_date, required_days)
        
        logger.info(
            f"Computing {self.name} for {len(prices.columns)} tickers "
            f"as of {as_of_date.date()}"
        )
        
        try:
            # Get prices at the two required dates
            p_recent = self._get_price_at_offset(prices, as_of_date, self.skip_days)
            p_old = self._get_price_at_offset(prices, as_of_date, self.lookback_days)
            
            # Compute momentum: (p_recent / p_old) - 1
            momentum = (p_recent / p_old) - 1.0
            
            # Handle edge cases
            momentum = self._handle_edge_cases(momentum, p_old)
            
            # Log statistics
            valid_scores = momentum.dropna()
            if len(valid_scores) > 0:
                logger.info(
                    f"{self.name}: Computed scores for {len(valid_scores)}/{len(momentum)} tickers. "
                    f"Range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}], "
                    f"Mean: {valid_scores.mean():.3f}, Median: {valid_scores.median():.3f}"
                )
            else:
                logger.warning(f"{self.name}: No valid scores computed (all NaN)")
            
            return momentum
            
        except Exception as e:
            logger.error(f"{self.name}: Error computing momentum: {e}")
            raise DataError(f"Momentum computation failed: {e}")
    
    def _handle_edge_cases(self, momentum: pd.Series, p_old: pd.Series) -> pd.Series:
        """
        Handle edge cases in momentum calculation.
        
        Args:
            momentum: Raw momentum scores
            p_old: Prices at lookback date
        
        Returns:
            Cleaned momentum scores with edge cases handled
        """
        # Handle division by zero (price = 0)
        zero_price_mask = (p_old == 0) | (p_old.isna())
        if zero_price_mask.any():
            zero_tickers = momentum.index[zero_price_mask].tolist()
            logger.warning(
                f"{self.name}: {len(zero_tickers)} tickers have zero/NaN price at "
                f"lookback date, setting momentum to NaN: {zero_tickers[:5]}"
            )
            momentum[zero_price_mask] = np.nan
        
        # Cap extreme values (optional safety check)
        # Momentum >1000% is likely data error
        extreme_mask = momentum.abs() > 10.0
        if extreme_mask.any():
            extreme_tickers = momentum.index[extreme_mask].tolist()
            logger.warning(
                f"{self.name}: {len(extreme_tickers)} tickers have extreme momentum "
                f"(>1000%), capping values: {extreme_tickers[:5]}"
            )
            momentum = momentum.clip(-10.0, 10.0)
        
        return momentum
    
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
        return self.lookback_days + 30  # Buffer for robustness


def compute_momentum(
    prices: pd.DataFrame,
    as_of_date: datetime,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.Series:
    """
    Standalone function to compute momentum scores.
    
    This is a convenience function that can be used without instantiating
    the MomentumFactor class.
    
    Args:
        prices: DataFrame with DatetimeIndex and ticker columns
        as_of_date: Calculation date
        lookback_days: Trading days to look back (default 252 ≈ 12 months)
        skip_days: Recent days to skip (default 21 ≈ 1 month)
    
    Returns:
        Series with momentum scores
    
    Example:
        >>> momentum_scores = compute_momentum(
        ...     prices,
        ...     datetime(2024, 1, 31),
        ...     lookback_days=252,
        ...     skip_days=21
        ... )
    """
    # Create temporary config
    config = {
        'lookback_months': lookback_days // 21,
        'skip_months': skip_days // 21,
        'enabled': True,
        'weight': 1.0
    }
    
    # Use class implementation
    factor = MomentumFactor(config)
    return factor.compute(prices, {}, as_of_date)