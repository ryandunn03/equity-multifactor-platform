"""
Factor combination using equal-weighting or IC-weighted with shrinkage.

This module combines multiple normalized factors into a composite score,
with support for Information Coefficient (IC) based weighting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from ..utils.logging_config import get_logger
from ..utils.exceptions import DataError

logger = get_logger(__name__)


class FactorCompositor:
    """
    Combine multiple factors using equal-weighting or IC-weighting.
    
    IC-weighting uses historical Information Coefficients (rank correlation
    between factor scores and forward returns) to weight factors based on
    their predictive power. Shrinkage toward equal-weight prevents overfitting.
    
    Attributes:
        method: Combination method ('equal_weighted' or 'ic_weighted')
        ic_lookback: Number of months of IC history to use
        shrinkage: Shrinkage toward equal-weight (0=pure IC, 1=equal)
        min_ic: Minimum IC threshold to include factor
        ic_history: Dict storing IC time series per factor
    
    Example:
        >>> compositor = FactorCompositor(config['factor_combination'])
        >>> composite = compositor.combine(normalized_scores)
        >>> # Update IC history after each period
        >>> compositor.update_ic_history(scores, forward_returns, date)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize compositor from configuration.
        
        Args:
            config: Configuration dict with keys:
                - method: 'equal_weighted' or 'ic_weighted'
                - ic_lookback_months: Months of IC history (default 12)
                - shrinkage: Shrinkage parameter 0-1 (default 0.4)
                - min_ic: Minimum IC to include factor (default -0.5)
        """
        self.method = config.get('method', 'equal_weighted')
        self.ic_lookback = config.get('ic_lookback_months', 12)
        self.shrinkage = config.get('shrinkage', 0.4)
        self.min_ic = config.get('min_ic', -0.5)
        
        # Store IC history: {factor_name: [ic_1, ic_2, ...]}
        self.ic_history: Dict[str, List[float]] = {}
        
        logger.info(
            f"Initialized FactorCompositor: method={self.method}, "
            f"ic_lookback={self.ic_lookback}m, shrinkage={self.shrinkage}"
        )
    
    def combine(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Combine multiple factors into composite score.
        
        Args:
            factor_scores: DataFrame with tickers as index, factors as columns
                          (should be normalized)
            forward_returns: Optional forward returns for IC calculation
                           (not needed if method='equal_weighted')
        
        Returns:
            Series with composite scores per ticker
        
        Example:
            >>> scores = pd.DataFrame({
            ...     'momentum': [0.5, -0.3, 1.2],
            ...     'value': [-0.2, 0.8, 0.1]
            ... }, index=['AAPL', 'MSFT', 'GOOGL'])
            >>> composite = compositor.combine(scores)
        """
        if factor_scores.empty:
            logger.warning("Empty factor_scores DataFrame, returning empty Series")
            return pd.Series(dtype=float)
        
        if len(factor_scores.columns) == 0:
            raise DataError("factor_scores has no factor columns")
        
        logger.info(
            f"Combining {len(factor_scores.columns)} factors for "
            f"{len(factor_scores)} tickers using {self.method}"
        )
        
        if self.method == 'equal_weighted':
            composite = self._combine_equal_weighted(factor_scores)
        elif self.method == 'ic_weighted':
            composite = self._combine_ic_weighted(factor_scores)
        else:
            raise DataError(f"Unknown combination method: {self.method}")
        
        # Log composite statistics
        logger.info(
            f"Composite scores: mean={composite.mean():.3f}, "
            f"std={composite.std():.3f}, range=[{composite.min():.3f}, {composite.max():.3f}]"
        )
        
        return composite
    
    def _combine_equal_weighted(self, factor_scores: pd.DataFrame) -> pd.Series:
        """
        Simple average across all factors.
        
        Args:
            factor_scores: Normalized factor scores
        
        Returns:
            Equal-weighted composite score
        """
        # Average across columns (factors)
        composite = factor_scores.mean(axis=1, skipna=True)
        
        logger.debug(
            f"Equal-weighted combination: {len(factor_scores.columns)} factors, "
            f"weight={1.0/len(factor_scores.columns):.3f} each"
        )
        
        return composite
    
    def _combine_ic_weighted(self, factor_scores: pd.DataFrame) -> pd.Series:
        """
        IC-weighted combination with shrinkage toward equal-weight.
        
        Args:
            factor_scores: Normalized factor scores
        
        Returns:
            IC-weighted composite score
        """
        # Check if we have sufficient IC history
        if not self._has_sufficient_ic_history():
            logger.warning(
                f"Insufficient IC history (<{self.ic_lookback} periods), "
                "falling back to equal-weighting"
            )
            return self._combine_equal_weighted(factor_scores)
        
        # Compute IC-based weights
        ic_weights = self._compute_ic_weights(factor_scores.columns.tolist())
        
        # Apply shrinkage toward equal-weight
        n_factors = len(factor_scores.columns)
        equal_weight = 1.0 / n_factors
        
        final_weights = {}
        for factor in factor_scores.columns:
            ic_w = ic_weights.get(factor, equal_weight)
            shrunk_w = self.shrinkage * equal_weight + (1 - self.shrinkage) * ic_w
            final_weights[factor] = shrunk_w
        
        # Normalize weights to sum to 1
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        # Log weights
        logger.info(f"IC-weighted factors: {final_weights}")
        
        # Compute weighted combination
        composite = pd.Series(0.0, index=factor_scores.index)
        for factor, weight in final_weights.items():
            composite += factor_scores[factor] * weight
        
        return composite
    
    def _compute_ic_weights(self, factors: List[str]) -> Dict[str, float]:
        """
        Compute factor weights based on mean IC.
        
        Args:
            factors: List of factor names
        
        Returns:
            Dict mapping factor to weight
        """
        mean_ics = {}
        
        for factor in factors:
            if factor not in self.ic_history:
                logger.warning(f"No IC history for factor '{factor}', assigning 0")
                mean_ics[factor] = 0.0
                continue
            
            # Compute mean IC over lookback window
            ic_values = self.ic_history[factor][-self.ic_lookback:]
            mean_ic = np.mean(ic_values)
            
            # Filter negative predictors
            if mean_ic < self.min_ic:
                logger.warning(
                    f"Factor '{factor}' has mean IC={mean_ic:.3f} < {self.min_ic}, "
                    "setting weight to 0"
                )
                mean_ic = 0.0
            
            mean_ics[factor] = max(mean_ic, 0.0)  # Ensure non-negative
        
        # Normalize to sum to 1
        total_ic = sum(mean_ics.values())
        
        if total_ic == 0:
            logger.warning("All factors have IC=0, falling back to equal weights")
            n = len(factors)
            return {f: 1.0/n for f in factors}
        
        weights = {f: ic/total_ic for f, ic in mean_ics.items()}
        
        return weights
    
    def update_ic_history(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        as_of_date: datetime,
    ) -> None:
        """
        Update IC history with new observations.
        
        Args:
            factor_scores: Factor scores at time t
            forward_returns: Returns from t to t+1
            as_of_date: Date of observation
        
        Example:
            >>> # After each rebalance period
            >>> compositor.update_ic_history(
            ...     factor_scores_t,
            ...     returns_t_to_t1,
            ...     datetime(2024, 1, 31)
            ... )
        """
        if forward_returns is None or forward_returns.empty:
            logger.warning("No forward returns provided, skipping IC update")
            return
        
        # Align indices
        common_tickers = factor_scores.index.intersection(forward_returns.index)
        
        if len(common_tickers) < 10:
            logger.warning(
                f"Only {len(common_tickers)} common tickers for IC calculation, "
                "skipping update"
            )
            return
        
        factor_scores_aligned = factor_scores.loc[common_tickers]
        forward_returns_aligned = forward_returns.loc[common_tickers]
        
        # Compute IC for each factor
        for factor in factor_scores.columns:
            ic = self._compute_ic(
                factor_scores_aligned[factor],
                forward_returns_aligned,
            )
            
            # Initialize history if needed
            if factor not in self.ic_history:
                self.ic_history[factor] = []
            
            # Append new IC
            self.ic_history[factor].append(ic)
            
            # Maintain max length (FIFO)
            if len(self.ic_history[factor]) > self.ic_lookback * 2:
                self.ic_history[factor] = self.ic_history[factor][-self.ic_lookback:]
            
            logger.debug(
                f"Updated IC for '{factor}' on {as_of_date.date()}: "
                f"IC={ic:.3f}, history_length={len(self.ic_history[factor])}"
            )
    
    def _compute_ic(
        self,
        factor_scores: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """
        Compute Information Coefficient (Spearman rank correlation).
        
        Args:
            factor_scores: Factor scores at time t
            forward_returns: Returns from t to t+1
        
        Returns:
            IC value (correlation coefficient)
        """
        # Drop NaN values
        valid_mask = factor_scores.notna() & forward_returns.notna()
        
        if valid_mask.sum() < 5:
            logger.warning("Too few valid observations for IC calculation")
            return 0.0
        
        scores = factor_scores[valid_mask]
        returns = forward_returns[valid_mask]
        
        # Compute Spearman correlation
        try:
            ic, p_value = spearmanr(scores, returns)
            
            if np.isnan(ic):
                logger.warning("IC calculation returned NaN")
                return 0.0
            
            return ic
            
        except Exception as e:
            logger.error(f"Error computing IC: {e}")
            return 0.0
    
    def _has_sufficient_ic_history(self) -> bool:
        """
        Check if we have sufficient IC history for weighting.
        
        Returns:
            True if sufficient history available
        """
        if not self.ic_history:
            return False
        
        # Check if any factor has sufficient history
        min_length = min(len(ics) for ics in self.ic_history.values())
        
        return min_length >= self.ic_lookback
    
    def get_current_weights(self) -> pd.Series:
        """
        Get current factor weights based on IC history.
        
        Returns:
            Series with factor weights
        
        Example:
            >>> weights = compositor.get_current_weights()
            >>> print(weights)
            momentum    0.35
            value       0.28
            quality     0.37
            dtype: float64
        """
        if self.method == 'equal_weighted' or not self._has_sufficient_ic_history():
            # Return equal weights
            factors = list(self.ic_history.keys())
            if not factors:
                return pd.Series(dtype=float)
            
            n = len(factors)
            return pd.Series(1.0/n, index=factors)
        
        # Compute IC weights
        factors = list(self.ic_history.keys())
        ic_weights = self._compute_ic_weights(factors)
        
        # Apply shrinkage
        n = len(factors)
        equal_weight = 1.0 / n
        
        final_weights = {}
        for factor, ic_w in ic_weights.items():
            final_weights[factor] = (
                self.shrinkage * equal_weight + (1 - self.shrinkage) * ic_w
            )
        
        return pd.Series(final_weights)
    
    def get_ic_statistics(self) -> pd.DataFrame:
        """
        Get IC statistics for all factors.
        
        Returns:
            DataFrame with IC statistics (mean, std, t-stat, count)
        
        Example:
            >>> stats = compositor.get_ic_statistics()
            >>> print(stats)
                      mean   std  t_stat  count
            momentum  0.042  0.12    3.8     36
            value     0.038  0.13    3.1     36
            quality   0.051  0.11    4.7     36
        """
        if not self.ic_history:
            return pd.DataFrame()
        
        stats = []
        for factor, ic_values in self.ic_history.items():
            if len(ic_values) == 0:
                continue
            
            ic_array = np.array(ic_values)
            mean_ic = ic_array.mean()
            std_ic = ic_array.std()
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_values))) if std_ic > 0 else 0
            
            stats.append({
                'factor': factor,
                'mean': mean_ic,
                'std': std_ic,
                't_stat': t_stat,
                'count': len(ic_values),
            })
        
        return pd.DataFrame(stats).set_index('factor')