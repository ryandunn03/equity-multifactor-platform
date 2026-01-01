"""
Sector-relative factor normalization with winsorization.

This module provides sector-relative z-score normalization to remove
systematic biases across sectors while preserving within-sector signal.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from scipy.stats import zscore

from ..utils.logging_config import get_logger
from ..utils.exceptions import DataError

logger = get_logger(__name__)


class SectorNormalizer:
    """
    Normalize factor scores within sectors using z-score transformation.
    
    This eliminates cross-sectional biases where certain sectors systematically
    have higher/lower factor values (e.g., Tech has higher growth, Utilities
    have lower volatility).
    
    Process:
        1. Winsorize extreme values within each sector
        2. Z-score within each sector: (x - sector_mean) / sector_std
        3. Handle missing values per policy
        4. Handle small sectors (< 5 stocks)
    
    Attributes:
        method: Normalization method ('sector_relative' or 'cross_sectional')
        winsorize_pct: Percentile for winsorization (e.g., 1 = clip at 1st/99th)
        handle_missing: Policy for NaN values ('neutral', 'drop', 'forward_fill')
        max_missing: Max number of missing factors before dropping ticker
    
    Example:
        >>> normalizer = SectorNormalizer(config['normalization'])
        >>> normalized = normalizer.normalize(raw_scores, sector_mapping)
        >>> # Verify: mean ≈ 0, std ≈ 1 within each sector
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize normalizer from configuration.
        
        Args:
            config: Configuration dict with keys:
                - method: 'sector_relative' or 'cross_sectional'
                - winsorize_percentile: Clip percentile (1-50)
                - handle_missing: 'neutral', 'drop', or 'forward_fill'
                - max_missing_factors: Max missing factors to tolerate
        """
        self.method = config.get('method', 'sector_relative')
        self.winsorize_pct = config.get('winsorize_percentile', 1)
        self.handle_missing = config.get('handle_missing', 'neutral')
        self.max_missing = config.get('max_missing_factors', 2)
        
        logger.info(
            f"Initialized SectorNormalizer: method={self.method}, "
            f"winsorize={self.winsorize_pct}%, missing={self.handle_missing}"
        )
    
    def normalize(
        self,
        factor_scores: pd.DataFrame,
        sectors: pd.Series,
    ) -> pd.DataFrame:
        """
        Normalize factor scores within sectors.
        
        Args:
            factor_scores: DataFrame with tickers as index, factors as columns
            sectors: Series mapping ticker to sector (same index as factor_scores)
        
        Returns:
            DataFrame with normalized scores (mean≈0, std≈1 within each sector)
        
        Raises:
            DataError: If inputs are invalid or incompatible
        
        Example:
            >>> raw_scores = pd.DataFrame({
            ...     'momentum': [0.1, 0.2, -0.1, 0.3],
            ...     'value': [1.5, 2.0, 1.8, 1.2]
            ... }, index=['AAPL', 'MSFT', 'GOOGL', 'XOM'])
            >>> sectors = pd.Series({
            ...     'AAPL': 'Technology',
            ...     'MSFT': 'Technology',
            ...     'GOOGL': 'Technology',
            ...     'XOM': 'Energy'
            ... })
            >>> normalized = normalizer.normalize(raw_scores, sectors)
        """
        # Validate inputs
        self._validate_inputs(factor_scores, sectors)
        
        # Align sectors with factor scores
        sectors = sectors.reindex(factor_scores.index)
        
        logger.info(
            f"Normalizing {len(factor_scores.columns)} factors for "
            f"{len(factor_scores)} tickers across {sectors.nunique()} sectors"
        )
        
        if self.method == 'sector_relative':
            normalized = self._normalize_sector_relative(factor_scores, sectors)
        elif self.method == 'cross_sectional':
            normalized = self._normalize_cross_sectional(factor_scores)
        else:
            raise DataError(f"Unknown normalization method: {self.method}")
        
        # Handle missing values
        normalized = self._handle_missing_values(normalized, factor_scores)
        
        # Log statistics
        self._log_normalization_stats(factor_scores, normalized, sectors)
        
        return normalized
    
    def _validate_inputs(
        self,
        factor_scores: pd.DataFrame,
        sectors: pd.Series,
    ) -> None:
        """Validate input data."""
        if not isinstance(factor_scores, pd.DataFrame):
            raise DataError(f"factor_scores must be DataFrame, got {type(factor_scores)}")
        
        if not isinstance(sectors, pd.Series):
            raise DataError(f"sectors must be Series, got {type(sectors)}")
        
        if factor_scores.empty:
            raise DataError("factor_scores DataFrame is empty")
        
        # Check alignment
        missing_sectors = factor_scores.index.difference(sectors.index)
        if len(missing_sectors) > 0:
            logger.warning(
                f"{len(missing_sectors)} tickers missing sector mapping, "
                f"will assign 'Unknown': {list(missing_sectors)[:5]}"
            )
    
    def _normalize_sector_relative(
        self,
        factor_scores: pd.DataFrame,
        sectors: pd.Series,
    ) -> pd.DataFrame:
        """
        Normalize within each sector separately.
        
        Args:
            factor_scores: Raw factor scores
            sectors: Sector mapping
        
        Returns:
            Sector-relative normalized scores
        """
        normalized = pd.DataFrame(index=factor_scores.index, columns=factor_scores.columns)
        
        # Fill missing sectors
        sectors = sectors.fillna('Unknown')
        
        # Process each factor column
        for factor in factor_scores.columns:
            factor_normalized = pd.Series(index=factor_scores.index, dtype=float)
            
            # Process each sector
            for sector in sectors.unique():
                sector_mask = (sectors == sector)
                sector_tickers = factor_scores.index[sector_mask]
                
                if len(sector_tickers) == 0:
                    continue
                
                sector_values = factor_scores.loc[sector_tickers, factor]
                
                # Winsorize within sector
                sector_values_winsorized = self._winsorize(sector_values)
                
                # Z-score within sector
                sector_normalized = self._zscore_sector(
                    sector_values_winsorized,
                    sector,
                    factor,
                )
                
                factor_normalized.loc[sector_tickers] = sector_normalized
            
            normalized[factor] = factor_normalized
        
        return normalized
    
    def _normalize_cross_sectional(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize across entire universe (no sector adjustment).
        
        Args:
            factor_scores: Raw factor scores
        
        Returns:
            Cross-sectionally normalized scores
        """
        normalized = pd.DataFrame(index=factor_scores.index, columns=factor_scores.columns)
        
        for factor in factor_scores.columns:
            values = factor_scores[factor]
            
            # Winsorize
            values_winsorized = self._winsorize(values)
            
            # Z-score across all tickers
            values_normalized = self._zscore_sector(values_winsorized, 'All', factor)
            
            normalized[factor] = values_normalized
        
        return normalized
    
    def _winsorize(self, values: pd.Series) -> pd.Series:
        """
        Clip extreme values at specified percentiles.
        
        Args:
            values: Raw values to winsorize
        
        Returns:
            Winsorized values
        """
        if values.isna().all():
            return values
        
        lower_pct = self.winsorize_pct
        upper_pct = 100 - self.winsorize_pct
        
        lower_bound = values.quantile(lower_pct / 100, interpolation='linear')
        upper_bound = values.quantile(upper_pct / 100, interpolation='linear')
        
        winsorized = values.clip(lower=lower_bound, upper=upper_bound)
        
        # Count clipped values
        n_clipped = ((values < lower_bound) | (values > upper_bound)).sum()
        if n_clipped > 0:
            logger.debug(
                f"Winsorized {n_clipped}/{len(values)} values at "
                f"[{lower_bound:.3f}, {upper_bound:.3f}]"
            )
        
        return winsorized
    
    def _zscore_sector(
        self,
        values: pd.Series,
        sector: str,
        factor: str,
    ) -> pd.Series:
        """
        Compute z-score with handling for small sectors.
        
        Args:
            values: Values to normalize
            sector: Sector name (for logging)
            factor: Factor name (for logging)
        
        Returns:
            Z-scored values
        """
        valid_values = values.dropna()
        
        if len(valid_values) < 2:
            logger.warning(
                f"Sector '{sector}' has <2 valid values for {factor}, "
                f"assigning neutral (0.0)"
            )
            return pd.Series(0.0, index=values.index)
        
        if len(valid_values) < 5:
            logger.debug(
                f"Sector '{sector}' has only {len(valid_values)} stocks for {factor}, "
                f"z-score may be unstable"
            )
        
        # Compute z-score
        mean = valid_values.mean()
        std = valid_values.std()
        
        if std == 0 or pd.isna(std):
            logger.warning(
                f"Sector '{sector}' has zero std for {factor}, "
                f"assigning neutral (0.0)"
            )
            return pd.Series(0.0, index=values.index)
        
        z_scores = (values - mean) / std
        
        return z_scores
    
    def _handle_missing_values(
        self,
        normalized: pd.DataFrame,
        original: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Handle missing values according to policy.
        
        Args:
            normalized: Normalized scores (may have NaN)
            original: Original scores (for reference)
        
        Returns:
            Scores with missing values handled
        """
        if self.handle_missing == 'neutral':
            # Replace NaN with 0.0 (neutral z-score)
            normalized = normalized.fillna(0.0)
            
        elif self.handle_missing == 'drop':
            # Count missing factors per ticker
            missing_per_ticker = normalized.isna().sum(axis=1)
            drop_mask = missing_per_ticker > self.max_missing
            
            if drop_mask.any():
                drop_tickers = normalized.index[drop_mask].tolist()
                logger.warning(
                    f"Dropping {len(drop_tickers)} tickers with >{self.max_missing} "
                    f"missing factors: {drop_tickers[:5]}"
                )
                normalized = normalized[~drop_mask]
        
        elif self.handle_missing == 'forward_fill':
            logger.warning(
                "forward_fill not applicable for cross-sectional data, "
                "using neutral instead"
            )
            normalized = normalized.fillna(0.0)
        
        return normalized
    
    def _log_normalization_stats(
        self,
        original: pd.DataFrame,
        normalized: pd.DataFrame,
        sectors: pd.Series,
    ) -> None:
        """Log normalization statistics for verification."""
        logger.info("Normalization statistics:")
        
        for factor in normalized.columns:
            orig_mean = original[factor].mean()
            orig_std = original[factor].std()
            norm_mean = normalized[factor].mean()
            norm_std = normalized[factor].std()
            
            logger.info(
                f"  {factor}: "
                f"Original(μ={orig_mean:.3f}, σ={orig_std:.3f}) → "
                f"Normalized(μ={norm_mean:.3f}, σ={norm_std:.3f})"
            )
        
        # Check sector balance
        if self.method == 'sector_relative':
            for sector in sectors.unique():
                sector_mask = sectors == sector
                n_stocks = sector_mask.sum()
                sector_means = normalized[sector_mask].mean()
                
                logger.debug(
                    f"  Sector '{sector}' ({n_stocks} stocks): "
                    f"mean z-scores = {sector_means.to_dict()}"
                )