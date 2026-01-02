"""
Factor library for multi-factor equity research platform.

This module provides a complete factor computation pipeline:
  1. BaseFactor - Abstract base class for all factors
  2. Concrete factors (Momentum, Value, Quality, etc.)
  3. SectorNormalizer - Sector-relative z-score normalization
  4. FactorCompositor - Factor combination (equal or IC-weighted)

Architecture:
  Raw Data → Factor Computation → Sector Normalization → Combination → Composite Score

Example Usage:
    >>> from src.factors import MomentumFactor, SectorNormalizer, FactorCompositor
    >>> import yaml
    >>> 
    >>> # Load configuration
    >>> with open('config/strategy_config.yaml') as f:
    ...     config = yaml.safe_load(f)
    >>> 
    >>> # Compute momentum scores
    >>> momentum = MomentumFactor(config['factors']['momentum'])
    >>> raw_scores = momentum.compute(prices, fundamentals, as_of_date)
    >>> 
    >>> # Normalize within sectors
    >>> normalizer = SectorNormalizer(config['normalization'])
    >>> normalized = normalizer.normalize(
    ...     pd.DataFrame({'momentum': raw_scores}),
    ...     sector_mapping
    ... )
    >>> 
    >>> # Combine factors
    >>> compositor = FactorCompositor(config['factor_combination'])
    >>> composite = compositor.combine(normalized)
    >>> 
    >>> # Use composite scores for portfolio construction
    >>> top_longs = composite.nlargest(100)
    >>> top_shorts = composite.nsmallest(100)

Version History:
    1.0.0 (2024-01-01): Initial release with momentum factor
"""

from src.factors.base import BaseFactor
from src.factors.momentum import MomentumFactor, compute_momentum
from src.factors.value import ValueFactor, compute_value
from src.factors.quality import QualityFactor, compute_quality
from src.factors.volatility import VolatilityFactor, compute_volatility
from src.factors.size import SizeFactor, compute_size
from src.factors.growth import GrowthFactor, compute_growth
from src.factors.normalizer import SectorNormalizer
from src.factors.compositor import FactorCompositor

__version__ = "2.0.0"

__all__ = [
    # Base class
    'BaseFactor',

    # Concrete factors
    'MomentumFactor',
    'ValueFactor',
    'QualityFactor',
    'VolatilityFactor',
    'SizeFactor',
    'GrowthFactor',

    # Standalone functions
    'compute_momentum',
    'compute_value',
    'compute_quality',
    'compute_volatility',
    'compute_size',
    'compute_growth',

    # Pipeline components
    'SectorNormalizer',
    'FactorCompositor',
]


# Module-level documentation
__doc_sections__ = {
    'factors': [
        'MomentumFactor - 12-1 month momentum',
        'ValueFactor - Book-to-Market + Earnings Yield',
        'QualityFactor - ROE + Profit Margin + Leverage',
        'VolatilityFactor - 60-day realized volatility (inverted)',
        'SizeFactor - Log market cap (inverted)',
        'GrowthFactor - Revenue + Earnings growth',
    ],
    'pipeline': [
        'SectorNormalizer - Sector-relative z-score normalization',
        'FactorCompositor - Equal-weighted or IC-weighted combination',
    ],
}


def get_available_factors():
    """
    Get list of available factor implementations.

    Returns:
        List of factor class names

    Example:
        >>> from src.factors import get_available_factors
        >>> factors = get_available_factors()
        >>> print(factors)
        ['MomentumFactor', 'ValueFactor', 'QualityFactor', 'VolatilityFactor', 'SizeFactor', 'GrowthFactor']
    """
    return ['MomentumFactor', 'ValueFactor', 'QualityFactor', 'VolatilityFactor', 'SizeFactor', 'GrowthFactor']


def get_factor_info(factor_name: str) -> dict:
    """
    Get information about a specific factor.

    Args:
        factor_name: Name of the factor class

    Returns:
        Dict with factor metadata

    Example:
        >>> from src.factors import get_factor_info
        >>> info = get_factor_info('MomentumFactor')
        >>> print(info['description'])
    """
    factor_info_map = {
        'MomentumFactor': {
            'description': '12-1 month momentum (trailing return excluding last month)',
            'data_required': ['prices'],
            'lookback_days': 252,
            'version': '2.0.0',
        },
        'ValueFactor': {
            'description': 'Value factor combining Book-to-Market and Earnings Yield',
            'data_required': ['prices', 'book_value', 'market_cap', 'earnings'],
            'lookback_days': 1,
            'version': '2.0.0',
        },
        'QualityFactor': {
            'description': 'Quality factor combining ROE, Profit Margin, and Leverage',
            'data_required': ['prices', 'net_income', 'equity', 'revenue', 'debt'],
            'lookback_days': 1,
            'version': '2.0.0',
        },
        'VolatilityFactor': {
            'description': 'Low volatility factor (60-day realized vol, inverted)',
            'data_required': ['prices'],
            'lookback_days': 60,
            'version': '2.0.0',
        },
        'SizeFactor': {
            'description': 'Size factor (log market cap, inverted for small cap premium)',
            'data_required': ['prices', 'market_cap'],
            'lookback_days': 1,
            'version': '2.0.0',
        },
        'GrowthFactor': {
            'description': 'Growth factor combining revenue and earnings growth',
            'data_required': ['prices', 'revenue', 'earnings'],
            'lookback_days': 365,
            'version': '2.0.0',
        },
    }

    return factor_info_map.get(factor_name, {})