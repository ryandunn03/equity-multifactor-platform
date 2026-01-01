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
from src.factors.normalizer import SectorNormalizer
from src.factors.compositor import FactorCompositor

__version__ = "1.0.0"

__all__ = [
    # Base class
    'BaseFactor',
    
    # Concrete factors
    'MomentumFactor',
    
    # Standalone functions
    'compute_momentum',
    
    # Pipeline components
    'SectorNormalizer',
    'FactorCompositor',
]


# Module-level documentation
__doc_sections__ = {
    'factors': [
        'MomentumFactor - 12-1 month momentum',
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
        ['MomentumFactor']
    """
    return ['MomentumFactor']


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
            'version': '1.0.0',
        },
    }
    
    return factor_info_map.get(factor_name, {})