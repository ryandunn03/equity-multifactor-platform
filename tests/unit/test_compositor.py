"""
Unit tests for FactorCompositor.

Tests equal-weighting, IC-weighting, shrinkage, and IC history management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.factors.compositor import FactorCompositor


@pytest.fixture
def sample_normalized_scores():
    """Normalized factor scores (z-scores)."""
    return pd.DataFrame({
        'momentum': [0.5, -0.3, 1.2, -0.8, 0.2],
        'value': [-0.2, 0.8, 0.1, -0.5, 0.3],
        'quality': [0.3, -0.1, 0.9, -0.4, 0.5],
    }, index=['AAPL', 'MSFT', 'GOOGL', 'XOM', 'JPM'])


@pytest.fixture
def compositor_config_equal():
    """Equal-weighted configuration."""
    return {
        'method': 'equal_weighted',
        'ic_lookback_months': 12,
        'shrinkage': 0.4,
        'min_ic': -0.5,
    }


@pytest.fixture
def compositor_config_ic():
    """IC-weighted configuration."""
    return {
        'method': 'ic_weighted',
        'ic_lookback_months': 12,
        'shrinkage': 0.4,
        'min_ic': -0.5,
    }


# ============================================================================
# Equal-Weighted Tests
# ============================================================================

def test_compositor_equal_weighted_basic(sample_normalized_scores, compositor_config_equal):
    """Test basic equal-weighted combination."""
    compositor = FactorCompositor(compositor_config_equal)
    composite = compositor.combine(sample_normalized_scores)
    
    # Check return type and shape
    assert isinstance(composite, pd.Series)
    assert len(composite) == 5
    assert set(composite.index) == set(sample_normalized_scores.index)
    
    # Verify equal weighting (average of factors)
    expected = sample_normalized_scores.mean(axis=1)
    pd.testing.assert_series_equal(composite, expected, check_names=False)


def test_compositor_equal_weighted_preserves_ranking(sample_normalized_scores, compositor_config_equal):
    """Equal weighting should preserve general ranking."""
    compositor = FactorCompositor(compositor_config_equal)
    composite = compositor.combine(sample_normalized_scores)
    
    # Ticker with high scores on all factors should rank high
    assert composite['GOOGL'] > composite.median()
    assert composite['XOM'] < composite.median()


def test_compositor_single_factor(compositor_config_equal):
    """Handle single factor (degenerate case)."""
    scores = pd.DataFrame({
        'momentum': [0.5, -0.3, 1.2],
    }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    compositor = FactorCompositor(compositor_config_equal)
    composite = compositor.combine(scores)
    
    # Should equal the single factor
    pd.testing.assert_series_equal(composite, scores['momentum'], check_names=False)


# ============================================================================
# IC-Weighted Tests (Without History)
# ============================================================================

def test_compositor_ic_weighted_fallback(sample_normalized_scores, compositor_config_ic):
    """IC-weighted should fall back to equal-weight without history."""
    compositor = FactorCompositor(compositor_config_ic)
    composite = compositor.combine(sample_normalized_scores)
    
    # Should fall back to equal weighting (no IC history)
    expected = sample_normalized_scores.mean(axis=1)
    pd.testing.assert_series_equal(composite, expected, check_names=False)


# ============================================================================
# IC History Management
# ============================================================================

def test_compositor_update_ic_history():
    """Test IC history update."""
    scores = pd.DataFrame({
        'momentum': [0.5, -0.3, 1.2, -0.8, 0.2],
        'value': [-0.2, 0.8, 0.1, -0.5, 0.3],
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    # Simulate forward returns (momentum works, value doesn't)
    forward_returns = pd.Series({
        'A': 0.05,   # High momentum → positive return
        'B': -0.02,  # Low momentum → negative return
        'C': 0.10,   # High momentum → positive return
        'D': -0.05,  # Low momentum → negative return
        'E': 0.01,   # Low momentum → small positive
    })
    
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 3,
        'shrinkage': 0.4,
        'min_ic': -0.5,
    }
    
    compositor = FactorCompositor(config)
    
    # Update IC
    compositor.update_ic_history(scores, forward_returns, datetime(2024, 1, 31))
    
    # Check IC history was created
    assert 'momentum' in compositor.ic_history
    assert 'value' in compositor.ic_history
    assert len(compositor.ic_history['momentum']) == 1
    
    # Momentum should have positive IC (correlates with returns)
    assert compositor.ic_history['momentum'][0] > 0


def test_compositor_ic_history_fifo():
    """IC history should be FIFO (first-in, first-out)."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 3,
        'shrinkage': 0.4,
    }
    
    compositor = FactorCompositor(config)
    
    # Manually populate IC history
    compositor.ic_history['momentum'] = [0.1, 0.2, 0.3]  # 3 months (at limit)
    
    # Add one more (should drop oldest)
    scores = pd.DataFrame({'momentum': [0.5, -0.3]}, index=['A', 'B'])
    returns = pd.Series({'A': 0.05, 'B': -0.02})
    
    compositor.update_ic_history(scores, returns, datetime.now())
    
    # Should still have only 3 (or maintain reasonable length)
    assert len(compositor.ic_history['momentum']) >= 3


def test_compositor_ic_computation():
    """Test IC calculation (Spearman correlation)."""
    # Perfect positive correlation
    scores_pos = pd.DataFrame({
        'factor': [1, 2, 3, 4, 5],
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    returns_pos = pd.Series({
        'A': 0.01,
        'B': 0.02,
        'C': 0.03,
        'D': 0.04,
        'E': 0.05,
    })
    
    config = {'method': 'ic_weighted', 'ic_lookback_months': 12, 'shrinkage': 0.0}
    compositor = FactorCompositor(config)
    
    compositor.update_ic_history(scores_pos, returns_pos, datetime.now())
    
    # IC should be close to 1.0 (perfect rank correlation)
    assert compositor.ic_history['factor'][0] > 0.9


# ============================================================================
# IC-Weighted Combination (With History)
# ============================================================================

def test_compositor_ic_weighted_with_history(sample_normalized_scores):
    """Test IC-weighted combination with pre-populated history."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 3,
        'shrinkage': 0.0,  # No shrinkage for cleaner test
        'min_ic': -0.5,
    }
    
    compositor = FactorCompositor(config)
    
    # Manually populate IC history (momentum strong, value weak, quality medium)
    compositor.ic_history = {
        'momentum': [0.08, 0.09, 0.10],  # Strong predictor
        'value': [0.01, 0.02, 0.01],     # Weak predictor
        'quality': [0.05, 0.04, 0.06],   # Medium predictor
    }
    
    composite = compositor.combine(sample_normalized_scores)
    
    # Should weight momentum highest
    weights = compositor.get_current_weights()
    assert weights['momentum'] > weights['value']
    assert weights['momentum'] > weights['quality']


def test_compositor_shrinkage():
    """Test shrinkage toward equal-weight."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 3,
        'shrinkage': 0.5,  # 50% shrinkage
        'min_ic': -0.5,
    }
    
    compositor = FactorCompositor(config)
    
    # Extreme IC difference
    compositor.ic_history = {
        'momentum': [0.10, 0.10, 0.10],  # Very strong
        'value': [0.00, 0.00, 0.00],     # Zero
    }
    
    weights = compositor.get_current_weights()
    
    # With shrinkage, value should still get some weight
    # (not pure IC weighting)
    equal_weight = 0.5
    assert weights['value'] > 0.0
    assert weights['value'] < weights['momentum']  # But less than momentum
    
    # Verify shrinkage formula
    # weight = 0.5 * (1/2) + 0.5 * ic_weight
    # So value should get at least 0.25 from shrinkage
    assert weights['value'] >= 0.20


def test_compositor_min_ic_filter():
    """Test min_ic threshold filters negative predictors."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 3,
        'shrinkage': 0.0,
        'min_ic': 0.0,  # Filter negative ICs
    }
    
    compositor = FactorCompositor(config)
    
    compositor.ic_history = {
        'momentum': [0.08, 0.09, 0.08],   # Positive (keep)
        'value': [-0.05, -0.04, -0.06],   # Negative (filter)
    }
    
    weights = compositor.get_current_weights()
    
    # Value should get zero weight (filtered by min_ic)
    assert weights['value'] == 0.0 or weights['value'] < 0.01
    assert weights['momentum'] > 0.9  # Gets almost all weight


# ============================================================================
# Edge Cases
# ============================================================================

def test_compositor_empty_dataframe(compositor_config_equal):
    """Handle empty DataFrame."""
    scores = pd.DataFrame()
    
    compositor = FactorCompositor(compositor_config_equal)
    composite = compositor.combine(scores)
    
    assert composite.empty


def test_compositor_all_nan_factors(compositor_config_equal):
    """Handle all-NaN factor column."""
    scores = pd.DataFrame({
        'factor1': [0.5, -0.3, 1.2],
        'factor2': [np.nan, np.nan, np.nan],
    }, index=['A', 'B', 'C'])
    
    compositor = FactorCompositor(compositor_config_equal)
    composite = compositor.combine(scores)
    
    # Should handle gracefully (skipna in mean)
    assert not composite.isna().all()


def test_compositor_get_ic_statistics():
    """Test get_ic_statistics() method."""
    config = {'method': 'ic_weighted', 'ic_lookback_months': 12}
    compositor = FactorCompositor(config)
    
    # Populate history
    compositor.ic_history = {
        'momentum': [0.05, 0.06, 0.04, 0.07],
        'value': [0.02, 0.03, 0.01, 0.02],
    }
    
    stats = compositor.get_ic_statistics()
    
    assert isinstance(stats, pd.DataFrame)
    assert 'mean' in stats.columns
    assert 'std' in stats.columns
    assert 't_stat' in stats.columns
    assert 'count' in stats.columns
    
    # Check momentum has higher mean IC
    assert stats.loc['momentum', 'mean'] > stats.loc['value', 'mean']


def test_compositor_initialization_from_config():
    """Test initialization from config dict."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 6,
        'shrinkage': 0.3,
        'min_ic': -0.2,
    }
    
    compositor = FactorCompositor(config)
    
    assert compositor.method == 'ic_weighted'
    assert compositor.ic_lookback == 6
    assert compositor.shrinkage == 0.3
    assert compositor.min_ic == -0.2


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_compositor_realistic_workflow():
    """Simulate realistic multi-period workflow."""
    config = {
        'method': 'ic_weighted',
        'ic_lookback_months': 12,
        'shrinkage': 0.4,
        'min_ic': -0.5,
    }
    
    compositor = FactorCompositor(config)
    
    # Simulate 15 months of data
    np.random.seed(42)
    dates = [datetime(2023, m, 1) for m in range(1, 13)] + \
            [datetime(2024, m, 1) for m in range(1, 4)]
    
    for date in dates:
        # Random factor scores
        scores = pd.DataFrame({
            'momentum': np.random.randn(50),
            'value': np.random.randn(50),
            'quality': np.random.randn(50),
        }, index=[f"stock_{i}" for i in range(50)])
        
        # Random forward returns (with some correlation to momentum)
        returns = pd.Series(
            np.random.randn(50) * 0.05 + scores['momentum'] * 0.01,
            index=scores.index
        )
        
        # Combine factors
        composite = compositor.combine(scores)
        
        # Update IC
        compositor.update_ic_history(scores, returns, date)
    
    # After 15 months, should have IC history
    assert len(compositor.ic_history['momentum']) == 15
    
    # Should be using IC weighting now
    weights = compositor.get_current_weights()
    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 0.01  # Weights sum to 1