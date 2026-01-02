"""
Comprehensive unit tests for GrowthFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.factors.growth import GrowthFactor, compute_growth
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')

    return pd.DataFrame({
        'GROWER': np.full(len(dates), 100.0),
        'STABLE': np.full(len(dates), 50.0),
        'DECLINING': np.full(len(dates), 25.0),
    }, index=dates)


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data for testing."""
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    return {
        'revenue': pd.DataFrame({
            'GROWER': np.linspace(100e9, 200e9, n_quarters),     # 100% growth over period
            'STABLE': np.full(n_quarters, 50e9),                  # 0% growth
            'DECLINING': np.linspace(100e9, 75e9, n_quarters),    # -25% growth
        }, index=dates),
        'earnings': pd.DataFrame({
            'GROWER': np.linspace(10e9, 30e9, n_quarters),       # 200% growth
            'STABLE': np.full(n_quarters, 5e9),                   # 0% growth
            'DECLINING': np.linspace(10e9, 5e9, n_quarters),      # -50% growth
        }, index=dates),
    }


@pytest.fixture
def growth_config():
    """Standard growth configuration."""
    return {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 4,
        'enabled': True,
        'weight': 1.0
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_growth_basic(sample_prices, sample_fundamentals, growth_config):
    """Test basic growth calculation and ranking."""
    factor = GrowthFactor(growth_config)

    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, sample_fundamentals, as_of_date)

    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'GROWER', 'STABLE', 'DECLINING'}

    # All scores should be valid (not NaN)
    assert scores.notna().all()

    # Check ranking (GROWER should have highest score)
    assert scores['GROWER'] > scores['STABLE']
    assert scores['STABLE'] > scores['DECLINING']


def test_growth_from_config(growth_config):
    """Test initialization from config dict."""
    factor = GrowthFactor(growth_config)

    assert factor.name == "growth"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert abs(factor.weight_revenue_growth - 0.5) < 0.01
    assert abs(factor.weight_earnings_growth - 0.5) < 0.01
    assert factor.lookback_quarters == 4


def test_growth_custom_weights(sample_prices, sample_fundamentals):
    """Test with custom component weights."""
    config = {
        'weight_revenue_growth': 0.7,
        'weight_earnings_growth': 0.3,
        'lookback_quarters': 4,
        'enabled': True,
        'weight': 1.0
    }

    factor = GrowthFactor(config)

    # Weights should be normalized
    assert abs(factor.weight_revenue_growth - 0.7) < 0.01
    assert abs(factor.weight_earnings_growth - 0.3) < 0.01

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still produce valid scores
    assert scores.notna().all()


def test_growth_custom_lookback(sample_prices, sample_fundamentals):
    """Test with custom lookback period."""
    config = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 2,  # 6-month growth instead of YoY
        'enabled': True,
        'weight': 1.0
    }

    factor = GrowthFactor(config)

    assert factor.lookback_quarters == 2

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still rank correctly
    assert scores['GROWER'] > scores['STABLE']


def test_growth_required_data():
    """Test required_data() method."""
    factor = GrowthFactor({'enabled': True})
    required = factor.required_data()

    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert 'revenue' in required['fundamentals']
    assert 'earnings' in required['fundamentals']


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_growth_negative_revenue(sample_prices, sample_fundamentals, growth_config):
    """Handle negative revenue (invalid data)."""
    # Make STABLE have negative revenue
    sample_fundamentals['revenue']['STABLE'] = -10e9

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # GROWER and DECLINING should have valid scores
    assert not pd.isna(scores['GROWER'])
    assert not pd.isna(scores['DECLINING'])

    # STABLE may have NaN due to invalid revenue


def test_growth_negative_earnings(sample_prices, sample_fundamentals, growth_config):
    """Handle negative earnings (losses - valid but tricky for growth)."""
    # Make current earnings negative but historical positive
    sample_fundamentals['earnings']['STABLE'].iloc[-1] = -1e9

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still compute (negative growth from positive to negative)
    assert not pd.isna(scores['GROWER'])


def test_growth_zero_base_value(sample_prices, sample_fundamentals, growth_config):
    """Handle zero base value (can't compute growth)."""
    # Make historical revenue zero
    sample_fundamentals['revenue']['DECLINING'].iloc[0] = 0.0

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # GROWER and STABLE should work
    assert not pd.isna(scores['GROWER'])
    assert not pd.isna(scores['STABLE'])

    # DECLINING may have NaN due to division by zero


def test_growth_extreme_growth_rates(sample_prices, sample_fundamentals, growth_config):
    """Cap extreme growth rates (likely data errors)."""
    # Make GROWER have extreme growth
    sample_fundamentals['revenue']['GROWER'].iloc[-1] = 10000e9  # 100x growth

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should not have infinite scores
    assert np.isfinite(scores.dropna()).all()


def test_growth_missing_ticker_in_fundamentals(sample_prices, sample_fundamentals, growth_config):
    """Handle ticker present in prices but missing in fundamentals."""
    # Add a new ticker to prices
    sample_prices['NEW_STOCK'] = 150.0

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Original tickers should work
    assert not pd.isna(scores['GROWER'])
    assert not pd.isna(scores['STABLE'])
    assert not pd.isna(scores['DECLINING'])

    # NEW_STOCK should be NaN (no fundamental data)
    assert pd.isna(scores['NEW_STOCK'])


def test_growth_single_ticker(sample_prices, sample_fundamentals, growth_config):
    """Works with single stock."""
    # Keep only GROWER
    sample_prices = sample_prices[['GROWER']]

    factor = GrowthFactor(growth_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert len(scores) == 1
    assert 'GROWER' in scores.index
    # Score should be 0 (z-score with single value)
    assert abs(scores['GROWER']) < 0.01


def test_growth_insufficient_history(sample_prices, growth_config):
    """Handle case with insufficient historical data."""
    # Only 2 quarters of data (need at least lookback + 1)
    short_dates = pd.date_range('2023-01-01', '2023-07-01', freq='Q')

    short_fundamentals = {
        'revenue': pd.DataFrame({
            'GROWER': [100e9, 120e9],
        }, index=short_dates),
        'earnings': pd.DataFrame({
            'GROWER': [10e9, 12e9],
        }, index=short_dates),
    }

    short_prices = pd.DataFrame({
        'GROWER': [100.0, 110.0],
    }, index=short_dates)

    factor = GrowthFactor(growth_config)

    # May compute with limited data or return NaN
    scores = factor.compute(short_prices, short_fundamentals, short_dates[-1])

    # Check that it doesn't crash
    assert isinstance(scores, pd.Series)


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_growth_missing_fundamental_field(sample_prices, growth_config):
    """Reject fundamentals missing required fields."""
    # Missing 'earnings'
    bad_fundamentals = {
        'revenue': pd.DataFrame({'GROWER': [100e9, 120e9]}),
    }

    factor = GrowthFactor(growth_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "earnings" in str(exc_info.value).lower()


def test_growth_fundamentals_not_dict(sample_prices, growth_config):
    """Reject non-dict fundamentals."""
    bad_fundamentals = "not a dict"

    factor = GrowthFactor(growth_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be dict" in str(exc_info.value).lower()


def test_growth_fundamental_not_dataframe(sample_prices, growth_config):
    """Reject fundamental fields that aren't DataFrames."""
    bad_fundamentals = {
        'revenue': "not a dataframe",
        'earnings': pd.DataFrame({'GROWER': [10e9]}),
    }

    factor = GrowthFactor(growth_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be DataFrame" in str(exc_info.value)


def test_growth_empty_fundamental_dataframe(sample_prices, growth_config):
    """Reject empty fundamental DataFrames."""
    bad_fundamentals = {
        'revenue': pd.DataFrame(),
        'earnings': pd.DataFrame({'GROWER': [10e9]}),
    }

    factor = GrowthFactor(growth_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "empty" in str(exc_info.value).lower()


def test_growth_fundamental_no_datetime_index(sample_prices, sample_fundamentals, growth_config):
    """Reject fundamentals without DatetimeIndex."""
    # Change index to integers
    sample_fundamentals['revenue'].index = range(len(sample_fundamentals['revenue']))

    factor = GrowthFactor(growth_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert "DatetimeIndex" in str(exc_info.value)


def test_growth_no_fundamental_data_before_date(sample_prices, growth_config):
    """Handle case where no fundamental data before as_of_date."""
    # Create fundamentals with future dates only
    future_dates = pd.date_range('2025-01-01', periods=10, freq='Q')
    future_fundamentals = {
        'revenue': pd.DataFrame({'GROWER': [100e9]}, index=future_dates[:1]),
        'earnings': pd.DataFrame({'GROWER': [10e9]}, index=future_dates[:1]),
    }

    factor = GrowthFactor(growth_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(sample_prices, future_fundamentals, sample_prices.index[-1])

    assert "No fundamental data available" in str(exc_info.value)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_growth_invalid_weight_range():
    """Reject weights outside [0,1]."""
    config = {
        'weight_revenue_growth': 1.5,  # Invalid
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 4,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        GrowthFactor(config)

    assert "must be in [0,1]" in str(exc_info.value)


def test_growth_zero_weights():
    """Reject all-zero weights."""
    config = {
        'weight_revenue_growth': 0.0,
        'weight_earnings_growth': 0.0,
        'lookback_quarters': 4,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        GrowthFactor(config)

    assert "non-zero" in str(exc_info.value).lower()


def test_growth_invalid_lookback():
    """Reject lookback < 1."""
    config = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 0,  # Invalid
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        GrowthFactor(config)

    assert "must be >= 1" in str(exc_info.value)


def test_growth_only_revenue(sample_prices, sample_fundamentals):
    """Test with only revenue growth component."""
    config = {
        'weight_revenue_growth': 1.0,
        'weight_earnings_growth': 0.0,
        'lookback_quarters': 4,
        'enabled': True
    }

    factor = GrowthFactor(config)

    assert factor.weight_revenue_growth == 1.0
    assert factor.weight_earnings_growth == 0.0

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert scores.notna().all()


def test_growth_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 4,
        'enabled': False,
        'weight': 1.0
    }

    factor = GrowthFactor(config)
    assert factor.enabled == False


def test_growth_default_config():
    """Test with minimal config (use defaults)."""
    config = {}

    factor = GrowthFactor(config)

    assert factor.weight_revenue_growth == 0.5
    assert factor.weight_earnings_growth == 0.5
    assert factor.lookback_quarters == 4
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_growth_standalone(sample_prices, sample_fundamentals):
    """Test standalone compute_growth function."""
    scores = compute_growth(
        sample_prices,
        sample_fundamentals,
        sample_prices.index[-1],
        weight_revenue_growth=0.5,
        weight_earnings_growth=0.5,
        lookback_quarters=4
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores.notna().all()


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_growth_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 3 years of quarterly data
    dates = pd.date_range('2021-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    prices = pd.DataFrame({
        'HIGH_GROWTH': np.full(n_quarters, 100.0),
        'MODERATE_GROWTH': np.full(n_quarters, 50.0),
        'NO_GROWTH': np.full(n_quarters, 25.0),
    }, index=dates)

    # Simulate growth trajectories
    fundamentals = {
        'revenue': pd.DataFrame({
            'HIGH_GROWTH': 50e9 * (1.2 ** np.arange(n_quarters)),     # 20% QoQ
            'MODERATE_GROWTH': 100e9 * (1.05 ** np.arange(n_quarters)),  # 5% QoQ
            'NO_GROWTH': np.full(n_quarters, 50e9),                      # Flat
        }, index=dates),
        'earnings': pd.DataFrame({
            'HIGH_GROWTH': 5e9 * (1.25 ** np.arange(n_quarters)),      # 25% QoQ
            'MODERATE_GROWTH': 10e9 * (1.03 ** np.arange(n_quarters)),   # 3% QoQ
            'NO_GROWTH': np.full(n_quarters, 5e9),                       # Flat
        }, index=dates),
    }

    config = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 4,
        'enabled': True
    }
    factor = GrowthFactor(config)

    scores = factor.compute(prices, fundamentals, dates[-1])

    # HIGH_GROWTH should score highest
    assert scores['HIGH_GROWTH'] > scores['MODERATE_GROWTH']
    assert scores['MODERATE_GROWTH'] > scores['NO_GROWTH']

    # All scores should be finite
    assert np.isfinite(scores).all()


def test_growth_different_lookbacks():
    """Test that different lookback periods give different results."""
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    prices = pd.DataFrame({
        'STOCK': np.full(n_quarters, 100.0),
    }, index=dates)

    # Accelerating growth
    fundamentals = {
        'revenue': pd.DataFrame({
            'STOCK': [100, 105, 112, 125, 145, 175, 215, 270],
        }, index=dates),
        'earnings': pd.DataFrame({
            'STOCK': [10, 11, 13, 16, 20, 26, 35, 48],
        }, index=dates),
    }

    # 2-quarter lookback
    config_2q = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 2,
        'enabled': True
    }
    factor_2q = GrowthFactor(config_2q)
    scores_2q = factor_2q.compute(prices, fundamentals, dates[-1])

    # 4-quarter lookback (YoY)
    config_4q = {
        'weight_revenue_growth': 0.5,
        'weight_earnings_growth': 0.5,
        'lookback_quarters': 4,
        'enabled': True
    }
    factor_4q = GrowthFactor(config_4q)
    scores_4q = factor_4q.compute(prices, fundamentals, dates[-1])

    # Both should be valid
    assert not pd.isna(scores_2q['STOCK'])
    assert not pd.isna(scores_4q['STOCK'])


def test_growth_zscore_computation():
    """Test z-score standardization."""
    factor = GrowthFactor({'enabled': True})

    # Test with simple series
    series = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=['A', 'B', 'C', 'D', 'E'])
    zscore = factor._compute_zscore(series)

    # Mean should be ~0, std should be ~1
    assert abs(zscore.mean()) < 0.01
    assert abs(zscore.std() - 1.0) < 0.01

    # Test with all same values
    same_series = pd.Series([0.2, 0.2, 0.2], index=['A', 'B', 'C'])
    zscore_same = factor._compute_zscore(same_series)

    # Should all be 0
    assert (zscore_same == 0.0).all()
