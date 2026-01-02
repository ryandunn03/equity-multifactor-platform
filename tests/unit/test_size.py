"""
Comprehensive unit tests for SizeFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.factors.size import SizeFactor, compute_size
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    return pd.DataFrame({
        'SMALL_CAP': np.full(100, 50.0),
        'MID_CAP': np.full(100, 100.0),
        'LARGE_CAP': np.full(100, 200.0),
    }, index=dates)


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data for testing."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    return {
        'market_cap': pd.DataFrame({
            'SMALL_CAP': np.full(100, 1e9),      # $1B market cap
            'MID_CAP': np.full(100, 50e9),       # $50B market cap
            'LARGE_CAP': np.full(100, 2000e9),   # $2T market cap
        }, index=dates),
    }


@pytest.fixture
def size_config():
    """Standard size configuration."""
    return {
        'use_log': True,
        'enabled': True,
        'weight': 1.0
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_size_basic(sample_prices, sample_fundamentals, size_config):
    """Test basic size calculation and ranking."""
    factor = SizeFactor(size_config)

    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, sample_fundamentals, as_of_date)

    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'SMALL_CAP', 'MID_CAP', 'LARGE_CAP'}

    # All scores should be valid (not NaN)
    assert scores.notna().all()

    # Check ranking (SMALL_CAP should have highest score since inverted)
    assert scores['SMALL_CAP'] > scores['MID_CAP']
    assert scores['MID_CAP'] > scores['LARGE_CAP']

    # All scores should be negative (inverted log)
    assert (scores < 0).all()


def test_size_log_transformation(sample_prices, sample_fundamentals, size_config):
    """Test that log transformation works correctly."""
    factor = SizeFactor(size_config)

    as_of_date = sample_prices.index[-1]

    # Compute expected values manually
    market_caps = sample_fundamentals['market_cap'].iloc[-1]
    expected_scores = -np.log(market_caps)

    # Get actual scores
    actual_scores = factor.compute(sample_prices, sample_fundamentals, as_of_date)

    # Should match
    pd.testing.assert_series_equal(actual_scores, expected_scores, check_names=False)


def test_size_no_log_transformation(sample_prices, sample_fundamentals):
    """Test without log transformation."""
    config = {
        'use_log': False,
        'enabled': True,
        'weight': 1.0
    }

    factor = SizeFactor(config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still rank correctly
    assert scores['SMALL_CAP'] > scores['MID_CAP']
    assert scores['MID_CAP'] > scores['LARGE_CAP']

    # All scores should be negative (inverted)
    assert (scores < 0).all()


def test_size_from_config(size_config):
    """Test initialization from config dict."""
    factor = SizeFactor(size_config)

    assert factor.name == "size"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert factor.use_log == True


def test_size_required_data():
    """Test required_data() method."""
    factor = SizeFactor({'enabled': True})
    required = factor.required_data()

    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert 'market_cap' in required['fundamentals']


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_size_zero_market_cap(sample_prices, sample_fundamentals, size_config):
    """Handle zero market cap gracefully."""
    # Make MID_CAP have zero market cap
    sample_fundamentals['market_cap']['MID_CAP'] = 0.0

    factor = SizeFactor(size_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # SMALL_CAP and LARGE_CAP should have valid scores
    assert not pd.isna(scores['SMALL_CAP'])
    assert not pd.isna(scores['LARGE_CAP'])

    # MID_CAP should be NaN (can't compute log of zero)
    assert pd.isna(scores['MID_CAP'])


def test_size_negative_market_cap(sample_prices, sample_fundamentals, size_config):
    """Handle negative market cap gracefully."""
    # Make LARGE_CAP have negative market cap (invalid data)
    sample_fundamentals['market_cap']['LARGE_CAP'] = -1e12

    factor = SizeFactor(size_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # SMALL_CAP and MID_CAP should have valid scores
    assert not pd.isna(scores['SMALL_CAP'])
    assert not pd.isna(scores['MID_CAP'])

    # LARGE_CAP should be NaN (invalid market cap)
    assert pd.isna(scores['LARGE_CAP'])


def test_size_extreme_market_caps(sample_prices, sample_fundamentals, size_config):
    """Handle extreme market cap values."""
    # Very small and very large market caps
    sample_fundamentals['market_cap']['SMALL_CAP'] = 1e6  # $1M (very small)
    sample_fundamentals['market_cap']['LARGE_CAP'] = 10e12  # $10T (very large)

    factor = SizeFactor(size_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should not have infinite scores
    assert np.isfinite(scores.dropna()).all()

    # Ranking should still work
    assert scores['SMALL_CAP'] > scores['MID_CAP']
    assert scores['MID_CAP'] > scores['LARGE_CAP']


def test_size_missing_ticker_in_fundamentals(sample_prices, sample_fundamentals, size_config):
    """Handle ticker present in prices but missing in fundamentals."""
    # Add a new ticker to prices
    sample_prices['MEGA_CAP'] = 300.0

    factor = SizeFactor(size_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Original tickers should work
    assert not pd.isna(scores['SMALL_CAP'])
    assert not pd.isna(scores['MID_CAP'])
    assert not pd.isna(scores['LARGE_CAP'])

    # MEGA_CAP should be NaN (no fundamental data)
    assert pd.isna(scores['MEGA_CAP'])


def test_size_single_ticker(sample_prices, sample_fundamentals, size_config):
    """Works with single stock."""
    # Keep only SMALL_CAP
    sample_prices = sample_prices[['SMALL_CAP']]

    factor = SizeFactor(size_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert len(scores) == 1
    assert 'SMALL_CAP' in scores.index
    assert scores['SMALL_CAP'] < 0  # Negative (inverted)


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_size_missing_fundamental_field(sample_prices, size_config):
    """Reject fundamentals missing required fields."""
    # Missing 'market_cap'
    bad_fundamentals = {
        'revenue': pd.DataFrame({'SMALL_CAP': [1e9]}),
    }

    factor = SizeFactor(size_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "market_cap" in str(exc_info.value).lower()


def test_size_fundamentals_not_dict(sample_prices, size_config):
    """Reject non-dict fundamentals."""
    bad_fundamentals = "not a dict"

    factor = SizeFactor(size_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be dict" in str(exc_info.value).lower()


def test_size_fundamental_not_dataframe(sample_prices, size_config):
    """Reject fundamental fields that aren't DataFrames."""
    bad_fundamentals = {
        'market_cap': "not a dataframe",
    }

    factor = SizeFactor(size_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be DataFrame" in str(exc_info.value)


def test_size_empty_fundamental_dataframe(sample_prices, size_config):
    """Reject empty fundamental DataFrames."""
    bad_fundamentals = {
        'market_cap': pd.DataFrame(),
    }

    factor = SizeFactor(size_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "empty" in str(exc_info.value).lower()


def test_size_fundamental_no_datetime_index(sample_prices, sample_fundamentals, size_config):
    """Reject fundamentals without DatetimeIndex."""
    # Change index to integers
    sample_fundamentals['market_cap'].index = range(len(sample_fundamentals['market_cap']))

    factor = SizeFactor(size_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert "DatetimeIndex" in str(exc_info.value)


def test_size_no_fundamental_data_before_date(sample_prices, size_config):
    """Handle case where no fundamental data before as_of_date."""
    # Create fundamentals with future dates only
    future_dates = pd.bdate_range('2025-01-01', periods=100)
    future_fundamentals = {
        'market_cap': pd.DataFrame({'SMALL_CAP': [1e9]}, index=future_dates[:1]),
    }

    factor = SizeFactor(size_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(sample_prices, future_fundamentals, sample_prices.index[-1])

    assert "No fundamental data available" in str(exc_info.value)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_size_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'use_log': True,
        'enabled': False,
        'weight': 1.0
    }

    factor = SizeFactor(config)
    assert factor.enabled == False


def test_size_default_config():
    """Test with minimal config (use defaults)."""
    config = {}

    factor = SizeFactor(config)

    assert factor.use_log == True
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_size_standalone(sample_prices, sample_fundamentals):
    """Test standalone compute_size function."""
    scores = compute_size(
        sample_prices,
        sample_fundamentals,
        sample_prices.index[-1],
        use_log=True
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores['SMALL_CAP'] > scores['LARGE_CAP']


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_size_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 2 years of quarterly data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    prices = pd.DataFrame({
        'SMALL': np.full(n_quarters, 10.0),
        'MID': np.full(n_quarters, 50.0),
        'LARGE': np.full(n_quarters, 100.0),
        'MEGA': np.full(n_quarters, 200.0),
    }, index=dates)

    # Realistic market cap progression
    fundamentals = {
        'market_cap': pd.DataFrame({
            'SMALL': np.full(n_quarters, 500e6),    # $500M
            'MID': np.full(n_quarters, 10e9),       # $10B
            'LARGE': np.full(n_quarters, 500e9),    # $500B
            'MEGA': np.full(n_quarters, 3000e9),    # $3T
        }, index=dates),
    }

    config = {'use_log': True, 'enabled': True}
    factor = SizeFactor(config)

    scores = factor.compute(prices, fundamentals, dates[-1])

    # Smaller companies should score higher
    assert scores['SMALL'] > scores['MID']
    assert scores['MID'] > scores['LARGE']
    assert scores['LARGE'] > scores['MEGA']

    # All scores should be finite and negative
    assert np.isfinite(scores).all()
    assert (scores < 0).all()


def test_size_log_vs_no_log():
    """Compare log transformation vs no log."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    prices = pd.DataFrame({
        'SMALL': np.full(100, 50.0),
        'LARGE': np.full(100, 200.0),
    }, index=dates)

    fundamentals = {
        'market_cap': pd.DataFrame({
            'SMALL': np.full(100, 1e9),
            'LARGE': np.full(100, 1000e9),  # 1000x larger
        }, index=dates),
    }

    # With log
    factor_log = SizeFactor({'use_log': True, 'enabled': True})
    scores_log = factor_log.compute(prices, fundamentals, dates[-1])

    # Without log
    factor_no_log = SizeFactor({'use_log': False, 'enabled': True})
    scores_no_log = factor_no_log.compute(prices, fundamentals, dates[-1])

    # Both should rank the same
    assert scores_log['SMALL'] > scores_log['LARGE']
    assert scores_no_log['SMALL'] > scores_no_log['LARGE']

    # But magnitudes should be different
    # Log reduces the difference between small and large
    log_diff = scores_log['SMALL'] - scores_log['LARGE']
    no_log_diff = scores_no_log['SMALL'] - scores_no_log['LARGE']

    assert abs(log_diff) < abs(no_log_diff)


def test_size_time_series():
    """Test that size factor uses point-in-time data correctly."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    prices = pd.DataFrame({
        'GROWING': np.linspace(50, 150, 100),
    }, index=dates)

    # Market cap grows over time
    market_caps = np.linspace(1e9, 100e9, 100)
    fundamentals = {
        'market_cap': pd.DataFrame({
            'GROWING': market_caps,
        }, index=dates),
    }

    factor = SizeFactor({'use_log': True, 'enabled': True})

    # Compute at different dates
    score_early = factor.compute(prices, fundamentals, dates[30])
    score_late = factor.compute(prices, fundamentals, dates[-1])

    # Later score should be more negative (larger company)
    assert score_late['GROWING'] < score_early['GROWING']


def test_size_numerical_stability():
    """Test numerical stability with very small and large market caps."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    prices = pd.DataFrame({
        'TINY': np.full(100, 1.0),
        'HUGE': np.full(100, 1000.0),
    }, index=dates)

    fundamentals = {
        'market_cap': pd.DataFrame({
            'TINY': np.full(100, 1e6),      # $1M
            'HUGE': np.full(100, 10e12),    # $10T
        }, index=dates),
    }

    factor = SizeFactor({'use_log': True, 'enabled': True})
    scores = factor.compute(prices, fundamentals, dates[-1])

    # Should not overflow or underflow
    assert np.isfinite(scores).all()
    assert not np.isnan(scores).any()

    # Log transformation should keep difference reasonable
    diff = scores['TINY'] - scores['HUGE']
    assert 0 < diff < 50  # Log(10T / 1M) = log(1e16) ≈ 37


def test_size_point_in_time_correctness():
    """Verify that factor uses most recent available data before as_of_date."""
    dates = pd.bdate_range('2023-01-01', periods=10)

    prices = pd.DataFrame({
        'STOCK': np.full(10, 100.0),
    }, index=dates)

    # Market cap changes over time
    fundamentals = {
        'market_cap': pd.DataFrame({
            'STOCK': [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9],
        }, index=dates),
    }

    factor = SizeFactor({'use_log': True, 'enabled': True})

    # Compute at dates[5] - should use market cap at dates[5]
    score_mid = factor.compute(prices, fundamentals, dates[5])

    # Expected: -log(6e9)
    expected = -np.log(6e9)

    assert abs(score_mid['STOCK'] - expected) < 0.01
