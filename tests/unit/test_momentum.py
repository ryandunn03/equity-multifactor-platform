"""
Comprehensive unit tests for MomentumFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.factors.momentum import MomentumFactor, compute_momentum
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """
    Create 300 days of synthetic price data for testing.
    
    Returns:
        DataFrame with 3 tickers showing different trends
    """
    dates = pd.bdate_range('2023-01-01', periods=300)
    
    return pd.DataFrame({
        'AAPL': np.linspace(100, 150, 300),   # Strong uptrend (+50%)
        'MSFT': np.linspace(100, 80, 300),    # Downtrend (-20%)
        'GOOGL': np.full(300, 100.0),         # Flat (0%)
    }, index=dates)


@pytest.fixture
def momentum_config():
    """Standard momentum configuration."""
    return {
        'lookback_months': 12,
        'skip_months': 1,
        'enabled': True,
        'weight': 1.0
    }


@pytest.fixture
def insufficient_prices():
    """Only 100 days of data (insufficient for 12-1 momentum)."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    return pd.DataFrame({
        'AAPL': np.linspace(100, 110, 100),
    }, index=dates)


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_momentum_basic(sample_prices, momentum_config):
    """Test basic momentum calculation and ranking."""
    factor = MomentumFactor(momentum_config)
    
    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, {}, as_of_date)
    
    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'AAPL', 'MSFT', 'GOOGL'}
    
    # Check ranking (uptrend > flat > downtrend)
    assert scores['AAPL'] > scores['GOOGL']
    assert scores['GOOGL'] > scores['MSFT']
    
    # Check sign
    assert scores['AAPL'] > 0  # Uptrend should be positive
    assert scores['MSFT'] < 0  # Downtrend should be negative
    assert abs(scores['GOOGL']) < 0.05  # Flat should be near zero


def test_momentum_values_reasonable(sample_prices, momentum_config):
    """Scores should be in reasonable range (not extreme)."""
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(sample_prices, {}, sample_prices.index[-1])
    
    # Momentum should typically be in [-1, 5] range
    # (100% loss to 500% gain over 12 months)
    assert scores.min() > -1.0
    assert scores.max() < 5.0
    
    # For our synthetic data, should be relatively modest
    assert abs(scores['AAPL']) < 1.0  # ~50% over 12 months
    assert abs(scores['MSFT']) < 1.0  # ~20% down


def test_momentum_from_config(momentum_config):
    """Test initialization from config dict."""
    factor = MomentumFactor(momentum_config)
    
    assert factor.name == "momentum"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert factor.lookback_months == 12
    assert factor.skip_months == 1
    assert factor.lookback_days == 252
    assert factor.skip_days == 21


def test_momentum_custom_lookback(sample_prices):
    """Test with 6-1 momentum variant."""
    config = {
        'lookback_months': 6,
        'skip_months': 1,
        'enabled': True,
        'weight': 1.0
    }
    
    factor = MomentumFactor(config)
    
    assert factor.lookback_months == 6
    assert factor.lookback_days == 126
    
    scores = factor.compute(sample_prices, {}, sample_prices.index[-1])
    
    # Should still rank correctly
    assert scores['AAPL'] > scores['GOOGL'] > scores['MSFT']


def test_momentum_required_data():
    """Test required_data() method."""
    factor = MomentumFactor({'enabled': True})
    required = factor.required_data()
    
    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert len(required['fundamentals']) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_momentum_insufficient_history(insufficient_prices, momentum_config):
    """Should raise InsufficientDataError with <180 days."""
    factor = MomentumFactor(momentum_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(insufficient_prices, {}, insufficient_prices.index[-1])

    assert "Requires" in str(exc_info.value) and "trading days" in str(exc_info.value)


def test_momentum_division_by_zero(momentum_config):
    """Handle zero prices gracefully."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 300),
        'ZERO': np.zeros(300),  # All zeros
    }, index=dates)
    
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(prices, {}, dates[-1])
    
    # AAPL should have valid score
    assert not pd.isna(scores['AAPL'])
    
    # ZERO should be NaN (can't compute momentum)
    assert pd.isna(scores['ZERO'])


def test_momentum_missing_prices(momentum_config):
    """Handle NaN in price series."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 300),
        'MSFT': np.linspace(100, 120, 300),
    }, index=dates)
    
    # Introduce NaN in middle of series
    prices.loc[dates[150:155], 'MSFT'] = np.nan
    
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(prices, {}, dates[-1])
    
    # Should still compute (forward-fill handles gaps)
    assert not pd.isna(scores['AAPL'])
    # MSFT may or may not be NaN depending on where gap is


def test_momentum_single_ticker(momentum_config):
    """Works with single stock."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 300),
    }, index=dates)
    
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(prices, {}, dates[-1])
    
    assert len(scores) == 1
    assert 'AAPL' in scores.index
    assert scores['AAPL'] > 0


def test_momentum_extreme_values(momentum_config):
    """Cap extreme momentum values (likely data errors)."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'NORMAL': np.linspace(100, 150, 300),
        'EXTREME': np.linspace(1, 1000, 300),  # 100,000% return
    }, index=dates)
    
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(prices, {}, dates[-1])
    
    # Extreme values should be capped
    assert scores['EXTREME'] <= 10.0  # Capped at 1000%


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_momentum_invalid_index_type(momentum_config):
    """Reject non-DatetimeIndex."""
    prices = pd.DataFrame({
        'AAPL': [100, 110, 120],
        'MSFT': [50, 55, 60],
    }, index=[0, 1, 2])  # Integer index, not DatetimeIndex
    
    factor = MomentumFactor(momentum_config)
    
    with pytest.raises(DataError) as exc_info:
        factor.compute(prices, {}, datetime.now())
    
    assert "DatetimeIndex" in str(exc_info.value)


def test_momentum_unsorted_index(momentum_config):
    """Handle unsorted dates (should auto-sort with warning)."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    shuffled_dates = dates.to_list()
    np.random.shuffle(shuffled_dates)
    
    prices = pd.DataFrame({
        'AAPL': np.random.randn(300) + 100,
    }, index=shuffled_dates)
    
    factor = MomentumFactor(momentum_config)
    
    # Should work (auto-sorts)
    # Note: This might log a warning but shouldn't raise
    scores = factor.compute(prices, {}, max(shuffled_dates))
    assert not scores.empty


def test_momentum_empty_dataframe(momentum_config):
    """Handle empty DataFrame."""
    prices = pd.DataFrame()

    factor = MomentumFactor(momentum_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(prices, {}, datetime.now())

    # Should raise error about index type or no valid columns
    error_msg = str(exc_info.value).lower()
    assert "datetimeindex" in error_msg or "no valid columns" in error_msg


def test_momentum_all_nan_column(momentum_config):
    """Handle ticker with all NaN prices."""
    dates = pd.bdate_range('2023-01-01', periods=300)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 300),
        'NAN_STOCK': np.full(300, np.nan),
    }, index=dates)
    
    factor = MomentumFactor(momentum_config)
    scores = factor.compute(prices, {}, dates[-1])
    
    # AAPL should work
    assert not pd.isna(scores['AAPL'])
    
    # NAN_STOCK should be excluded or NaN
    # (Implementation may drop it with warning)


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_momentum_standalone(sample_prices):
    """Test standalone compute_momentum function."""
    scores = compute_momentum(
        sample_prices,
        sample_prices.index[-1],
        lookback_days=252,
        skip_days=21
    )
    
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores['AAPL'] > scores['MSFT']


# ============================================================================
# Configuration Tests
# ============================================================================

def test_momentum_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'lookback_months': 12,
        'skip_months': 1,
        'enabled': False,  # Disabled
        'weight': 1.0
    }
    
    factor = MomentumFactor(config)
    assert factor.enabled == False


def test_momentum_default_config():
    """Test with minimal config (use defaults)."""
    config = {}  # Empty config, should use all defaults
    
    factor = MomentumFactor(config)
    
    assert factor.lookback_months == 12
    assert factor.skip_months == 1
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_momentum_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 5 years of daily data
    dates = pd.bdate_range('2019-01-01', '2024-01-01')
    n_days = len(dates)
    
    # Simulate realistic returns
    np.random.seed(42)
    aapl_returns = np.random.normal(0.0005, 0.015, n_days)  # 0.05% daily, 1.5% vol
    msft_returns = np.random.normal(0.0004, 0.012, n_days)
    
    aapl_prices = 100 * np.cumprod(1 + aapl_returns)
    msft_prices = 100 * np.cumprod(1 + msft_returns)
    
    prices = pd.DataFrame({
        'AAPL': aapl_prices,
        'MSFT': msft_prices,
    }, index=dates)
    
    config = {'lookback_months': 12, 'skip_months': 1, 'enabled': True}
    factor = MomentumFactor(config)
    
    # Compute at multiple dates
    test_dates = [dates[-1], dates[-60], dates[-120]]
    
    for test_date in test_dates:
        scores = factor.compute(prices, {}, test_date)
        
        assert len(scores) == 2
        assert not scores.isna().all()
        
        # Scores should be reasonable
        assert -1 < scores['AAPL'] < 5
        assert -1 < scores['MSFT'] < 5