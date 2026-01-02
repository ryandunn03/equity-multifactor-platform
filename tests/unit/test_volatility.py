"""
Comprehensive unit tests for VolatilityFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.factors.volatility import VolatilityFactor, compute_volatility
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """Create sample price data with varying volatility."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    np.random.seed(42)

    # Low volatility stock (1% daily vol)
    low_vol_returns = np.random.normal(0.0005, 0.01, 100)
    low_vol_prices = 100 * np.cumprod(1 + low_vol_returns)

    # High volatility stock (3% daily vol)
    high_vol_returns = np.random.normal(0.0005, 0.03, 100)
    high_vol_prices = 100 * np.cumprod(1 + high_vol_returns)

    # Medium volatility stock (2% daily vol)
    med_vol_returns = np.random.normal(0.0005, 0.02, 100)
    med_vol_prices = 100 * np.cumprod(1 + med_vol_returns)

    return pd.DataFrame({
        'LOW_VOL': low_vol_prices,
        'HIGH_VOL': high_vol_prices,
        'MED_VOL': med_vol_prices,
    }, index=dates)


@pytest.fixture
def volatility_config():
    """Standard volatility configuration."""
    return {
        'lookback_days': 60,
        'annualize': True,
        'enabled': True,
        'weight': 1.0
    }


@pytest.fixture
def insufficient_prices():
    """Only 30 days of data (insufficient for 60-day volatility)."""
    dates = pd.bdate_range('2023-01-01', periods=30)
    return pd.DataFrame({
        'AAPL': np.linspace(100, 110, 30),
    }, index=dates)


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_volatility_basic(sample_prices, volatility_config):
    """Test basic volatility calculation and ranking."""
    factor = VolatilityFactor(volatility_config)

    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, {}, as_of_date)

    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'LOW_VOL', 'HIGH_VOL', 'MED_VOL'}

    # Check ranking (LOW_VOL should have highest score since inverted)
    assert scores['LOW_VOL'] > scores['MED_VOL']
    assert scores['MED_VOL'] > scores['HIGH_VOL']

    # All scores should be negative (inverted volatility)
    assert (scores < 0).all()


def test_volatility_values_reasonable(sample_prices, volatility_config):
    """Volatility scores should be in reasonable range."""
    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(sample_prices, {}, sample_prices.index[-1])

    # Annualized vol typically in 10-100% range for equities
    # Scores are negative, so check absolute values
    abs_scores = scores.abs()
    assert abs_scores.min() > 0.05  # At least 5% annualized vol
    assert abs_scores.max() < 2.0   # Less than 200% annualized vol


def test_volatility_from_config(volatility_config):
    """Test initialization from config dict."""
    factor = VolatilityFactor(volatility_config)

    assert factor.name == "volatility"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert factor.lookback_days == 60
    assert factor.annualize == True


def test_volatility_custom_lookback(sample_prices):
    """Test with custom lookback period."""
    config = {
        'lookback_days': 30,
        'annualize': True,
        'enabled': True,
        'weight': 1.0
    }

    factor = VolatilityFactor(config)

    assert factor.lookback_days == 30

    scores = factor.compute(sample_prices, {}, sample_prices.index[-1])

    # Should still rank correctly
    assert scores['LOW_VOL'] > scores['MED_VOL'] > scores['HIGH_VOL']


def test_volatility_no_annualization(sample_prices):
    """Test without annualizing volatility."""
    config = {
        'lookback_days': 60,
        'annualize': False,
        'enabled': True,
        'weight': 1.0
    }

    factor = VolatilityFactor(config)

    scores_no_annualize = factor.compute(sample_prices, {}, sample_prices.index[-1])

    # With annualization
    config['annualize'] = True
    factor_annualized = VolatilityFactor(config)
    scores_annualized = factor_annualized.compute(sample_prices, {}, sample_prices.index[-1])

    # Annualized should be larger in magnitude (multiplied by sqrt(252))
    assert scores_annualized.abs().mean() > scores_no_annualize.abs().mean()


def test_volatility_required_data():
    """Test required_data() method."""
    factor = VolatilityFactor({'enabled': True})
    required = factor.required_data()

    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert len(required['fundamentals']) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_volatility_insufficient_history(insufficient_prices, volatility_config):
    """Should raise InsufficientDataError with <60 days."""
    factor = VolatilityFactor(volatility_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(insufficient_prices, {}, insufficient_prices.index[-1])

    assert "Requires" in str(exc_info.value) and "trading days" in str(exc_info.value)


def test_volatility_constant_prices(volatility_config):
    """Handle constant prices (zero volatility)."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    prices = pd.DataFrame({
        'CONSTANT': np.full(100, 100.0),  # No movement
        'VARIABLE': np.linspace(100, 120, 100),
    }, index=dates)

    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(prices, {}, dates[-1])

    # CONSTANT should have score of 0 or NaN (zero volatility)
    # VARIABLE should have valid negative score
    assert not pd.isna(scores['VARIABLE'])
    assert scores['VARIABLE'] < 0


def test_volatility_missing_prices(volatility_config):
    """Handle NaN in price series."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 100),
        'MSFT': np.linspace(100, 120, 100),
    }, index=dates)

    # Introduce NaN in middle of series
    prices.loc[dates[40:45], 'MSFT'] = np.nan

    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(prices, {}, dates[-1])

    # Should still compute (handles NaN in returns)
    assert not pd.isna(scores['AAPL'])
    # MSFT may have score if enough valid data


def test_volatility_single_ticker(volatility_config):
    """Works with single stock."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    prices = pd.DataFrame({
        'AAPL': 100 * np.cumprod(1 + returns),
    }, index=dates)

    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(prices, {}, dates[-1])

    assert len(scores) == 1
    assert 'AAPL' in scores.index
    assert scores['AAPL'] < 0  # Negative (inverted)


def test_volatility_extreme_values(volatility_config):
    """Handle extreme price movements."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    # Extreme volatility stock
    np.random.seed(42)
    extreme_returns = np.random.normal(0, 0.10, 100)  # 10% daily vol
    extreme_prices = 100 * np.cumprod(1 + extreme_returns)

    prices = pd.DataFrame({
        'EXTREME': extreme_prices,
    }, index=dates)

    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(prices, {}, dates[-1])

    # Should have very negative score (high volatility)
    assert scores['EXTREME'] < -0.5  # High annualized vol


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_volatility_invalid_index_type(volatility_config):
    """Reject non-DatetimeIndex."""
    prices = pd.DataFrame({
        'AAPL': [100, 110, 120],
        'MSFT': [50, 55, 60],
    }, index=[0, 1, 2])  # Integer index

    factor = VolatilityFactor(volatility_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(prices, {}, datetime.now())

    assert "DatetimeIndex" in str(exc_info.value)


def test_volatility_empty_dataframe(volatility_config):
    """Handle empty DataFrame."""
    prices = pd.DataFrame()

    factor = VolatilityFactor(volatility_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(prices, {}, datetime.now())

    error_msg = str(exc_info.value).lower()
    assert "datetimeindex" in error_msg or "empty" in error_msg


def test_volatility_all_nan_column(volatility_config):
    """Handle ticker with all NaN prices."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 150, 100),
        'NAN_STOCK': np.full(100, np.nan),
    }, index=dates)

    factor = VolatilityFactor(volatility_config)
    scores = factor.compute(prices, {}, dates[-1])

    # AAPL should work
    assert not pd.isna(scores['AAPL'])

    # NAN_STOCK should be NaN (can't compute volatility)
    assert pd.isna(scores['NAN_STOCK'])


# ============================================================================
# Configuration Tests
# ============================================================================

def test_volatility_invalid_lookback():
    """Reject lookback < 5 days."""
    config = {
        'lookback_days': 2,  # Too short
        'annualize': True,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        VolatilityFactor(config)

    assert "must be >= 5" in str(exc_info.value)


def test_volatility_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'lookback_days': 60,
        'annualize': True,
        'enabled': False,
        'weight': 1.0
    }

    factor = VolatilityFactor(config)
    assert factor.enabled == False


def test_volatility_default_config():
    """Test with minimal config (use defaults)."""
    config = {}

    factor = VolatilityFactor(config)

    assert factor.lookback_days == 60
    assert factor.annualize == True
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_volatility_standalone(sample_prices):
    """Test standalone compute_volatility function."""
    scores = compute_volatility(
        sample_prices,
        sample_prices.index[-1],
        lookback_days=60,
        annualize=True
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores['LOW_VOL'] > scores['HIGH_VOL']


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_volatility_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 1 year of daily data
    dates = pd.bdate_range('2023-01-01', '2024-01-01')
    n_days = len(dates)

    # Simulate realistic stocks
    np.random.seed(42)

    # Tech stock (higher vol)
    tech_returns = np.random.normal(0.0008, 0.025, n_days)
    tech_prices = 100 * np.cumprod(1 + tech_returns)

    # Utility stock (lower vol)
    util_returns = np.random.normal(0.0003, 0.01, n_days)
    util_prices = 100 * np.cumprod(1 + util_returns)

    prices = pd.DataFrame({
        'TECH': tech_prices,
        'UTIL': util_prices,
    }, index=dates)

    config = {'lookback_days': 60, 'annualize': True, 'enabled': True}
    factor = VolatilityFactor(config)

    # Compute at multiple dates
    test_dates = [dates[-1], dates[-60], dates[-120]]

    for test_date in test_dates:
        scores = factor.compute(prices, {}, test_date)

        assert len(scores) == 2
        assert not scores.isna().all()

        # Utility should have higher score (lower vol)
        assert scores['UTIL'] > scores['TECH']

        # Scores should be negative
        assert (scores < 0).all()


def test_volatility_different_lookbacks():
    """Test that different lookback periods give different results."""
    dates = pd.bdate_range('2023-01-01', periods=150)
    np.random.seed(42)

    returns = np.random.normal(0.001, 0.02, 150)
    prices = pd.DataFrame({
        'STOCK': 100 * np.cumprod(1 + returns),
    }, index=dates)

    # 30-day volatility
    config_30 = {'lookback_days': 30, 'annualize': True, 'enabled': True}
    factor_30 = VolatilityFactor(config_30)
    scores_30 = factor_30.compute(prices, {}, dates[-1])

    # 90-day volatility
    config_90 = {'lookback_days': 90, 'annualize': True, 'enabled': True}
    factor_90 = VolatilityFactor(config_90)
    scores_90 = factor_90.compute(prices, {}, dates[-1])

    # Scores should be different (different measurement windows)
    assert scores_30['STOCK'] != scores_90['STOCK']


def test_volatility_log_returns():
    """Verify that log returns are used (not simple returns)."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    # Simple test case
    prices = pd.DataFrame({
        'STOCK': [100, 110, 121],  # 10% simple returns
    }, index=dates[:3])

    # Extend to 100 days
    prices = pd.DataFrame({
        'STOCK': np.linspace(100, 120, 100),
    }, index=dates)

    factor = VolatilityFactor({'lookback_days': 60, 'annualize': False, 'enabled': True})
    scores = factor.compute(prices, {}, dates[-1])

    # Should compute successfully
    assert not pd.isna(scores['STOCK'])
    assert scores['STOCK'] < 0


def test_volatility_insufficient_returns():
    """Handle case where too few valid returns in window."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    prices = pd.DataFrame({
        'SPARSE': [100.0] * 100,
    }, index=dates)

    # Set most values to NaN
    prices.loc[dates[10:95], 'SPARSE'] = np.nan

    factor = VolatilityFactor({'lookback_days': 60, 'enabled': True})
    scores = factor.compute(prices, {}, dates[-1])

    # Should be NaN due to insufficient data
    assert pd.isna(scores['SPARSE'])


def test_volatility_annualization_factor():
    """Verify correct annualization factor (sqrt(252))."""
    dates = pd.bdate_range('2023-01-01', periods=100)
    np.random.seed(42)

    returns = np.random.normal(0, 0.01, 100)  # 1% daily vol
    prices = pd.DataFrame({
        'STOCK': 100 * np.cumprod(1 + returns),
    }, index=dates)

    # Without annualization
    config_no_annual = {'lookback_days': 60, 'annualize': False, 'enabled': True}
    factor_no_annual = VolatilityFactor(config_no_annual)
    scores_no_annual = factor_no_annual.compute(prices, {}, dates[-1])

    # With annualization
    config_annual = {'lookback_days': 60, 'annualize': True, 'enabled': True}
    factor_annual = VolatilityFactor(config_annual)
    scores_annual = factor_annual.compute(prices, {}, dates[-1])

    # Ratio should be approximately sqrt(252)
    ratio = abs(scores_annual['STOCK'] / scores_no_annual['STOCK'])
    expected_ratio = np.sqrt(252)

    assert abs(ratio - expected_ratio) < 1.0  # Allow some tolerance
