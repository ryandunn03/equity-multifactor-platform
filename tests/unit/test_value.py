"""
Comprehensive unit tests for ValueFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.factors.value import ValueFactor, compute_value
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """
    Create sample price data for testing.

    Returns:
        DataFrame with 3 tickers
    """
    dates = pd.bdate_range('2023-01-01', periods=100)

    return pd.DataFrame({
        'AAPL': np.full(100, 150.0),
        'MSFT': np.full(100, 300.0),
        'GOOGL': np.full(100, 100.0),
    }, index=dates)


@pytest.fixture
def sample_fundamentals():
    """
    Create sample fundamental data for testing.

    Returns:
        Dict with book_value, market_cap, earnings DataFrames
    """
    dates = pd.bdate_range('2023-01-01', periods=100)

    return {
        'book_value': pd.DataFrame({
            'AAPL': np.full(100, 50_000_000_000.0),   # $50B book value
            'MSFT': np.full(100, 100_000_000_000.0),  # $100B book value
            'GOOGL': np.full(100, 200_000_000_000.0), # $200B book value
        }, index=dates),
        'market_cap': pd.DataFrame({
            'AAPL': np.full(100, 2_000_000_000_000.0),   # $2T market cap
            'MSFT': np.full(100, 2_500_000_000_000.0),   # $2.5T market cap
            'GOOGL': np.full(100, 1_500_000_000_000.0),  # $1.5T market cap
        }, index=dates),
        'earnings': pd.DataFrame({
            'AAPL': np.full(100, 100_000_000_000.0),  # $100B earnings
            'MSFT': np.full(100, 75_000_000_000.0),   # $75B earnings
            'GOOGL': np.full(100, 90_000_000_000.0),  # $90B earnings
        }, index=dates),
    }


@pytest.fixture
def value_config():
    """Standard value configuration."""
    return {
        'weight_book_to_market': 0.5,
        'weight_earnings_yield': 0.5,
        'enabled': True,
        'weight': 1.0
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_value_basic(sample_prices, sample_fundamentals, value_config):
    """Test basic value calculation and ranking."""
    factor = ValueFactor(value_config)

    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, sample_fundamentals, as_of_date)

    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'AAPL', 'MSFT', 'GOOGL'}

    # All scores should be valid (not NaN)
    assert scores.notna().all()

    # GOOGL should have highest value score (highest B/M and E/Y)
    # B/M: GOOGL=0.133, MSFT=0.04, AAPL=0.025
    # E/Y: GOOGL=0.06, AAPL=0.05, MSFT=0.03
    assert scores['GOOGL'] > scores['AAPL']
    assert scores['GOOGL'] > scores['MSFT']


def test_value_book_to_market_calculation(sample_prices, sample_fundamentals, value_config):
    """Test that book-to-market is calculated correctly."""
    factor = ValueFactor(value_config)

    as_of_date = sample_prices.index[-1]

    # Compute B/M manually
    book_values = sample_fundamentals['book_value'].iloc[-1]
    market_caps = sample_fundamentals['market_cap'].iloc[-1]
    expected_bm = book_values / market_caps

    # Get B/M from factor
    actual_bm = factor._compute_book_to_market(
        sample_fundamentals,
        ['AAPL', 'MSFT', 'GOOGL'],
        as_of_date
    )

    # Should match
    pd.testing.assert_series_equal(actual_bm, expected_bm, check_names=False)


def test_value_earnings_yield_calculation(sample_prices, sample_fundamentals, value_config):
    """Test that earnings yield is calculated correctly."""
    factor = ValueFactor(value_config)

    as_of_date = sample_prices.index[-1]

    # Compute E/Y manually
    earnings = sample_fundamentals['earnings'].iloc[-1]
    market_caps = sample_fundamentals['market_cap'].iloc[-1]
    expected_ey = earnings / market_caps

    # Get E/Y from factor
    actual_ey = factor._compute_earnings_yield(
        sample_fundamentals,
        ['AAPL', 'MSFT', 'GOOGL'],
        as_of_date
    )

    # Should match
    pd.testing.assert_series_equal(actual_ey, expected_ey, check_names=False)


def test_value_from_config(value_config):
    """Test initialization from config dict."""
    factor = ValueFactor(value_config)

    assert factor.name == "value"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert abs(factor.weight_book_to_market - 0.5) < 0.01
    assert abs(factor.weight_earnings_yield - 0.5) < 0.01


def test_value_custom_weights(sample_prices, sample_fundamentals):
    """Test with custom component weights."""
    config = {
        'weight_book_to_market': 0.7,
        'weight_earnings_yield': 0.3,
        'enabled': True,
        'weight': 1.0
    }

    factor = ValueFactor(config)

    # Weights should be normalized
    assert abs(factor.weight_book_to_market - 0.7) < 0.01
    assert abs(factor.weight_earnings_yield - 0.3) < 0.01

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still produce valid scores
    assert scores.notna().all()


def test_value_required_data():
    """Test required_data() method."""
    factor = ValueFactor({'enabled': True})
    required = factor.required_data()

    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert 'book_value' in required['fundamentals']
    assert 'market_cap' in required['fundamentals']
    assert 'earnings' in required['fundamentals']


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_value_negative_book_value(sample_prices, sample_fundamentals, value_config):
    """Handle negative book value (distressed companies)."""
    # Make MSFT have negative book value
    sample_fundamentals['book_value']['MSFT'] = -10_000_000_000.0

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # AAPL and GOOGL should have valid scores
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['GOOGL'])

    # MSFT may have NaN or reduced score due to negative book value
    # (implementation filters out negative book values)


def test_value_negative_earnings(sample_prices, sample_fundamentals, value_config):
    """Handle negative earnings (losses)."""
    # Make AAPL have negative earnings
    sample_fundamentals['earnings']['AAPL'] = -5_000_000_000.0

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # MSFT and GOOGL should have valid scores
    assert not pd.isna(scores['MSFT'])
    assert not pd.isna(scores['GOOGL'])

    # AAPL may have reduced score due to negative earnings
    # (implementation filters out negative earnings)


def test_value_zero_market_cap(sample_prices, sample_fundamentals, value_config):
    """Handle zero market cap gracefully."""
    # Make GOOGL have zero market cap
    sample_fundamentals['market_cap']['GOOGL'] = 0.0

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # AAPL and MSFT should have valid scores
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['MSFT'])

    # GOOGL should be NaN (can't compute ratios with zero market cap)
    assert pd.isna(scores['GOOGL'])


def test_value_extreme_ratios(sample_prices, sample_fundamentals, value_config):
    """Cap extreme value ratios (likely data errors)."""
    # Make AAPL have extreme B/M ratio
    sample_fundamentals['book_value']['AAPL'] = 50_000_000_000_000.0  # $50T

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should not have infinite or extreme scores
    assert np.isfinite(scores.dropna()).all()
    assert scores.dropna().abs().max() < 100  # Reasonable range


def test_value_missing_ticker_in_fundamentals(sample_prices, sample_fundamentals, value_config):
    """Handle ticker present in prices but missing in fundamentals."""
    # Add a new ticker to prices
    sample_prices['TSLA'] = 200.0

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Original tickers should work
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['MSFT'])
    assert not pd.isna(scores['GOOGL'])

    # TSLA should be NaN (no fundamental data)
    assert pd.isna(scores['TSLA'])


def test_value_single_ticker(sample_prices, sample_fundamentals, value_config):
    """Works with single stock."""
    # Keep only AAPL
    sample_prices = sample_prices[['AAPL']]

    factor = ValueFactor(value_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert len(scores) == 1
    assert 'AAPL' in scores.index
    # Score should be 0 (z-score with single value)
    assert abs(scores['AAPL']) < 0.01


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_value_missing_fundamental_field(sample_prices, value_config):
    """Reject fundamentals missing required fields."""
    # Missing 'earnings'
    bad_fundamentals = {
        'book_value': pd.DataFrame({'AAPL': [1e9]}),
        'market_cap': pd.DataFrame({'AAPL': [1e12]}),
    }

    factor = ValueFactor(value_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "earnings" in str(exc_info.value).lower()


def test_value_fundamentals_not_dict(sample_prices, value_config):
    """Reject non-dict fundamentals."""
    bad_fundamentals = "not a dict"

    factor = ValueFactor(value_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be dict" in str(exc_info.value).lower()


def test_value_fundamental_not_dataframe(sample_prices, value_config):
    """Reject fundamental fields that aren't DataFrames."""
    bad_fundamentals = {
        'book_value': "not a dataframe",
        'market_cap': pd.DataFrame({'AAPL': [1e12]}),
        'earnings': pd.DataFrame({'AAPL': [1e10]}),
    }

    factor = ValueFactor(value_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be DataFrame" in str(exc_info.value)


def test_value_empty_fundamental_dataframe(sample_prices, value_config):
    """Reject empty fundamental DataFrames."""
    bad_fundamentals = {
        'book_value': pd.DataFrame(),
        'market_cap': pd.DataFrame({'AAPL': [1e12]}),
        'earnings': pd.DataFrame({'AAPL': [1e10]}),
    }

    factor = ValueFactor(value_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "empty" in str(exc_info.value).lower()


def test_value_fundamental_no_datetime_index(sample_prices, sample_fundamentals, value_config):
    """Reject fundamentals without DatetimeIndex."""
    # Change index to integers
    sample_fundamentals['book_value'].index = range(len(sample_fundamentals['book_value']))

    factor = ValueFactor(value_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert "DatetimeIndex" in str(exc_info.value)


def test_value_no_fundamental_data_before_date(sample_prices, value_config):
    """Handle case where no fundamental data before as_of_date."""
    # Create fundamentals with future dates only
    future_dates = pd.bdate_range('2025-01-01', periods=100)
    future_fundamentals = {
        'book_value': pd.DataFrame({'AAPL': [1e9]}, index=future_dates[:1]),
        'market_cap': pd.DataFrame({'AAPL': [1e12]}, index=future_dates[:1]),
        'earnings': pd.DataFrame({'AAPL': [1e10]}, index=future_dates[:1]),
    }

    factor = ValueFactor(value_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(sample_prices, future_fundamentals, sample_prices.index[-1])

    assert "No fundamental data available" in str(exc_info.value)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_value_invalid_weight_range():
    """Reject weights outside [0,1]."""
    config = {
        'weight_book_to_market': 1.5,  # Invalid
        'weight_earnings_yield': 0.5,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        ValueFactor(config)

    assert "must be in [0,1]" in str(exc_info.value)


def test_value_zero_weights():
    """Reject all-zero weights."""
    config = {
        'weight_book_to_market': 0.0,
        'weight_earnings_yield': 0.0,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        ValueFactor(config)

    assert "non-zero" in str(exc_info.value).lower()


def test_value_only_book_to_market(sample_prices, sample_fundamentals):
    """Test with only book-to-market component."""
    config = {
        'weight_book_to_market': 1.0,
        'weight_earnings_yield': 0.0,
        'enabled': True
    }

    factor = ValueFactor(config)

    assert factor.weight_book_to_market == 1.0
    assert factor.weight_earnings_yield == 0.0

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert scores.notna().all()


def test_value_only_earnings_yield(sample_prices, sample_fundamentals):
    """Test with only earnings yield component."""
    config = {
        'weight_book_to_market': 0.0,
        'weight_earnings_yield': 1.0,
        'enabled': True
    }

    factor = ValueFactor(config)

    assert factor.weight_book_to_market == 0.0
    assert factor.weight_earnings_yield == 1.0

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert scores.notna().all()


def test_value_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'weight_book_to_market': 0.5,
        'weight_earnings_yield': 0.5,
        'enabled': False,
        'weight': 1.0
    }

    factor = ValueFactor(config)
    assert factor.enabled == False


def test_value_default_config():
    """Test with minimal config (use defaults)."""
    config = {}

    factor = ValueFactor(config)

    assert factor.weight_book_to_market == 0.5
    assert factor.weight_earnings_yield == 0.5
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_value_standalone(sample_prices, sample_fundamentals):
    """Test standalone compute_value function."""
    scores = compute_value(
        sample_prices,
        sample_fundamentals,
        sample_prices.index[-1],
        weight_book_to_market=0.5,
        weight_earnings_yield=0.5
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores.notna().all()


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_value_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 2 years of quarterly data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    prices = pd.DataFrame({
        'AAPL': np.full(n_quarters, 150.0),
        'VALUE_STOCK': np.full(n_quarters, 50.0),
        'GROWTH_STOCK': np.full(n_quarters, 500.0),
    }, index=dates)

    # VALUE_STOCK: High B/M, high E/Y
    # GROWTH_STOCK: Low B/M, low E/Y
    # AAPL: Medium
    fundamentals = {
        'book_value': pd.DataFrame({
            'AAPL': np.full(n_quarters, 50e9),
            'VALUE_STOCK': np.full(n_quarters, 40e9),  # High B/M
            'GROWTH_STOCK': np.full(n_quarters, 10e9), # Low B/M
        }, index=dates),
        'market_cap': pd.DataFrame({
            'AAPL': np.full(n_quarters, 2000e9),
            'VALUE_STOCK': np.full(n_quarters, 100e9),  # Small cap
            'GROWTH_STOCK': np.full(n_quarters, 1000e9), # Large cap
        }, index=dates),
        'earnings': pd.DataFrame({
            'AAPL': np.full(n_quarters, 100e9),
            'VALUE_STOCK': np.full(n_quarters, 8e9),   # High E/Y
            'GROWTH_STOCK': np.full(n_quarters, 10e9), # Low E/Y
        }, index=dates),
    }

    config = {'weight_book_to_market': 0.5, 'weight_earnings_yield': 0.5, 'enabled': True}
    factor = ValueFactor(config)

    scores = factor.compute(prices, fundamentals, dates[-1])

    # VALUE_STOCK should score highest
    assert scores['VALUE_STOCK'] > scores['AAPL']
    assert scores['AAPL'] > scores['GROWTH_STOCK']

    # All scores should be finite
    assert np.isfinite(scores).all()


def test_value_zscore_computation():
    """Test z-score standardization."""
    factor = ValueFactor({'enabled': True})

    # Test with simple series
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=['A', 'B', 'C', 'D', 'E'])
    zscore = factor._compute_zscore(series)

    # Mean should be ~0, std should be ~1
    assert abs(zscore.mean()) < 0.01
    assert abs(zscore.std() - 1.0) < 0.01

    # Test with all same values
    same_series = pd.Series([5.0, 5.0, 5.0], index=['A', 'B', 'C'])
    zscore_same = factor._compute_zscore(same_series)

    # Should all be 0
    assert (zscore_same == 0.0).all()
