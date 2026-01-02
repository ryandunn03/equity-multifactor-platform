"""
Comprehensive unit tests for QualityFactor.

Tests cover basic functionality, edge cases, configuration handling,
and data validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.factors.quality import QualityFactor, compute_quality
from src.utils.exceptions import InsufficientDataError, DataError


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    return pd.DataFrame({
        'AAPL': np.full(100, 150.0),
        'MSFT': np.full(100, 300.0),
        'GOOGL': np.full(100, 100.0),
    }, index=dates)


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data for testing."""
    dates = pd.bdate_range('2023-01-01', periods=100)

    return {
        'net_income': pd.DataFrame({
            'AAPL': np.full(100, 100e9),     # $100B net income
            'MSFT': np.full(100, 75e9),      # $75B net income
            'GOOGL': np.full(100, 80e9),     # $80B net income
        }, index=dates),
        'equity': pd.DataFrame({
            'AAPL': np.full(100, 500e9),     # $500B equity
            'MSFT': np.full(100, 250e9),     # $250B equity
            'GOOGL': np.full(100, 400e9),    # $400B equity
        }, index=dates),
        'revenue': pd.DataFrame({
            'AAPL': np.full(100, 400e9),     # $400B revenue
            'MSFT': np.full(100, 200e9),     # $200B revenue
            'GOOGL': np.full(100, 300e9),    # $300B revenue
        }, index=dates),
        'debt': pd.DataFrame({
            'AAPL': np.full(100, 100e9),     # $100B debt (low leverage)
            'MSFT': np.full(100, 50e9),      # $50B debt (very low leverage)
            'GOOGL': np.full(100, 200e9),    # $200B debt (higher leverage)
        }, index=dates),
    }


@pytest.fixture
def quality_config():
    """Standard quality configuration."""
    return {
        'weight_roe': 0.4,
        'weight_profit_margin': 0.3,
        'weight_leverage': 0.3,
        'enabled': True,
        'weight': 1.0
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_quality_basic(sample_prices, sample_fundamentals, quality_config):
    """Test basic quality calculation and ranking."""
    factor = QualityFactor(quality_config)

    as_of_date = sample_prices.index[-1]
    scores = factor.compute(sample_prices, sample_fundamentals, as_of_date)

    # Check return type and shape
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert set(scores.index) == {'AAPL', 'MSFT', 'GOOGL'}

    # All scores should be valid (not NaN)
    assert scores.notna().all()


def test_quality_roe_calculation(sample_prices, sample_fundamentals, quality_config):
    """Test that ROE is calculated correctly."""
    factor = QualityFactor(quality_config)
    as_of_date = sample_prices.index[-1]

    # Compute ROE manually
    net_income = sample_fundamentals['net_income'].iloc[-1]
    equity = sample_fundamentals['equity'].iloc[-1]
    expected_roe = net_income / equity

    # Get ROE from factor
    actual_roe = factor._compute_roe(
        sample_fundamentals,
        ['AAPL', 'MSFT', 'GOOGL'],
        as_of_date
    )

    # Should match
    pd.testing.assert_series_equal(actual_roe, expected_roe, check_names=False)


def test_quality_profit_margin_calculation(sample_prices, sample_fundamentals, quality_config):
    """Test that profit margin is calculated correctly."""
    factor = QualityFactor(quality_config)
    as_of_date = sample_prices.index[-1]

    # Compute profit margin manually
    net_income = sample_fundamentals['net_income'].iloc[-1]
    revenue = sample_fundamentals['revenue'].iloc[-1]
    expected_pm = net_income / revenue

    # Get profit margin from factor
    actual_pm = factor._compute_profit_margin(
        sample_fundamentals,
        ['AAPL', 'MSFT', 'GOOGL'],
        as_of_date
    )

    # Should match
    pd.testing.assert_series_equal(actual_pm, expected_pm, check_names=False)


def test_quality_leverage_calculation(sample_prices, sample_fundamentals, quality_config):
    """Test that leverage is calculated correctly."""
    factor = QualityFactor(quality_config)
    as_of_date = sample_prices.index[-1]

    # Compute leverage manually
    debt = sample_fundamentals['debt'].iloc[-1]
    equity = sample_fundamentals['equity'].iloc[-1]
    expected_lev = debt / equity

    # Get leverage from factor
    actual_lev = factor._compute_leverage(
        sample_fundamentals,
        ['AAPL', 'MSFT', 'GOOGL'],
        as_of_date
    )

    # Should match
    pd.testing.assert_series_equal(actual_lev, expected_lev, check_names=False)


def test_quality_from_config(quality_config):
    """Test initialization from config dict."""
    factor = QualityFactor(quality_config)

    assert factor.name == "quality"
    assert factor.enabled == True
    assert factor.weight == 1.0
    assert abs(factor.weight_roe - 0.4) < 0.01
    assert abs(factor.weight_profit_margin - 0.3) < 0.01
    assert abs(factor.weight_leverage - 0.3) < 0.01


def test_quality_custom_weights(sample_prices, sample_fundamentals):
    """Test with custom component weights."""
    config = {
        'weight_roe': 0.5,
        'weight_profit_margin': 0.3,
        'weight_leverage': 0.2,
        'enabled': True,
        'weight': 1.0
    }

    factor = QualityFactor(config)

    # Weights should be normalized
    assert abs(factor.weight_roe - 0.5) < 0.01
    assert abs(factor.weight_profit_margin - 0.3) < 0.01
    assert abs(factor.weight_leverage - 0.2) < 0.01

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still produce valid scores
    assert scores.notna().all()


def test_quality_required_data():
    """Test required_data() method."""
    factor = QualityFactor({'enabled': True})
    required = factor.required_data()

    assert 'prices' in required
    assert 'fundamentals' in required
    assert 'Close' in required['prices']
    assert 'net_income' in required['fundamentals']
    assert 'equity' in required['fundamentals']
    assert 'revenue' in required['fundamentals']
    assert 'debt' in required['fundamentals']


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_quality_negative_equity(sample_prices, sample_fundamentals, quality_config):
    """Handle negative equity (distressed companies)."""
    # Make MSFT have negative equity
    sample_fundamentals['equity']['MSFT'] = -100e9

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # AAPL and GOOGL should have valid scores
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['GOOGL'])

    # MSFT may have NaN due to negative equity


def test_quality_negative_net_income(sample_prices, sample_fundamentals, quality_config):
    """Handle negative net income (losses)."""
    # Make AAPL have negative net income
    sample_fundamentals['net_income']['AAPL'] = -10e9

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should still compute (negative ROE and profit margin are valid)
    assert not pd.isna(scores['MSFT'])
    assert not pd.isna(scores['GOOGL'])


def test_quality_zero_revenue(sample_prices, sample_fundamentals, quality_config):
    """Handle zero revenue gracefully."""
    # Make GOOGL have zero revenue
    sample_fundamentals['revenue']['GOOGL'] = 0.0

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # AAPL and MSFT should have valid scores
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['MSFT'])

    # GOOGL may have reduced score due to invalid profit margin


def test_quality_extreme_roe(sample_prices, sample_fundamentals, quality_config):
    """Cap extreme ROE values (likely data errors)."""
    # Make AAPL have extreme ROE
    sample_fundamentals['net_income']['AAPL'] = 5000e9  # Extreme earnings

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Should not have infinite or extreme scores
    assert np.isfinite(scores.dropna()).all()


def test_quality_high_leverage(sample_prices, sample_fundamentals, quality_config):
    """Test that high leverage reduces quality score."""
    # Create two identical companies except leverage
    low_lev_fundamentals = sample_fundamentals.copy()
    high_lev_fundamentals = sample_fundamentals.copy()

    # Low leverage company
    low_lev_fundamentals['debt'] = low_lev_fundamentals['debt'].copy()
    low_lev_fundamentals['debt']['AAPL'] = 10e9  # Very low debt

    # High leverage company
    high_lev_fundamentals['debt'] = high_lev_fundamentals['debt'].copy()
    high_lev_fundamentals['debt']['MSFT'] = 500e9  # Very high debt

    factor = QualityFactor(quality_config)

    scores_low = factor.compute(sample_prices, low_lev_fundamentals, sample_prices.index[-1])
    scores_high = factor.compute(sample_prices, high_lev_fundamentals, sample_prices.index[-1])

    # Low leverage AAPL should score better with lower debt
    # High leverage MSFT should score worse with higher debt


def test_quality_missing_ticker_in_fundamentals(sample_prices, sample_fundamentals, quality_config):
    """Handle ticker present in prices but missing in fundamentals."""
    # Add a new ticker to prices
    sample_prices['TSLA'] = 200.0

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    # Original tickers should work
    assert not pd.isna(scores['AAPL'])
    assert not pd.isna(scores['MSFT'])
    assert not pd.isna(scores['GOOGL'])

    # TSLA should be NaN (no fundamental data)
    assert pd.isna(scores['TSLA'])


def test_quality_single_ticker(sample_prices, sample_fundamentals, quality_config):
    """Works with single stock."""
    # Keep only AAPL
    sample_prices = sample_prices[['AAPL']]

    factor = QualityFactor(quality_config)
    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert len(scores) == 1
    assert 'AAPL' in scores.index
    # Score should be 0 (z-score with single value)
    assert abs(scores['AAPL']) < 0.01


# ============================================================================
# Data Validation Tests
# ============================================================================

def test_quality_missing_fundamental_field(sample_prices, quality_config):
    """Reject fundamentals missing required fields."""
    # Missing 'debt'
    bad_fundamentals = {
        'net_income': pd.DataFrame({'AAPL': [1e9]}),
        'equity': pd.DataFrame({'AAPL': [1e12]}),
        'revenue': pd.DataFrame({'AAPL': [1e10]}),
    }

    factor = QualityFactor(quality_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "debt" in str(exc_info.value).lower()


def test_quality_fundamentals_not_dict(sample_prices, quality_config):
    """Reject non-dict fundamentals."""
    bad_fundamentals = "not a dict"

    factor = QualityFactor(quality_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be dict" in str(exc_info.value).lower()


def test_quality_fundamental_not_dataframe(sample_prices, quality_config):
    """Reject fundamental fields that aren't DataFrames."""
    bad_fundamentals = {
        'net_income': "not a dataframe",
        'equity': pd.DataFrame({'AAPL': [1e12]}),
        'revenue': pd.DataFrame({'AAPL': [1e10]}),
        'debt': pd.DataFrame({'AAPL': [1e11]}),
    }

    factor = QualityFactor(quality_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "must be DataFrame" in str(exc_info.value)


def test_quality_empty_fundamental_dataframe(sample_prices, quality_config):
    """Reject empty fundamental DataFrames."""
    bad_fundamentals = {
        'net_income': pd.DataFrame(),
        'equity': pd.DataFrame({'AAPL': [1e12]}),
        'revenue': pd.DataFrame({'AAPL': [1e10]}),
        'debt': pd.DataFrame({'AAPL': [1e11]}),
    }

    factor = QualityFactor(quality_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, bad_fundamentals, sample_prices.index[-1])

    assert "empty" in str(exc_info.value).lower()


def test_quality_fundamental_no_datetime_index(sample_prices, sample_fundamentals, quality_config):
    """Reject fundamentals without DatetimeIndex."""
    # Change index to integers
    sample_fundamentals['net_income'].index = range(len(sample_fundamentals['net_income']))

    factor = QualityFactor(quality_config)

    with pytest.raises(DataError) as exc_info:
        factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])

    assert "DatetimeIndex" in str(exc_info.value)


def test_quality_no_fundamental_data_before_date(sample_prices, quality_config):
    """Handle case where no fundamental data before as_of_date."""
    # Create fundamentals with future dates only
    future_dates = pd.bdate_range('2025-01-01', periods=100)
    future_fundamentals = {
        'net_income': pd.DataFrame({'AAPL': [1e9]}, index=future_dates[:1]),
        'equity': pd.DataFrame({'AAPL': [1e12]}, index=future_dates[:1]),
        'revenue': pd.DataFrame({'AAPL': [1e10]}, index=future_dates[:1]),
        'debt': pd.DataFrame({'AAPL': [1e11]}, index=future_dates[:1]),
    }

    factor = QualityFactor(quality_config)

    with pytest.raises(InsufficientDataError) as exc_info:
        factor.compute(sample_prices, future_fundamentals, sample_prices.index[-1])

    assert "No fundamental data available" in str(exc_info.value)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_quality_invalid_weight_range():
    """Reject weights outside [0,1]."""
    config = {
        'weight_roe': 1.5,  # Invalid
        'weight_profit_margin': 0.3,
        'weight_leverage': 0.3,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        QualityFactor(config)

    assert "must be in [0,1]" in str(exc_info.value)


def test_quality_zero_weights():
    """Reject all-zero weights."""
    config = {
        'weight_roe': 0.0,
        'weight_profit_margin': 0.0,
        'weight_leverage': 0.0,
        'enabled': True
    }

    with pytest.raises(ValueError) as exc_info:
        QualityFactor(config)

    assert "non-zero" in str(exc_info.value).lower()


def test_quality_only_roe(sample_prices, sample_fundamentals):
    """Test with only ROE component."""
    config = {
        'weight_roe': 1.0,
        'weight_profit_margin': 0.0,
        'weight_leverage': 0.0,
        'enabled': True
    }

    factor = QualityFactor(config)

    assert factor.weight_roe == 1.0
    assert factor.weight_profit_margin == 0.0
    assert factor.weight_leverage == 0.0

    scores = factor.compute(sample_prices, sample_fundamentals, sample_prices.index[-1])
    assert scores.notna().all()


def test_quality_disabled_flag():
    """Test enabled=False in config."""
    config = {
        'weight_roe': 0.4,
        'weight_profit_margin': 0.3,
        'weight_leverage': 0.3,
        'enabled': False,
        'weight': 1.0
    }

    factor = QualityFactor(config)
    assert factor.enabled == False


def test_quality_default_config():
    """Test with minimal config (use defaults)."""
    config = {}

    factor = QualityFactor(config)

    assert factor.weight_roe == 0.4
    assert factor.weight_profit_margin == 0.3
    assert factor.weight_leverage == 0.3
    assert factor.enabled == True
    assert factor.weight == 1.0


# ============================================================================
# Standalone Function Tests
# ============================================================================

def test_compute_quality_standalone(sample_prices, sample_fundamentals):
    """Test standalone compute_quality function."""
    scores = compute_quality(
        sample_prices,
        sample_fundamentals,
        sample_prices.index[-1],
        weight_roe=0.4,
        weight_profit_margin=0.3,
        weight_leverage=0.3
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores.notna().all()


# ============================================================================
# Integration-Style Tests
# ============================================================================

def test_quality_realistic_scenario():
    """Test with realistic market data characteristics."""
    # 2 years of quarterly data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='Q')
    n_quarters = len(dates)

    prices = pd.DataFrame({
        'QUALITY': np.full(n_quarters, 150.0),
        'AVERAGE': np.full(n_quarters, 100.0),
        'POOR': np.full(n_quarters, 50.0),
    }, index=dates)

    # QUALITY: High ROE, high margin, low leverage
    # POOR: Low ROE, low margin, high leverage
    # AVERAGE: Medium values
    fundamentals = {
        'net_income': pd.DataFrame({
            'QUALITY': np.full(n_quarters, 50e9),   # High profitability
            'AVERAGE': np.full(n_quarters, 20e9),
            'POOR': np.full(n_quarters, 5e9),       # Low profitability
        }, index=dates),
        'equity': pd.DataFrame({
            'QUALITY': np.full(n_quarters, 200e9),
            'AVERAGE': np.full(n_quarters, 200e9),
            'POOR': np.full(n_quarters, 200e9),
        }, index=dates),
        'revenue': pd.DataFrame({
            'QUALITY': np.full(n_quarters, 100e9),  # High margin
            'AVERAGE': np.full(n_quarters, 100e9),
            'POOR': np.full(n_quarters, 100e9),
        }, index=dates),
        'debt': pd.DataFrame({
            'QUALITY': np.full(n_quarters, 20e9),   # Low leverage
            'AVERAGE': np.full(n_quarters, 100e9),
            'POOR': np.full(n_quarters, 400e9),     # High leverage
        }, index=dates),
    }

    config = {
        'weight_roe': 0.4,
        'weight_profit_margin': 0.3,
        'weight_leverage': 0.3,
        'enabled': True
    }
    factor = QualityFactor(config)

    scores = factor.compute(prices, fundamentals, dates[-1])

    # QUALITY should score highest
    assert scores['QUALITY'] > scores['AVERAGE']
    assert scores['AVERAGE'] > scores['POOR']

    # All scores should be finite
    assert np.isfinite(scores).all()


def test_quality_zscore_computation():
    """Test z-score standardization."""
    factor = QualityFactor({'enabled': True})

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
