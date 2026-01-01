"""
Integration test for complete factor pipeline.

Tests end-to-end flow: compute → normalize → combine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

from src.factors.momentum import MomentumFactor
from src.factors.normalizer import SectorNormalizer
from src.factors.compositor import FactorCompositor


@pytest.fixture
def synthetic_prices():
    """
    Create synthetic price data with known momentum characteristics.
    
    5 sectors, 10 stocks each, 500 trading days
    Each sector has distinct momentum pattern.
    """
    dates = pd.bdate_range('2022-01-01', periods=500)
    n_stocks_per_sector = 10
    
    # Define sector trends
    sector_trends = {
        'Technology': 0.30,      # +30% over period
        'Healthcare': 0.15,      # +15%
        'Consumer': 0.00,        # Flat
        'Energy': -0.15,         # -15%
        'Financials': -0.30,     # -30%
    }
    
    prices_dict = {}
    
    for sector, trend in sector_trends.items():
        for i in range(n_stocks_per_sector):
            ticker = f"{sector[:4].upper()}_{i}"
            
            # Create price series with sector trend + noise
            daily_return = trend / len(dates)
            noise = np.random.normal(0, 0.01, len(dates))
            returns = daily_return + noise
            
            prices = 100 * np.cumprod(1 + returns)
            prices_dict[ticker] = prices
    
    return pd.DataFrame(prices_dict, index=dates)


@pytest.fixture
def sector_mapping(synthetic_prices):
    """Create sector mapping for synthetic stocks."""
    sectors = {}
    
    for ticker in synthetic_prices.columns:
        sector_code = ticker.split('_')[0]
        sector_map = {
            'TECH': 'Technology',
            'HEAL': 'Healthcare',
            'CONS': 'Consumer',
            'ENER': 'Energy',
            'FINA': 'Financials',
        }
        sectors[ticker] = sector_map[sector_code]
    
    return pd.Series(sectors)


@pytest.fixture
def full_config():
    """Load full configuration."""
    config = {
        'factors': {
            'momentum': {
                'lookback_months': 12,
                'skip_months': 1,
                'enabled': True,
                'weight': 1.0
            }
        },
        'normalization': {
            'method': 'sector_relative',
            'winsorize_percentile': 1,
            'handle_missing': 'neutral',
            'max_missing_factors': 2,
        },
        'factor_combination': {
            'method': 'equal_weighted',
            'ic_lookback_months': 12,
            'shrinkage': 0.4,
            'min_ic': -0.5,
        }
    }
    return config


# ============================================================================
# Single-Factor Pipeline Tests
# ============================================================================

def test_complete_pipeline_single_factor(synthetic_prices, sector_mapping, full_config):
    """Test complete pipeline with momentum factor."""
    as_of_date = synthetic_prices.index[-1]
    
    # Step 1: Compute raw momentum scores
    momentum = MomentumFactor(full_config['factors']['momentum'])
    raw_scores = momentum.compute(synthetic_prices, {}, as_of_date)
    
    # Verify raw scores exist
    assert isinstance(raw_scores, pd.Series)
    assert len(raw_scores) == 50  # 5 sectors × 10 stocks
    
    # Verify momentum ranking by sector (Technology > Energy)
    tech_scores = raw_scores[[t for t in raw_scores.index if 'TECH' in t]]
    ener_scores = raw_scores[[t for t in raw_scores.index if 'ENER' in t]]
    assert tech_scores.mean() > ener_scores.mean()
    
    # Step 2: Normalize within sectors
    normalizer = SectorNormalizer(full_config['normalization'])
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sector_mapping
    )
    
    # Verify normalization
    assert isinstance(normalized, pd.DataFrame)
    assert normalized.shape == (50, 1)
    
    # Check mean ≈ 0, std ≈ 1 within each sector
    for sector in sector_mapping.unique():
        sector_mask = sector_mapping == sector
        sector_scores = normalized[sector_mask]['momentum']
        
        assert abs(sector_scores.mean()) < 0.2  # Near zero
        assert abs(sector_scores.std() - 1.0) < 0.3  # Near 1
    
    # Step 3: Combine factors (single factor case)
    compositor = FactorCompositor(full_config['factor_combination'])
    composite = compositor.combine(normalized)
    
    # Verify composite
    assert isinstance(composite, pd.Series)
    assert len(composite) == 50
    
    # Single factor: composite should equal normalized
    pd.testing.assert_series_equal(
        composite,
        normalized['momentum'],
        check_names=False
    )


def test_pipeline_preserves_cross_sector_ranking(synthetic_prices, sector_mapping, full_config):
    """
    Within-sector normalization should preserve cross-sector ordering
    for consistent trends.
    """
    as_of_date = synthetic_prices.index[-1]
    
    # Compute and normalize
    momentum = MomentumFactor(full_config['factors']['momentum'])
    raw_scores = momentum.compute(synthetic_prices, {}, as_of_date)
    
    normalizer = SectorNormalizer(full_config['normalization'])
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sector_mapping
    )
    
    # Average normalized score per sector should still rank correctly
    sector_avg_normalized = {}
    for sector in sector_mapping.unique():
        sector_mask = sector_mapping == sector
        sector_avg_normalized[sector] = normalized[sector_mask]['momentum'].mean()
    
    # Technology (strong uptrend) should have higher avg than Energy (downtrend)
    assert sector_avg_normalized['Technology'] > sector_avg_normalized['Energy']


# ============================================================================
# Multi-Factor Pipeline Tests
# ============================================================================

def test_pipeline_multi_factor(synthetic_prices, sector_mapping, full_config):
    """Test pipeline with multiple synthetic factors."""
    as_of_date = synthetic_prices.index[-1]
    
    # Create two factors: momentum and synthetic "volatility"
    momentum = MomentumFactor(full_config['factors']['momentum'])
    momentum_scores = momentum.compute(synthetic_prices, {}, as_of_date)
    
    # Synthetic volatility factor (inverse of realized vol)
    returns = synthetic_prices.pct_change()
    volatility_scores = -returns.std() * np.sqrt(252)  # Inverted and annualized
    
    # Combine into DataFrame
    factor_scores = pd.DataFrame({
        'momentum': momentum_scores,
        'volatility': volatility_scores,
    })
    
    # Normalize
    normalizer = SectorNormalizer(full_config['normalization'])
    normalized = normalizer.normalize(factor_scores, sector_mapping)
    
    # Verify shape
    assert normalized.shape == (50, 2)
    
    # Combine
    compositor = FactorCompositor(full_config['factor_combination'])
    composite = compositor.combine(normalized)
    
    # Verify composite is average of two factors (equal-weighted)
    expected = normalized.mean(axis=1)
    pd.testing.assert_series_equal(composite, expected, check_names=False)


# ============================================================================
# IC-Weighted Pipeline Test
# ============================================================================

def test_pipeline_with_ic_weighting(synthetic_prices, sector_mapping):
    """Test pipeline with IC-weighted combination."""
    config = {
        'factors': {
            'momentum': {
                'lookback_months': 12,
                'skip_months': 1,
                'enabled': True,
            }
        },
        'normalization': {
            'method': 'sector_relative',
            'winsorize_percentile': 1,
            'handle_missing': 'neutral',
        },
        'factor_combination': {
            'method': 'ic_weighted',
            'ic_lookback_months': 3,
            'shrinkage': 0.4,
            'min_ic': -0.5,
        }
    }
    
    # Simulate multiple periods to build IC history
    momentum = MomentumFactor(config['factors']['momentum'])
    normalizer = SectorNormalizer(config['normalization'])
    compositor = FactorCompositor(config['factor_combination'])
    
    # Use last 50 days as multiple "rebalance" periods
    test_dates = synthetic_prices.index[-50::10]  # Every 10 days
    
    for i, date in enumerate(test_dates[:-1]):
        # Compute momentum
        scores = momentum.compute(synthetic_prices, {}, date)
        
        # Normalize
        normalized = normalizer.normalize(
            pd.DataFrame({'momentum': scores}),
            sector_mapping
        )
        
        # Compute forward returns to next period
        next_date = test_dates[i + 1]
        forward_returns = (
            synthetic_prices.loc[next_date] / synthetic_prices.loc[date] - 1
        )
        
        # Update IC history
        compositor.update_ic_history(normalized, forward_returns, date)
        
        # Combine (will use IC weights once history sufficient)
        composite = compositor.combine(normalized)
    
    # Verify IC history was built
    assert 'momentum' in compositor.ic_history
    assert len(compositor.ic_history['momentum']) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_pipeline_with_missing_prices(synthetic_prices, sector_mapping, full_config):
    """Test pipeline handles missing price data."""
    prices_with_gaps = synthetic_prices.copy()
    
    # Introduce gaps in some tickers
    prices_with_gaps.iloc[100:110, 0:5] = np.nan
    
    as_of_date = prices_with_gaps.index[-1]
    
    # Should still work (momentum handles gaps)
    momentum = MomentumFactor(full_config['factors']['momentum'])
    raw_scores = momentum.compute(prices_with_gaps, {}, as_of_date)
    
    # Some scores may be NaN, but pipeline should continue
    assert not raw_scores.isna().all()
    
    # Normalize (neutral policy handles NaN)
    normalizer = SectorNormalizer(full_config['normalization'])
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sector_mapping
    )
    
    # Should have scores (NaN replaced with 0)
    assert not normalized.isna().all().all()


def test_pipeline_with_small_sector(full_config):
    """Test pipeline with sector having only 2 stocks."""
    # Small dataset
    dates = pd.bdate_range('2022-01-01', periods=300)
    
    prices = pd.DataFrame({
        'A': np.linspace(100, 120, 300),
        'B': np.linspace(100, 125, 300),
        'C': np.linspace(100, 80, 300),  # Different sector
    }, index=dates)
    
    sectors = pd.Series({
        'A': 'Large',
        'B': 'Large',
        'C': 'Small',  # Singleton sector
    })
    
    as_of_date = dates[-1]
    
    # Compute momentum
    momentum = MomentumFactor(full_config['factors']['momentum'])
    raw_scores = momentum.compute(prices, {}, as_of_date)
    
    # Normalize (small sector should get neutral or reasonable scores)
    normalizer = SectorNormalizer(full_config['normalization'])
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sectors
    )
    
    # Should work without error
    assert normalized.shape == (3, 1)
    assert not normalized.isna().all().all()


# ============================================================================
# Data Quality Tests
# ============================================================================

def test_pipeline_output_shapes(synthetic_prices, sector_mapping, full_config):
    """Verify all output shapes are consistent."""
    as_of_date = synthetic_prices.index[-1]
    
    momentum = MomentumFactor(full_config['factors']['momentum'])
    normalizer = SectorNormalizer(full_config['normalization'])
    compositor = FactorCompositor(full_config['factor_combination'])
    
    # Raw scores
    raw_scores = momentum.compute(synthetic_prices, {}, as_of_date)
    assert len(raw_scores) == 50
    
    # Normalized scores
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sector_mapping
    )
    assert normalized.shape == (50, 1)
    assert set(normalized.index) == set(raw_scores.index)
    
    # Composite scores
    composite = compositor.combine(normalized)
    assert len(composite) == 50
    assert set(composite.index) == set(normalized.index)


def test_pipeline_no_data_loss(synthetic_prices, sector_mapping, full_config):
    """Verify no tickers are dropped unexpectedly."""
    as_of_date = synthetic_prices.index[-1]
    
    momentum = MomentumFactor(full_config['factors']['momentum'])
    normalizer = SectorNormalizer(full_config['normalization'])
    compositor = FactorCompositor(full_config['factor_combination'])
    
    # Track tickers through pipeline
    original_tickers = set(synthetic_prices.columns)
    
    raw_scores = momentum.compute(synthetic_prices, {}, as_of_date)
    assert set(raw_scores.index) == original_tickers
    
    normalized = normalizer.normalize(
        pd.DataFrame({'momentum': raw_scores}),
        sector_mapping
    )
    assert set(normalized.index) == original_tickers
    
    composite = compositor.combine(normalized)
    assert set(composite.index) == original_tickers


# ============================================================================
# Realistic Scenario Test
# ============================================================================

def test_pipeline_realistic_backtest_simulation():
    """Simulate realistic multi-period backtest workflow."""
    # Generate 3 years of daily data
    dates = pd.bdate_range('2021-01-01', '2024-01-01')
    np.random.seed(42)
    
    # 30 stocks across 3 sectors
    tickers = [f"TECH_{i}" for i in range(10)] + \
              [f"HEAL_{i}" for i in range(10)] + \
              [f"ENER_{i}" for i in range(10)]
    
    # Random walk with drift
    prices_dict = {}
    for ticker in tickers:
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices_dict[ticker] = 100 * np.cumprod(1 + returns)
    
    prices = pd.DataFrame(prices_dict, index=dates)
    
    sectors = pd.Series({
        **{f"TECH_{i}": 'Technology' for i in range(10)},
        **{f"HEAL_{i}": 'Healthcare' for i in range(10)},
        **{f"ENER_{i}": 'Energy' for i in range(10)},
    })
    
    # Configuration
    config = {
        'factors': {'momentum': {'lookback_months': 12, 'skip_months': 1, 'enabled': True}},
        'normalization': {'method': 'sector_relative', 'winsorize_percentile': 1, 'handle_missing': 'neutral'},
        'factor_combination': {'method': 'equal_weighted', 'ic_lookback_months': 12, 'shrinkage': 0.4},
    }
    
    # Simulate monthly rebalances
    rebalance_dates = pd.date_range(
        dates[290],  # Start after sufficient history (12m + 1m skip + 30d buffer)
        dates[-1],
        freq='M'
    )
    
    results = []
    
    for rebal_date in rebalance_dates[:5]:  # Test first 5 rebalances
        # Find closest trading day
        closest_date = prices.index[prices.index.get_indexer([rebal_date], method='nearest')[0]]
        
        # Run pipeline
        momentum = MomentumFactor(config['factors']['momentum'])
        raw_scores = momentum.compute(prices, {}, closest_date)
        
        normalizer = SectorNormalizer(config['normalization'])
        normalized = normalizer.normalize(
            pd.DataFrame({'momentum': raw_scores}),
            sectors
        )
        
        compositor = FactorCompositor(config['factor_combination'])
        composite = compositor.combine(normalized)
        
        results.append({
            'date': closest_date,
            'n_scores': len(composite),
            'mean_score': composite.mean(),
            'std_score': composite.std(),
        })
    
    # Verify all rebalances succeeded
    assert len(results) == 5
    for result in results:
        assert result['n_scores'] == 30
        assert not np.isnan(result['mean_score'])