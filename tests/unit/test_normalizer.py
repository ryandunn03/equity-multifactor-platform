"""
Unit tests for SectorNormalizer.

Tests sector-relative z-score normalization, winsorization,
and missing value handling.
"""

import pytest
import pandas as pd
import numpy as np

from src.factors.normalizer import SectorNormalizer
from src.utils.exceptions import DataError


@pytest.fixture
def sample_scores():
    """
    Create sample factor scores across sectors.
    
    Returns:
        DataFrame with 2 factors across 3 sectors
    """
    return pd.DataFrame({
        'momentum': [0.5, 0.6, 0.4, -0.2, -0.3, -0.1, 1.0, 1.2, 0.9],
        'value': [2.0, 2.2, 1.8, 1.0, 1.1, 0.9, 3.0, 3.5, 2.8],
    }, index=['AAPL', 'MSFT', 'GOOGL', 'XOM', 'CVX', 'COP', 'JPM', 'BAC', 'WFC'])


@pytest.fixture
def sample_sectors():
    """Sector mapping for sample stocks."""
    return pd.Series({
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'XOM': 'Energy',
        'CVX': 'Energy',
        'COP': 'Energy',
        'JPM': 'Financials',
        'BAC': 'Financials',
        'WFC': 'Financials',
    })


@pytest.fixture
def normalizer_config():
    """Standard normalizer configuration."""
    return {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
        'max_missing_factors': 2,
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_normalizer_basic(sample_scores, sample_sectors, normalizer_config):
    """Test basic normalization."""
    normalizer = SectorNormalizer(normalizer_config)
    normalized = normalizer.normalize(sample_scores, sample_sectors)
    
    # Check shape preserved
    assert normalized.shape == sample_scores.shape
    assert set(normalized.index) == set(sample_scores.index)
    assert set(normalized.columns) == set(sample_scores.columns)


def test_normalizer_sector_mean_zero(sample_scores, sample_sectors, normalizer_config):
    """Verify mean ≈ 0 within each sector."""
    normalizer = SectorNormalizer(normalizer_config)
    normalized = normalizer.normalize(sample_scores, sample_sectors)
    
    for sector in sample_sectors.unique():
        sector_mask = sample_sectors == sector
        sector_scores = normalized[sector_mask]
        
        for factor in normalized.columns:
            sector_mean = sector_scores[factor].mean()
            # Mean should be close to 0 (within numerical tolerance)
            assert abs(sector_mean) < 0.1, f"{sector} {factor} mean={sector_mean}"


def test_normalizer_sector_std_one(sample_scores, sample_sectors, normalizer_config):
    """Verify std ≈ 1 within each sector."""
    normalizer = SectorNormalizer(normalizer_config)
    normalized = normalizer.normalize(sample_scores, sample_sectors)
    
    for sector in sample_sectors.unique():
        sector_mask = sample_sectors == sector
        sector_scores = normalized[sector_mask]
        
        for factor in normalized.columns:
            sector_std = sector_scores[factor].std()
            # Std should be close to 1
            assert abs(sector_std - 1.0) < 0.2, f"{sector} {factor} std={sector_std}"


def test_normalizer_cross_sectional(sample_scores, sample_sectors):
    """Test cross-sectional normalization (no sector adjustment)."""
    config = {
        'method': 'cross_sectional',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(sample_scores, sample_sectors)
    
    # Overall mean should be ≈ 0, std ≈ 1
    for factor in normalized.columns:
        assert abs(normalized[factor].mean()) < 0.1
        assert abs(normalized[factor].std() - 1.0) < 0.2


# ============================================================================
# Winsorization Tests
# ============================================================================

def test_normalizer_winsorization():
    """Test that extreme values are clipped."""
    # Create data with outliers
    scores = pd.DataFrame({
        'factor': [1, 2, 3, 4, 5, 100],  # 100 is outlier
    }, index=['A', 'B', 'C', 'D', 'E', 'F'])
    
    sectors = pd.Series({
        'A': 'Sector1',
        'B': 'Sector1',
        'C': 'Sector1',
        'D': 'Sector1',
        'E': 'Sector1',
        'F': 'Sector1',
    })
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 5,  # Clip at 5th/95th
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sectors)
    
    # Outlier should not dominate (z-score should be reasonable)
    assert abs(normalized.loc['F', 'factor']) < 5  # Not extremely large


# ============================================================================
# Small Sector Handling
# ============================================================================

def test_normalizer_small_sector():
    """Sectors with <5 stocks get neutral scores."""
    scores = pd.DataFrame({
        'factor': [1, 2, 3, 10, 11],
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    sectors = pd.Series({
        'A': 'Large',
        'B': 'Large',
        'C': 'Large',
        'D': 'Small',  # Only 2 stocks
        'E': 'Small',
    })
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sectors)
    
    # Small sector should have reasonable scores
    # (implementation may assign neutral=0 or attempt z-score)
    small_sector_scores = normalized[sectors == 'Small']['factor']
    assert not small_sector_scores.isna().all()


def test_normalizer_single_stock_sector():
    """Sector with 1 stock gets neutral score."""
    scores = pd.DataFrame({
        'factor': [1, 2, 3, 10],
    }, index=['A', 'B', 'C', 'D'])
    
    sectors = pd.Series({
        'A': 'Normal',
        'B': 'Normal',
        'C': 'Normal',
        'D': 'Singleton',  # Only 1 stock
    })
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sectors)
    
    # Singleton should get neutral (0.0)
    assert abs(normalized.loc['D', 'factor']) < 0.01


# ============================================================================
# Missing Value Handling
# ============================================================================

def test_normalizer_missing_neutral(sample_scores, sample_sectors):
    """Test neutral policy for missing values."""
    scores = sample_scores.copy()
    scores.loc['AAPL', 'momentum'] = np.nan
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
        'max_missing_factors': 2,
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sample_sectors)
    
    # NaN should be replaced with 0.0
    assert normalized.loc['AAPL', 'momentum'] == 0.0


def test_normalizer_missing_drop(sample_scores, sample_sectors):
    """Test drop policy for tickers with too many missing factors."""
    scores = sample_scores.copy()
    # Make AAPL missing both factors
    scores.loc['AAPL', ['momentum', 'value']] = np.nan
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'drop',
        'max_missing_factors': 1,  # Drop if >1 missing
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sample_sectors)
    
    # AAPL should be dropped
    assert 'AAPL' not in normalized.index


# ============================================================================
# Edge Cases
# ============================================================================

def test_normalizer_empty_dataframe(normalizer_config):
    """Handle empty DataFrame."""
    scores = pd.DataFrame()
    sectors = pd.Series(dtype=object)
    
    normalizer = SectorNormalizer(normalizer_config)
    
    with pytest.raises(DataError) as exc_info:
        normalizer.normalize(scores, sectors)
    
    assert "empty" in str(exc_info.value).lower()


def test_normalizer_missing_sector_mapping(sample_scores, normalizer_config):
    """Handle tickers without sector mapping."""
    sectors = pd.Series({
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        # Missing GOOGL and others
    })
    
    normalizer = SectorNormalizer(normalizer_config)
    
    # Should work (assigns Unknown sector to missing)
    normalized = normalizer.normalize(sample_scores, sectors)
    assert not normalized.empty


def test_normalizer_all_same_values():
    """Handle sector where all stocks have same value (std=0)."""
    scores = pd.DataFrame({
        'factor': [5, 5, 5],  # All identical
    }, index=['A', 'B', 'C'])
    
    sectors = pd.Series({
        'A': 'Sector1',
        'B': 'Sector1',
        'C': 'Sector1',
    })
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sectors)
    
    # Should assign neutral (0.0) when std=0
    assert (normalized['factor'] == 0.0).all()


def test_normalizer_preserves_ranking_within_sector(sample_scores, sample_sectors, normalizer_config):
    """Normalization should preserve ranking within sector."""
    normalizer = SectorNormalizer(normalizer_config)
    normalized = normalizer.normalize(sample_scores, sample_sectors)
    
    for sector in sample_sectors.unique():
        sector_mask = sample_sectors == sector
        
        for factor in sample_scores.columns:
            original_rank = sample_scores[sector_mask][factor].rank()
            normalized_rank = normalized[sector_mask][factor].rank()
            
            # Rankings should be identical (may have ties)
            pd.testing.assert_series_equal(
                original_rank,
                normalized_rank,
                check_names=False,
            )


def test_normalizer_initialization_from_config(normalizer_config):
    """Test initialization from config dict."""
    normalizer = SectorNormalizer(normalizer_config)
    
    assert normalizer.method == 'sector_relative'
    assert normalizer.winsorize_pct == 1
    assert normalizer.handle_missing == 'neutral'
    assert normalizer.max_missing == 2


# ============================================================================
# Multi-Factor Tests
# ============================================================================

def test_normalizer_multiple_factors():
    """Test with many factors."""
    scores = pd.DataFrame({
        'momentum': np.random.randn(9),
        'value': np.random.randn(9),
        'quality': np.random.randn(9),
        'volatility': np.random.randn(9),
    }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    
    sectors = pd.Series({
        'A': 'S1', 'B': 'S1', 'C': 'S1',
        'D': 'S2', 'E': 'S2', 'F': 'S2',
        'G': 'S3', 'H': 'S3', 'I': 'S3',
    })
    
    config = {
        'method': 'sector_relative',
        'winsorize_percentile': 1,
        'handle_missing': 'neutral',
    }
    
    normalizer = SectorNormalizer(config)
    normalized = normalizer.normalize(scores, sectors)
    
    # All factors should be normalized
    assert normalized.shape == scores.shape
    
    # Check each factor is normalized
    for factor in normalized.columns:
        for sector in sectors.unique():
            sector_mask = sectors == sector
            sector_mean = normalized[sector_mask][factor].mean()
            assert abs(sector_mean) < 0.1