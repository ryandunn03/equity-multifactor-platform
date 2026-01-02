"""
Quick smoke test: 6-factor pipeline with real-ish data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from src.factors import (
    MomentumFactor, ValueFactor, QualityFactor,
    VolatilityFactor, SizeFactor, GrowthFactor,
    SectorNormalizer, FactorCompositor
)

# Generate synthetic data
dates = pd.bdate_range('2020-01-01', periods=400)
tickers = [f'STOCK_{i}' for i in range(30)]
np.random.seed(42)
prices = pd.DataFrame(
    np.random.randn(400, 30).cumsum(axis=0) + 100,
    index=dates, columns=tickers
)
sectors = pd.Series({t: ['Tech', 'Health', 'Energy'][i % 3] for i, t in enumerate(tickers)})

# Mock fundamentals - create proper DataFrames for each field
fund_dates = pd.date_range('2020-01-01', periods=8, freq='Q')

fundamentals = {
    'book_value': pd.DataFrame({
        ticker: np.random.uniform(1e9, 10e9, 8) for ticker in tickers
    }, index=fund_dates),
    'market_cap': pd.DataFrame({
        ticker: np.random.uniform(5e9, 500e9, 8) for ticker in tickers
    }, index=fund_dates),
    'earnings': pd.DataFrame({
        ticker: np.random.uniform(1e8, 1e9, 8) for ticker in tickers
    }, index=fund_dates),
    'net_income': pd.DataFrame({
        ticker: np.random.uniform(1e8, 1e9, 8) for ticker in tickers
    }, index=fund_dates),
    'equity': pd.DataFrame({
        ticker: np.random.uniform(1e9, 10e9, 8) for ticker in tickers
    }, index=fund_dates),
    'revenue': pd.DataFrame({
        ticker: np.random.uniform(2e9, 20e9, 8) for ticker in tickers
    }, index=fund_dates),
    'debt': pd.DataFrame({
        ticker: np.random.uniform(1e9, 10e9, 8) for ticker in tickers
    }, index=fund_dates),
}

# Compute all 6 factors
as_of = dates[-1]
print(f"Computing 6 factors as of {as_of.date()}...")

momentum = MomentumFactor({'enabled': True}).compute(prices, {}, as_of)
value = ValueFactor({'enabled': True}).compute(prices, fundamentals, as_of)
quality = QualityFactor({'enabled': True}).compute(prices, fundamentals, as_of)
volatility = VolatilityFactor({'enabled': True}).compute(prices, {}, as_of)
size = SizeFactor({'enabled': True}).compute(prices, fundamentals, as_of)
growth = GrowthFactor({'enabled': True}).compute(prices, fundamentals, as_of)

print(f"✓ All 6 factors computed successfully")

# Combine into DataFrame
all_factors = pd.DataFrame({
    'momentum': momentum,
    'value': value,
    'quality': quality,
    'volatility': volatility,
    'size': size,
    'growth': growth
})

print(f"✓ Factor scores combined into DataFrame: {all_factors.shape}")

# Normalize
normalizer = SectorNormalizer({
    'method': 'sector_relative',
    'winsorize_percentile': 1,
    'handle_missing': 'neutral'
})
normalized = normalizer.normalize(all_factors, sectors)

print(f"✓ Sector normalization complete: {normalized.shape}")

# Combine
compositor = FactorCompositor({'method': 'equal_weighted'})
composite = compositor.combine(normalized)

print(f"✓ Factor combination complete: {len(composite)} stocks")

# Validation
assert len(all_factors.columns) == 6, "Should have 6 factors"
assert len(all_factors) == 30, "Should have 30 stocks"
assert composite.notna().sum() > 0, "Should have valid composite scores"

print(f"\n{'='*60}")
print(f"✅ 6-FACTOR INTEGRATION TEST PASSED")
print(f"{'='*60}")
print(f"   Computed {len(all_factors.columns)} factors for {len(all_factors)} stocks")
print(f"   Composite score range: [{composite.min():.2f}, {composite.max():.2f}]")
print(f"   Valid scores: {composite.notna().sum()}/{len(composite)}")
print(f"\nFactor coverage:")
for col in all_factors.columns:
    valid_pct = all_factors[col].notna().sum() / len(all_factors) * 100
    print(f"   {col:12s}: {valid_pct:5.1f}% ({all_factors[col].notna().sum()}/{len(all_factors)})")
print(f"{'='*60}")
