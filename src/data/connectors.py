"""
Data connectors for fetching market data from various sources.

This module provides connectors to external data sources like Yahoo Finance,
Alpha Vantage, and other market data providers.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
from pathlib import Path


class BaseConnector:
    """
    Base class for data connectors.

    This is a stub implementation for Phase 2. Full implementation will include:
    - Rate limiting and retry logic
    - API key management
    - Response caching
    - Error handling for network issues
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the connector.

        Args:
            config: Configuration dictionary with API keys, timeouts, etc.
        """
        self.config = config or {}

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with price data (index=dates, columns=tickers)
        """
        raise NotImplementedError("Subclasses must implement fetch_prices")

    def fetch_fundamentals(
        self,
        tickers: List[str],
        metrics: List[str],
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch fundamental data for given tickers.

        Args:
            tickers: List of ticker symbols
            metrics: List of fundamental metrics to fetch
            as_of_date: Date for point-in-time data

        Returns:
            DataFrame with fundamental data
        """
        raise NotImplementedError("Subclasses must implement fetch_fundamentals")


class YahooFinanceConnector(BaseConnector):
    """
    Connector for Yahoo Finance data.

    Stub implementation - will use yfinance library in production.
    """

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        # Stub - returns empty DataFrame
        return pd.DataFrame()


class AlphaVantageConnector(BaseConnector):
    """
    Connector for Alpha Vantage API.

    Stub implementation - will use Alpha Vantage API in production.
    """

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alpha Vantage connector.

        Args:
            api_key: Alpha Vantage API key
            config: Additional configuration
        """
        super().__init__(config)
        self.api_key = api_key

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch price data from Alpha Vantage."""
        # Stub - returns empty DataFrame
        return pd.DataFrame()
