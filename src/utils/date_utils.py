"""Date and business day utilities for the equity factor platform.

Provides functions for handling trading days, business calendars,
and point-in-time data alignment.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd


def get_business_days(
    start_date: datetime,
    end_date: datetime,
) -> pd.DatetimeIndex:
    """
    Get all business days between start and end dates.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        DatetimeIndex of business days
    
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 31)
        >>> days = get_business_days(start, end)
        >>> len(days)  # Approximately 22-23 business days
    """
    return pd.bdate_range(start=start_date, end=end_date)


def get_month_ends(
    start_date: datetime,
    end_date: datetime,
) -> List[datetime]:
    """
    Get all month-end dates between start and end.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of month-end datetimes
    
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 12, 31)
        >>> month_ends = get_month_ends(start, end)
        >>> len(month_ends)  # 12 months
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    return [d.to_pydatetime() for d in dates]


def get_quarter_ends(
    start_date: datetime,
    end_date: datetime,
) -> List[datetime]:
    """
    Get all quarter-end dates between start and end.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of quarter-end datetimes
    
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 12, 31)
        >>> quarter_ends = get_quarter_ends(start, end)
        >>> len(quarter_ends)  # 4 quarters
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    return [d.to_pydatetime() for d in dates]


def add_business_days(
    date: datetime,
    days: int,
) -> datetime:
    """
    Add business days to a date.
    
    Args:
        date: Starting date
        days: Number of business days to add (can be negative)
    
    Returns:
        New date after adding business days
    
    Example:
        >>> date = datetime(2024, 1, 1)  # Monday
        >>> new_date = add_business_days(date, 5)
        >>> # Returns following Monday (skipping weekend)
    """
    if days == 0:
        return date
    
    # Use pandas business day offset
    offset = pd.offsets.BDay(days)
    result = pd.Timestamp(date) + offset
    return result.to_pydatetime()


def get_previous_month_end(date: datetime) -> datetime:
    """
    Get the previous month-end date.
    
    Args:
        date: Reference date
    
    Returns:
        Previous month-end datetime
    
    Example:
        >>> date = datetime(2024, 1, 15)
        >>> prev_month_end = get_previous_month_end(date)
        >>> prev_month_end  # datetime(2023, 12, 31)
    """
    # Go to first of current month, then subtract one day
    first_of_month = date.replace(day=1)
    return first_of_month - timedelta(days=1)


def apply_reporting_lag(
    date: datetime,
    lag_days: int = 60,
) -> datetime:
    """
    Apply reporting lag to a date for point-in-time data alignment.
    
    Args:
        date: Original date (e.g., quarter end)
        lag_days: Number of calendar days lag (default 60)
    
    Returns:
        Date after applying lag
    
    Example:
        >>> quarter_end = datetime(2024, 3, 31)
        >>> available_date = apply_reporting_lag(quarter_end, 60)
        >>> available_date  # datetime(2024, 5, 30)
    """
    return date + timedelta(days=lag_days)


def is_business_day(date: datetime) -> bool:
    """
    Check if a date is a business day.
    
    Args:
        date: Date to check
    
    Returns:
        True if business day, False otherwise
    
    Example:
        >>> date = datetime(2024, 1, 1)  # Monday
        >>> is_business_day(date)
        True
        >>> date = datetime(2024, 1, 6)  # Saturday
        >>> is_business_day(date)
        False
    """
    # Check if weekday (Monday=0, Sunday=6)
    if date.weekday() >= 5:
        return False
    
    # Could add holiday checking here if needed
    return True


def align_to_business_day(
    date: datetime,
    direction: str = 'previous',
) -> datetime:
    """
    Align a date to the nearest business day.
    
    Args:
        date: Date to align
        direction: 'previous' or 'next' business day
    
    Returns:
        Aligned business day
    
    Example:
        >>> date = datetime(2024, 1, 6)  # Saturday
        >>> aligned = align_to_business_day(date, 'previous')
        >>> aligned  # datetime(2024, 1, 5) - Friday
    """
    if is_business_day(date):
        return date
    
    if direction == 'previous':
        offset = pd.offsets.BDay(-1)
    elif direction == 'next':
        offset = pd.offsets.BDay(1)
    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    result = pd.Timestamp(date) + offset
    return result.to_pydatetime()


def get_trading_days_between(
    start_date: datetime,
    end_date: datetime,
) -> int:
    """
    Count trading days between two dates.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Number of trading days
    
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 31)
        >>> count = get_trading_days_between(start, end)
        >>> count  # Approximately 22-23
    """
    business_days = get_business_days(start_date, end_date)
    return len(business_days)
