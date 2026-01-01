"""Custom exception classes for the equity factor platform.

Provides a hierarchy of exceptions for precise error handling across
the application.
"""


class FactorPlatformError(Exception):
    """Base exception for all platform-specific errors."""
    pass


class DataError(FactorPlatformError):
    """Raised when data is invalid, malformed, or inconsistent."""
    pass


class DataNotFoundError(DataError):
    """Raised when required data is not found."""
    pass


class DataQualityError(DataError):
    """Raised when data fails quality checks."""
    pass


class APIError(DataError):
    """Raised when API calls fail."""
    pass


class ConfigurationError(FactorPlatformError):
    """Raised when configuration is invalid or missing."""
    pass


class FactorError(FactorPlatformError):
    """Raised when factor computation fails."""
    pass


class PortfolioError(FactorPlatformError):
    """Raised when portfolio construction fails."""
    pass


class ConstraintViolationError(PortfolioError):
    """Raised when portfolio constraints are violated."""
    pass


class OptimizationError(PortfolioError):
    """Raised when portfolio optimization fails."""
    pass


class BacktestError(FactorPlatformError):
    """Raised when backtesting fails."""
    pass


class InsufficientDataError(BacktestError):
    """Raised when insufficient historical data is available."""
    pass


class ReportingError(FactorPlatformError):
    """Raised when report generation fails."""
    pass
