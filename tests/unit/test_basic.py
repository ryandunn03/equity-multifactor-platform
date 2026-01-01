"""
Basic tests to verify project structure and imports.
"""

import pytest
from pathlib import Path


def test_project_structure():
    """Verify expected directories exist."""
    expected_dirs = [
        'src/data',
        'src/factors',
        'src/portfolio',
        'src/backtest',
        'src/analytics',
        'config',
        'tests/unit',
    ]
    
    for dir_path in expected_dirs:
        assert Path(dir_path).exists(), f"Missing directory: {dir_path}"


def test_imports():
    """Verify core modules can be imported."""
    # These will work once you've created the files
    from src.utils import logging_config
    from src.utils import exceptions
    from src.utils import date_utils
    from src.data import cache
    from src.data import connectors
    
    assert hasattr(logging_config, 'setup_logger')
    assert hasattr(exceptions, 'FactorPlatformError')
    assert hasattr(cache, 'CacheManager')


def test_config_files_exist():
    """Verify configuration files are present."""
    assert Path('config/strategy_config.yaml').exists()
    assert Path('config/data_config.yaml').exists()
    assert Path('pyproject.toml').exists()
