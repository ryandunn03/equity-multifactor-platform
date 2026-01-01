"""
CLI entry point for running backtests.

Example:
    python run_backtest.py --config config/strategy_config.yaml
"""

import argparse
from pathlib import Path

# TODO: Implement full backtest entry point
# This is a placeholder that will be completed during Week 4

def main():
    parser = argparse.ArgumentParser(
        description='Run multi-factor equity backtest'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/strategy_config.yaml',
        help='Path to strategy configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/backtests',
        help='Output directory for results'
    )
    parser.add_argument(
        '--generate-pdf',
        action='store_true',
        help='Generate PDF tear sheet'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Factor Equity Research Platform")
    print("=" * 60)
    print(f"\nConfiguration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Generate PDF: {args.generate_pdf}")
    print("\n⚠️  Backtest engine not yet implemented.")
    print("📚 See IMPLEMENTATION_GUIDE.md for build instructions.")
    print("\nTo start building:")
    print("  1. Implement factors (src/factors/*.py)")
    print("  2. Build portfolio construction (src/portfolio/*.py)")
    print("  3. Create backtest engine (src/backtest/*.py)")
    print("  4. Add analytics (src/analytics/*.py)")
    print("  5. Return here to complete this entry point")
    

if __name__ == '__main__':
    main()
