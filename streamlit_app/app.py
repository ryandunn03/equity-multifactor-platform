"""
Streamlit dashboard for interactive exploration of backtest results.

To run:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Multi-Factor Equity Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Multi-Factor Equity Research Platform")

st.markdown("""
## 🚧 Dashboard Under Construction

This interactive dashboard will provide:
- **Performance Overview**: Cumulative returns, risk metrics, drawdown analysis
- **Factor Analysis**: IC time series, factor contribution, single-factor backtests
- **Risk & Attribution**: Factor loadings, sector exposures, concentration metrics
- **Diagnostics**: Turnover analysis, transaction costs, data quality

### 📚 Implementation Status
- ✅ Project setup complete
- ✅ Data layer implemented
- ⬜ Factor library (Week 2)
- ⬜ Portfolio construction (Week 3)
- ⬜ Backtest engine (Week 4)
- ⬜ Analytics (Week 5)
- ⬜ Dashboard (Week 6)

See `IMPLEMENTATION_GUIDE.md` for detailed build instructions.
""")

st.info("💡 Run a backtest first using: `python run_backtest.py`")
