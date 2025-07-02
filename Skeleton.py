import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="💹 Portfolio Studio", layout="wide")

# Custom CSS to make sidebar wider
st.markdown("""
<style>
    .css-1d391kg {
        width: 350px;
    }
    .css-1lcbmhc {
        width: 350px;
    }
    .css-17eq0hr {
        width: 350px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=21600)
def load_prices(tickers, start, end):
    return yf.download(tickers, start=start, end=end, progress=False)["Close"].ffill().dropna()

@st.cache_data(ttl=86400)
def ticker_name(t):
    try:
        return yf.Ticker(t).info.get("shortName", t)
    except Exception:
        return t

CATEGORIES = {
    "US Market": ["SPY","DIA","QQQ","IWM","MDY"],
    "Sectors": ["XLK","XLF","XLY","XLI","XLE","XLP","XLV","XLU","XLRE"],
    "Mega Caps": ["AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","BRK-B"],
    "Fixed Income": ["BND","AGG","LQD","IEF","TLT","TIP"],
    "Commodities": ["GLD","SLV","USO","DBO","UNG","DBB"],
    "Crypto": ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"],
    "International": ["VGK","EWJ","EWZ","EEM","VWO","INDA"],
}
BUCKETS = ["Very Conservative","Conservative","Balanced","Aggressive","Very Aggressive"]

# Risk classification for tickers
RISK_LEVELS = {
    "low": ["SPY","DIA","QQQ","IWM","MDY","BND","AGG","LQD","IEF","TLT","TIP"],
    "medium": ["XLK","XLF","XLY","XLI","XLE","XLP","XLV","XLU","XLRE","VGK","EWJ","EWZ","EEM","VWO","INDA"],
    "high": ["AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","BRK-B","GLD","SLV","USO","DBO","UNG","DBB","BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
}

# Maximum allocation to high-risk assets by risk profile
RISK_LIMITS = {
    "Very Conservative": 0.15,  # Max 15% in high-risk assets
    "Conservative": 0.30,       # Max 30% in high-risk assets  
    "Balanced": 0.50,          # Max 50% in high-risk assets
    "Aggressive": 0.75,        # Max 75% in high-risk assets
    "Very Aggressive": 1.0     # No limits
}

# Robust ARIMA forecasting function with fallbacks
def create_arima_forecast(returns_series, periods=90, name=""):
    """Create ARIMA forecast with robust error handling"""
    try:
        # Clean the data first
        clean_series = returns_series.dropna()
        
        # Need minimum data points
        if len(clean_series) < 20:
            return create_simple_fallback(clean_series, periods)
        
        # Try ARIMA with different orders - start simple
        arima_orders = [(1,1,1), (1,0,1), (2,1,1), (1,1,0), (0,1,1)]
        
        for order in arima_orders:
            try:
                # Fit ARIMA model
                model = ARIMA(clean_series, order=order)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=periods)
                
                # Add realistic volatility
                volatility = clean_series.std()
                np.random.seed(42)
                noise = np.random.normal(0, volatility * 0.3, len(forecast))
                forecast_with_noise = forecast + noise
                
                return pd.Series(forecast_with_noise)
                
            except Exception:
                # Try next ARIMA order
                continue
        
        # If all ARIMA orders fail, use simple fallback
        return create_simple_fallback(clean_series, periods)
        
    except Exception:
        # Ultimate fallback
        return create_simple_fallback(returns_series, periods)

def create_simple_fallback(returns_series, periods):
    """Simple fallback when ARIMA fails"""
    try:
        clean_series = returns_series.dropna()
        mean_return = clean_series.mean() if len(clean_series) > 0 else 0.001
        volatility = clean_series.std() if len(clean_series) > 1 else 0.01
        
        # Generate simple forecast
        np.random.seed(42)
        forecast_returns = []
        for i in range(periods):
            noise = np.random.normal(0, volatility * 0.2)
            forecast_returns.append(mean_return + noise)
        
        return pd.Series(forecast_returns)
        
    except Exception:
        # Ultimate fallback - small positive returns
        return pd.Series([0.0005] * periods)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Profile")
    # Create dropdown options with risk limits
    bucket_options = [
        "Very Conservative (≤15% high-risk)",
        "Conservative (≤30% high-risk)", 
        "Balanced (≤50% high-risk)",
        "Aggressive (≤75% high-risk)",
        "Very Aggressive (no limits)",
        "I don't know"
    ]
    bucket_select = st.selectbox("Risk appetite", bucket_options, index=2)
    if bucket_select == "I don't know":
        with st.expander("Quick risk quiz"):
            a1 = st.selectbox("If portfolio drops 10 %…", ["Sell all","Sell some","Hold","Buy more"])
            a2 = st.selectbox("Goal", ["Preserve capital","Income","Moderate growth","Max growth"])
            a3 = st.selectbox("Horizon", ["<1 yr","1‑3 yrs","3‑7 yrs",">7 yrs"])
            score = sum([
                ["Sell all","Sell some","Hold","Buy more"].index(a1),
                ["Preserve capital","Income","Moderate growth","Max growth"].index(a2),
                ["<1 yr","1‑3 yrs","3‑7 yrs",">7 yrs"].index(a3),
            ])
            risk_level = BUCKETS[min(int(score/2),4)]
            st.info(f"Diagnosed: {risk_level}")
    else:
        # Map dropdown selection back to original risk level names
        risk_mapping = {
            "Very Conservative (≤15% high-risk)": "Very Conservative",
            "Conservative (≤30% high-risk)": "Conservative", 
            "Balanced (≤50% high-risk)": "Balanced",
            "Aggressive (≤75% high-risk)": "Aggressive",
            "Very Aggressive (no limits)": "Very Aggressive"
        }
        risk_level = risk_mapping.get(bucket_select, "Balanced")

    st.markdown("---")
    cats = st.multiselect("Categories", list(CATEGORIES.keys()), default=["US Market"])
    universe = sorted({t for c in cats for t in CATEGORIES[c]})
    tickers = st.multiselect("Tickers", universe, default=universe[:5])
    
    # Store in session state for main area access
    st.session_state.selected_categories = cats
    st.session_state.selected_tickers = tickers
    
    if tickers:
        if st.button("📖 Learn about selected tickers"):
            st.session_state.show_ticker_info = True

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2020,1,1))
    with col2:
        end_date = st.date_input("End", datetime.today())

    st.markdown("---")
    whole = st.checkbox("Whole‑percent weights", True)
    n_port = st.slider("Simulations", 5000, 300000, 40000, 5000)

    st.markdown("---")
    optimize = st.button("🔧 Optimize")
    
    st.markdown("---")
    up_file = st.file_uploader("Upload CSV (ticker,weight%)")
    st.caption("""
    **CSV Format Expected:**
    ```
    ticker,weight
    SPY,47.30
    QQQ,0.67
    IEI,23.89
    ```
    • Column 1: `ticker` (stock/ETF symbols)
    • Column 2: `weight` (percentage, e.g., 47.30 for 47.30%)
    """)
    df_existing=None
    if up_file:
        try:
            df_existing = pd.read_csv(up_file)
            df_existing.weight = df_existing.weight/100
            st.success("CSV loaded")
        except Exception as e:
            st.error(f"CSV error: {e}")

    compare = st.button("📊 Compare")

# ── MAIN ───────────────────────────────────────────────

# Display ticker information if requested
if st.session_state.get('show_ticker_info', False):
    st.header("📖 Ticker Information")
    
    # Get tickers from session state
    tickers = st.session_state.get('selected_tickers', [])
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            st.subheader(f"{ticker} - {info.get('shortName', ticker)}")
            
            if info.get('marketCap'):
                st.write(f"**Market Cap:** ${info.get('marketCap', 0):,.0f}")
            if info.get('currency'):
                st.write(f"**Currency:** {info.get('currency')}")
            if info.get('exchange'):
                st.write(f"**Exchange:** {info.get('exchange')}")
            
            if info.get('longBusinessSummary'):
                st.write("**Business Summary:**")
                st.write(info['longBusinessSummary'][:500] + "..." if len(info['longBusinessSummary']) > 500 else info['longBusinessSummary'])
            
            st.markdown("---")
        
        except Exception as e:
            st.warning(f"Could not load information for {ticker}: {str(e)}")
    
    if st.button("❌ Close ticker information"):
        st.session_state.show_ticker_info = False
        st.rerun()
    
    st.markdown("---")

run = optimize or compare
if run:
    if len(tickers) < 2:
        st.error("Select at least two tickers")
        st.stop()
    
    if compare and df_existing is None:
        st.error("📊 Please upload a CSV file first to compare portfolios")
        st.stop()

    prices = load_prices(tickers, str(start_date), str(end_date))
    rets = prices.pct_change().dropna()
    mu, cov = rets.mean()*252, rets.cov()*252
    m=len(tickers)

    def apply_risk_constraints(weights_row, tickers, risk_limit):
        """Apply risk constraints to a single portfolio's weights"""
        if risk_limit >= 1.0:  # No constraints for Very Aggressive
            return weights_row
            
        # Identify high-risk positions
        high_risk_mask = np.array([t in RISK_LEVELS["high"] for t in tickers])
        high_risk_allocation = weights_row[high_risk_mask].sum()
        
        if high_risk_allocation <= risk_limit:
            return weights_row  # Already within limits
            
        # Scale down high-risk allocations
        scale_factor = risk_limit / high_risk_allocation
        weights_row[high_risk_mask] *= scale_factor
        
        # Redistribute excess to low/medium risk assets
        excess = 1.0 - weights_row.sum()
        low_med_mask = ~high_risk_mask & (weights_row > 0)
        if low_med_mask.any():
            weights_row[low_med_mask] += excess * weights_row[low_med_mask] / weights_row[low_med_mask].sum()
        
        return weights_row

    rng=np.random.default_rng(0)
    w_int=np.zeros((n_port,m),int)
    risk_limit = RISK_LIMITS.get(risk_level, 1.0)
    
    # Use all selected tickers for each portfolio simulation
    for i in range(n_port):
        if whole:
            splits=rng.dirichlet(np.ones(m))*100
            ints=np.round(splits).astype(int); diff=100-ints.sum(); ints[rng.choice(m)]+=diff
            w_int[i,:]=ints
        else:
            w=rng.random(m); w=w/w.sum(); w_int[i,:]=(w*10000).astype(int)
    
    weights=w_int/100 if whole else w_int/10000
    
    # Apply risk constraints to each portfolio
    for i in range(n_port):
        weights[i] = apply_risk_constraints(weights[i], tickers, risk_limit)

    port_r=weights@mu.values
    port_v=np.sqrt(np.einsum('ij,jk,ik->i',weights,cov,weights))
    sharpe=port_r/port_v
    df_mc=pd.DataFrame({'Return':port_r,'Vol':port_v,'Sharpe':sharpe})
    for j,t in enumerate(tickers):
        df_mc[t]=weights[:,j]

    pct_map={BUCKETS[0]:.05,BUCKETS[1]:.25,BUCKETS[2]:.5,BUCKETS[3]:.75,BUCKETS[4]:None}
    pick=sharpe.argmax() if pct_map[risk_level] is None else df_mc.sort_values('Vol').iloc[int(pct_map[risk_level]*(len(df_mc)-1))].name
    chosen=df_mc.loc[pick]
    w_chosen=chosen[tickers]

    # Efficient frontier
    st.subheader("Efficient frontier")
    df_mc['Return%']=df_mc.Return*100; df_mc['Vol%']=df_mc.Vol*100
    fig,ax=plt.subplots(figsize=(8,4))
    ax.scatter(df_mc['Vol%'],df_mc['Return%'],c=df_mc['Sharpe'],cmap='plasma',s=4,alpha=.4)
    ax.scatter(chosen.Vol*100, chosen.Return*100, c='red', s=200, label='Recommended')
    if df_existing is not None and compare:
        ex_tic=df_existing.ticker.tolist(); ex_w=df_existing.weight.values
        ex_prices=load_prices(ex_tic,str(start_date),str(end_date)); ex_ret=ex_prices.pct_change().dropna()
        ex_mu=ex_ret.mean()*252; ex_cov=ex_ret.cov()*252
        ex_r=np.dot(ex_w,ex_mu); ex_v=np.sqrt(ex_w@ex_cov@ex_w)
        ax.scatter(ex_v*100, ex_r*100, marker='D', c='orange', s=160, label='Uploaded')
    ax.set_xlabel('Volatility (%)'); ax.set_ylabel('Expected Return (%)'); ax.legend()
    st.pyplot(fig, use_container_width=False)

    # Allocation + metrics
    st.subheader("Allocation & Metrics")
    alloc=pd.DataFrame({
        'Ticker':tickers,
        'Company':[ticker_name(t) for t in tickers],
        'Weight %':(w_chosen*100).round(1)
    })
    st.dataframe(alloc)
    metrics=pd.Series({
        'Expected Return %':round(chosen.Return*100,2),
        'Expected Vol %':round(chosen.Vol*100,2),
        'Sharpe':round(chosen.Sharpe,2)
    })
    st.write(metrics.to_frame().T)

    # Historical growth
    st.subheader("Historical growth")
    
    # Fixed forecast period - no user selection to avoid crashes
    st.write("**Forecast:** ARIMA time series prediction (90-day projection)")
    
    portfolio_returns = rets@w_chosen
    growth_new = portfolio_returns.add(1).cumprod()
    curves=pd.DataFrame({'Recommended':growth_new})
    
    # Add S&P 500 baseline
    spy_prices = load_prices(['SPY'], str(start_date), str(end_date))
    spy_returns = spy_prices.pct_change().dropna()
    curves['S&P 500 (SPY)'] = spy_returns['SPY'].add(1).cumprod()
    
    if df_existing is not None and compare:
        ex_ret=load_prices(df_existing.ticker.tolist(),str(start_date),str(end_date)).pct_change().dropna()
        existing_returns = ex_ret@df_existing.weight.values
        curves['Existing'] = existing_returns.add(1).cumprod()
    
    # Generate forecasts (daily frequency) - FIXED PERIOD
    forecast_periods = 90  # Fixed 90-day forecast
    last_date = curves.index[-1]
    last_value_recommended = curves['Recommended'].iloc[-1]
    last_value_spy = curves['S&P 500 (SPY)'].iloc[-1]
    
    # Create daily forecast dates
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=forecast_periods, freq='D')
    
    # Select forecast function based on user choice
    forecast_func = create_arima_forecast
    model_label = "ARIMA Forecast"
    
    try:
        # Show progress for complex models and run forecasting
        # Recommended portfolio forecast
        portfolio_forecast = forecast_func(portfolio_returns, forecast_periods, "Recommended")
        
        # SPY forecast
        spy_forecast = forecast_func(spy_returns['SPY'], forecast_periods, "S&P 500")
        
        # Add existing portfolio forecast if comparing
        if df_existing is not None and compare:
            existing_forecast = forecast_func(existing_returns, forecast_periods, "Existing")
        else:
            existing_forecast = None
        
        # Convert to growth values starting from last historical value
        portfolio_forecast_growth = []
        current_value = last_value_recommended
        for ret in portfolio_forecast:
            current_value = current_value * (1 + ret)
            portfolio_forecast_growth.append(current_value)
        
        # Convert SPY forecast to growth values
        spy_forecast_growth = []
        current_value = last_value_spy
        for ret in spy_forecast:
            current_value = current_value * (1 + ret)
            spy_forecast_growth.append(current_value)
        
        # Create forecast dataframe with dynamic labeling
        forecast_data = {
            f'Recommended ({model_label} Forecast)': portfolio_forecast_growth,
            f'S&P 500 ({model_label} Forecast)': spy_forecast_growth
        }
        
        # Add existing portfolio forecast if available
        if existing_forecast is not None:
            last_value_existing = curves['Existing'].iloc[-1]
            existing_forecast_growth = []
            current_value = last_value_existing
            for ret in existing_forecast:
                current_value = current_value * (1 + ret)
                existing_forecast_growth.append(current_value)
            forecast_data[f'Existing ({model_label} Forecast)'] = existing_forecast_growth
        
        forecast_df = pd.DataFrame(forecast_data, index=forecast_dates)
        
        # Combine historical and forecast data
        combined_curves = pd.concat([curves, forecast_df])
        st.line_chart(combined_curves)
        
    except Exception as e:
        st.error(f"{model_label} forecasting failed: {str(e)}")
        st.info("Showing historical data only.")
        st.line_chart(curves)
        
    # Dynamic chart explanation based on selected model
    forecast_description = "ARIMA time series predictions using autoregressive integrated moving average modeling with multiple fallbacks."

    st.markdown(f"""
    **How to read the chart**  
    • **X-axis** – calendar date (historical data + {model_label.lower()} forecast).  
    • **Y-axis** – how many dollar today for every 1 dollar invested at the start (1 = no gain).  
    • **Historical lines** – Actual performance based on market data.
    • **Forecast lines** – {forecast_description}
    """)
else:
    st.write("← Configure inputs, click **Optimize**.  Upload CSV + click **Compare** to overlay your mix.")
