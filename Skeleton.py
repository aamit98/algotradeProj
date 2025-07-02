import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2020,1,1))
    with col2:
        end_date = st.date_input("End", datetime.today())

    st.markdown("---")
    min_n,max_n = st.slider("Tickers in mix", 2, len(tickers), (3, min(6,len(tickers))))
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

run = optimize or compare
# ── MAIN ───────────────────────────────────────────────
if run:
    if len(tickers)<2:
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
    counts=rng.integers(min_n,max_n+1,n_port)
    w_int=np.zeros((n_port,m),int)
    risk_limit = RISK_LIMITS.get(risk_level, 1.0)
    
    for i,k in enumerate(counts):
        idx=rng.choice(m,k,replace=False)
        if whole:
            splits=rng.dirichlet(np.ones(k))*100
            ints=np.round(splits).astype(int); diff=100-ints.sum(); ints[rng.choice(k)]+=diff
            w_int[i,idx]=ints
        else:
            w=rng.random(k); w=w/w.sum(); w_int[i,idx]=(w*10000).astype(int)
    
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
    growth_new=(rets@w_chosen).add(1).cumprod()
    curves=pd.DataFrame({'Recommended':growth_new})
    
    # Add S&P 500 baseline
    spy_prices = load_prices(['SPY'], str(start_date), str(end_date))
    spy_returns = spy_prices.pct_change().dropna()
    curves['S&P 500 (SPY)'] = spy_returns['SPY'].add(1).cumprod()
    
    if df_existing is not None and compare:
        ex_ret=load_prices(df_existing.ticker.tolist(),str(start_date),str(end_date)).pct_change().dropna()
        curves['Existing']=(ex_ret@df_existing.weight.values).add(1).cumprod()
    st.line_chart(curves)
    st.markdown("""
    **How to read the chart**  
    • **X-axis** – calendar date.  
    • **Y-axis** – how many dollar today for every 1 dollar invested at the start (1 =no gain).  
    """)
else:
    st.write("← Configure inputs, click **Optimize**.  Upload CSV + click **Compare** to overlay your mix.")
