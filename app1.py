import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="StockPro: AI Analysis Platform", page_icon="📈")

# Custom CSS for a professional look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .stMetric {
        background-color: #0e1117;
        padding: 10px;
        border-radius: 5px;
    }
    .undervalued { color: #00ff00; font-weight: bold; }
    .overvalued { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_stock_data(ticker_symbol, period, interval):
    """Fetches historical data with error handling"""
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period=period, interval=interval)
        return stock, history
    except Exception as e:
        return None, None

def format_number(num):
    """Formats large numbers into K, M, B, T"""
    if num is None: return "N/A"
    if num >= 1_000_000_000_000: return f"{num/1_000_000_000_000:.2f}T"
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.2f}B"
    if num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    return f"{num:,.2f}"

# ==========================================
# 3. VALUATION MODELS
# ==========================================

def calculate_graham_number(info):
    """Graham Number = Sqrt(22.5 * EPS * BVPS)"""
    try:
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        if eps > 0 and bvps > 0:
            graham_num = np.sqrt(22.5 * eps * bvps)
            return graham_num
        return 0
    except:
        return 0

def calculate_dcf(info):
    """Simplified DCF Model"""
    try:
        fcf = info.get('freeCashflow', 0)
        if not fcf: # Fallback if FCF is missing
            fcf = info.get('operatingCashflow', 0) - info.get('capitalExpenditures', 0)
            
        shares = info.get('sharesOutstanding', 1)
        growth_rate = 0.08  # Assumption: 8% growth
        discount_rate = 0.10 # Assumption: 10% discount
        terminal_growth = 0.025
        
        # Project 5 years
        future_fcf = []
        for i in range(1, 6):
            future_fcf.append(fcf * ((1 + growth_rate) ** i))
            
        # Terminal Value
        terminal_val = (future_fcf[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        # Discount back to present
        dcf_value = 0
        for i, val in enumerate(future_fcf):
            dcf_value += val / ((1 + discount_rate) ** (i + 1))
            
        dcf_value += terminal_val / ((1 + discount_rate) ** 5)
        
        return dcf_value / shares
    except:
        return 0

def get_sector_benchmarks(sector):
    """Returns industry average approximations for comparison"""
    benchmarks = {
        'Technology': {'pe': 25, 'pb': 6, 'description': 'High Growth'},
        'Financial Services': {'pe': 12, 'pb': 1.2, 'description': 'Value/Dividends'},
        'Healthcare': {'pe': 20, 'pb': 4, 'description': 'Defensive'},
        'Energy': {'pe': 10, 'pb': 1.5, 'description': 'Cyclical'},
        'Consumer Cyclical': {'pe': 18, 'pb': 3, 'description': 'Cyclical'},
        'Industrials': {'pe': 18, 'pb': 2.5, 'description': 'Stable'},
    }
    return benchmarks.get(sector, {'pe': 20, 'pb': 3, 'description': 'General'})

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

# Sidebar
with st.sidebar:
    st.header("🔍 Stock Search")
    ticker_input = st.text_input("Enter Ticker", value="AAPL").upper()
    
    st.subheader("⚙️ Chart Settings")
    timeframe = st.selectbox("Timeframe", ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y', 'MAX'])
    
    # Map timeframe to yfinance interval
    tf_map = {
        '1D': ('1d', '15m'), '1W': ('5d', '1h'), 
        '1M': ('1mo', '1d'), '3M': ('3mo', '1d'),
        '6M': ('6mo', '1d'), '1Y': ('1y', '1d'), 
        '5Y': ('5y', '1wk'), 'MAX': ('max', '1mo')
    }
    period, interval = tf_map[timeframe]

if ticker_input:
    stock, history = get_stock_data(ticker_input, period, interval)
    
    if stock is None or history.empty:
        st.error(f"Could not find data for {ticker_input}. Please check the symbol.")
    else:
        info = stock.info
        current_price = history['Close'].iloc[-1]
        
        # --- TAB STRUCTURE ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "📈 Pro Charts", "💰 Valuation (The Logic)", "📑 Financials", "🤖 AI Insights"
        ])
        
        # === TAB 1: DASHBOARD ===
        with tab1:
            st.title(f"{info.get('longName', ticker_input)}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", 
                          f"{((current_price - history['Open'].iloc[-1])/history['Open'].iloc[-1]*100):.2f}%")
            with col2:
                st.metric("Market Cap", format_number(info.get('marketCap')))
            with col3:
                st.metric("Sector", info.get('sector', 'N/A'))
            with col4:
                st.metric("Beta", f"{info.get('beta', 'N/A')}")
            
            st.markdown("---")
            
            # Valuation Quick View
            st.subheader("⚖️ Valuation Snapshot")
            
            sector = info.get('sector', 'General')
            graham = calculate_graham_number(info)
            dcf = calculate_dcf(info)
            
            # Logic for Verdict
            intrinsic_value = 0
            method_used = ""
            
            if sector == 'Financial Services':
                # Banks don't work well with DCF/Graham usually, use P/B or Dividend logic
                # For this free tool, we stick to Graham but modified or P/B context
                intrinsic_value = graham 
                method_used = "Graham Number (Financials)"
            elif sector == 'Technology':
                intrinsic_value = dcf 
                method_used = "DCF (Growth Focus)"
            else: # Manufacturing / General
                intrinsic_value = graham 
                method_used = "Graham Number (Manufacturing/Stable)"
                
            # Verdict Logic
            if intrinsic_value > 0:
                diff_pct = ((intrinsic_value - current_price) / current_price) * 100
                verdict_color = "green" if diff_pct > 0 else "red"
                verdict_text = "UNDERVALUED" if diff_pct > 0 else "OVERVALUED"
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown(f"### Intrinsic Value: **${intrinsic_value:.2f}**")
                    st.caption(f"Based on {method_used}")
                with v_col2:
                    st.markdown(f"### The Stock is <span style='color:{verdict_color}'>{verdict_text}</span> by **{abs(diff_pct):.2f}%**", unsafe_allow_html=True)
                    st.progress(min(max((diff_pct + 100) / 200, 0.0), 1.0)) # Visual bar
            else:
                st.warning("Insufficient data to calculate specific intrinsic value.")

            # Industry Comparison
            st.subheader("🏢 Industry Comparison")
            benchmarks = get_sector_benchmarks(sector)
            p_pe = info.get('trailingPE', 0)
            p_pb = info.get('priceToBook', 0)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**P/E Ratio**")
                st.write(f"This Stock: **{p_pe:.2f}**")
                st.write(f"Industry Avg: **{benchmarks['pe']}**")
                if p_pe < benchmarks['pe']: st.success("Better than Industry Avg")
                else: st.error("Expensive vs Industry")
                
            with c2:
                st.info(f"**P/B Ratio**")
                st.write(f"This Stock: **{p_pb:.2f}**")
                st.write(f"Industry Avg: **{benchmarks['pb']}**")
                if p_pb < benchmarks['pb']: st.success("Better than Industry Avg")
                else: st.error("Expensive vs Industry")

        # === TAB 2: PRO CHARTS ===
        with tab2:
            st.subheader(f"Price Action ({timeframe})")
            
            # Candlestick + Volume Subplot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f'{ticker_input} Price', 'Volume'), 
                                row_width=[0.2, 0.7])

            # Candlestick
            fig.add_trace(go.Candlestick(x=history.index,
                                         open=history['Open'], high=history['High'],
                                         low=history['Low'], close=history['Close'], name='OHLC'), 
                                         row=1, col=1)

            # Moving Averages (Only for sufficient data)
            if len(history) > 50:
                history['SMA50'] = history['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA50'], mode='lines', name='SMA 50', line=dict(color='orange')), row=1, col=1)
            
            if len(history) > 200:
                history['SMA200'] = history['Close'].rolling(window=200).mean()
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA200'], mode='lines', name='SMA 200', line=dict(color='blue')), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='Volume', marker_color='teal'), row=2, col=1)

            fig.update_layout(height=600, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # === TAB 3: VALUATION DETAILS ===
        with tab3:
            st.header("🧮 Valuation Methodology")
            st.markdown(f"**Selected Sector:** {sector}")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("1. Graham Number")
                st.write("Best for: **Manufacturing, Industrials, Stable Companies**")
                st.latex(r"\sqrt{22.5 \times EPS \times BVPS}")
                st.write(f"EPS: {info.get('trailingEps')}")
                st.write(f"Book Value/Share: {info.get('bookValue')}")
                st.markdown(f"**Result: ${graham:.2f}**")
                
            with c2:
                st.subheader("2. Discounted Cash Flow (DCF)")
                st.write("Best for: **Tech, Growth, Predictable Cash Flow**")
                st.write("Assumptions: 10% Discount Rate, 3% Terminal Growth")
                st.markdown(f"**Result: ${dcf:.2f}**")

        # === TAB 4: FINANCIALS ===
        with tab4:
            st.subheader("Income Statement (Annual)")
            fin = stock.financials
            if not fin.empty:
                st.dataframe(fin.loc[['Total Revenue', 'Net Income', 'Operating Income']])
                
                # Plot Revenue
                rev_data = fin.loc['Total Revenue'].sort_index()
                st.bar_chart(rev_data)
            else:
                st.write("Financial data not available.")

        # === TAB 5: AI ANALYSIS ===
        with tab5:
            st.header("🤖 AI Sentiment & News")
            
            try:
                # Fetch news
                news = stock.news
                if news:
                    sentiment_score = 0
                    count = 0
                    
                    for item in news[:5]:
                        title = item['title']
                        blob = TextBlob(title)
                        polarity = blob.sentiment.polarity
                        sentiment_score += polarity
                        count += 1
                        
                        # Display news item
                        with st.expander(f"{title} ({'🟢 Bullish' if polarity > 0 else '🔴 Bearish' if polarity < 0 else '⚪ Neutral'})"):
                            st.write(f"Source: {item['publisher']}")
                            st.write(f"Link: [Read More]({item['link']})")
                    
                    avg_score = sentiment_score / count if count > 0 else 0
                    
                    st.divider()
                    st.subheader("Aggregated Sentiment Score")
                    st.markdown(f"### Score: {avg_score:.2f} / 1.0")
                    
                    if avg_score > 0.1:
                        st.success("Overall Market Sentiment is POSITIVE")
                    elif avg_score < -0.1:
                        st.error("Overall Market Sentiment is NEGATIVE")
                    else:
                        st.info("Overall Market Sentiment is NEUTRAL")
                        
                else:
                    st.write("No news found for this ticker.")
            except Exception as e:
                st.write("Could not fetch latest news analysis.")