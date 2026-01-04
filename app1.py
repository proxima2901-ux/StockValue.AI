import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="StockPro: AI Analysis", page_icon="📈")

# Custom CSS for UI styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stMetric {
        background-color: #0e1117;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED DATA FUNCTIONS (Fixes Serialization Error)
# ==========================================

@st.cache_data(ttl=86400) # Cache ticker list for 24 hours
def load_tickers():
    """Loads a list of S&P 500 tickers for the smart dropdown"""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        df = pd.read_csv(url)
        # Create a search label: "AAPL - Apple Inc."
        df['Search_Label'] = df['Symbol'] + " - " + df['Security']
        return df
    except:
        # Fallback if URL fails
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'], 
            'Search_Label': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Google', 'TSLA - Tesla', 'NVDA - Nvidia']
        })

@st.cache_data(ttl=300)
def get_stock_history(ticker, period, interval):
    """Fetches ONLY the history dataframe (Serializable)"""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, interval=interval)
    return history

@st.cache_data(ttl=300)
def get_stock_info(ticker):
    """Fetches ONLY the info dictionary (Serializable)"""
    stock = yf.Ticker(ticker)
    return stock.info

@st.cache_data(ttl=300)
def get_stock_financials(ticker):
    """Fetches ONLY the financials dataframe (Serializable)"""
    stock = yf.Ticker(ticker)
    return stock.financials

@st.cache_data(ttl=900) # Cache news for 15 min
def get_stock_news(ticker):
    """Fetches ONLY the news list (Serializable)"""
    stock = yf.Ticker(ticker)
    return stock.news

# ==========================================
# 3. VALUATION & HELPER FUNCTIONS
# ==========================================

def format_number(num):
    if num is None: return "N/A"
    if num >= 1_000_000_000_000: return f"{num/1_000_000_000_000:.2f}T"
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.2f}B"
    if num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    return f"{num:,.2f}"

def calculate_graham_number(eps, bvps):
    """Graham Number = Sqrt(22.5 * EPS * BVPS)"""
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return 0
    return np.sqrt(22.5 * eps * bvps)

def calculate_dcf(fcf, shares, growth_rate=0.08, discount_rate=0.10):
    """Simplified DCF Model"""
    if fcf is None or shares is None or shares == 0:
        return 0
    
    terminal_growth = 0.025
    future_fcf = []
    
    # Project 5 years
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

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Smart Search")
    
    # Load Smart Ticker List
    ticker_df = load_tickers()
    
    # Search Mode Toggle
    search_mode = st.radio("Search Method", ["List (S&P 500)", "Manual Entry (Global)"])
    
    if search_mode == "List (S&P 500)":
        selected_label = st.selectbox("Select Company", ticker_df['Search_Label'])
        ticker_input = selected_label.split(" - ")[0]
    else:
        ticker_input = st.text_input("Enter Ticker (e.g., RELIANCE.NS)", value="AAPL").upper()
        
    st.divider()
    st.subheader("⚙️ Settings")
    timeframe = st.selectbox("Timeframe", ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y', 'MAX'], index=5)
    
    # Map timeframe to yfinance period/interval
    tf_map = {
        '1D': ('1d', '15m'), '1W': ('5d', '1h'), 
        '1M': ('1mo', '1d'), '3M': ('3mo', '1d'),
        '6M': ('6mo', '1d'), '1Y': ('1y', '1d'), 
        '5Y': ('5y', '1wk'), 'MAX': ('max', '1mo')
    }
    period, interval = tf_map[timeframe]

# --- MAIN PAGE ---
if ticker_input:
    # Fetch Data using CACHED functions
    history = get_stock_history(ticker_input, period, interval)
    info = get_stock_info(ticker_input)
    
    if history.empty:
        st.error(f"❌ Could not load data for **{ticker_input}**. Please check the ticker symbol.")
    else:
        current_price = history['Close'].iloc[-1]
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "📈 Pro Charts", "💰 Valuation", "📑 Financials", "🤖 AI News"
        ])
        
        # === TAB 1: DASHBOARD ===
        with tab1:
            st.title(f"{info.get('longName', ticker_input)} ({ticker_input})")
            
            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Price", f"${current_price:,.2f}", f"{((current_price - history['Open'].iloc[-1])/history['Open'].iloc[-1]*100):.2f}%")
            with m2: st.metric("Market Cap", format_number(info.get('marketCap')))
            with m3: st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            with m4: st.metric("Sector", info.get('sector', 'N/A'))
            
            st.divider()
            
            # Quick Valuation Check
            st.subheader("⚡ Quick Verdict")
            graham = calculate_graham_number(info.get('trailingEps'), info.get('bookValue'))
            
            # Use Free Cash Flow or Operating Cash Flow for DCF
            fcf = info.get('freeCashflow')
            if fcf is None:
                fcf = info.get('operatingCashflow', 0)
            dcf = calculate_dcf(fcf, info.get('sharesOutstanding'))
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.info(f"**Graham Number:** ${graham:.2f}")
                if graham > current_price:
                    st.success("Undervalued (Graham)")
                else:
                    st.error("Overvalued (Graham)")
            with v_col2:
                st.info(f"**DCF Value:** ${dcf:.2f}")
                if dcf > current_price:
                    st.success("Undervalued (DCF)")
                else:
                    st.error("Overvalued (DCF)")

        # === TAB 2: PRO CHARTS ===
        with tab2:
            st.subheader(f"Price Action ({timeframe})")
            
            # Create Subplots: Row 1 = Price, Row 2 = Volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # Candlestick
            fig.add_trace(go.Candlestick(x=history.index,
                                         open=history['Open'], high=history['High'],
                                         low=history['Low'], close=history['Close'], name='OHLC'), 
                                         row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='Volume', marker_color='teal'), row=2, col=1)

            # Update Layout
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # === TAB 3: VALUATION ===
        with tab3:
            st.header("🧮 Valuation Details")
            st.write("This section breaks down the math behind the numbers.")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Graham Number Logic")
                st.latex(r"\sqrt{22.5 \times EPS \times BVPS}")
                st.write(f"Trailing EPS: {info.get('trailingEps', 0)}")
                st.write(f"Book Value/Share: {info.get('bookValue', 0)}")
            
            with c2:
                st.subheader("DCF Model Logic")
                st.write("We project Free Cash Flow (FCF) for 5 years at 8% growth and discount it back to today.")
                st.write(f"FCF Used: {format_number(fcf)}")

        # === TAB 4: FINANCIALS ===
        with tab4:
            st.subheader("Annual Financials")
            fin_df = get_stock_financials(ticker_input)
            if not fin_df.empty:
                st.dataframe(fin_df)
                
                # Simple chart for Revenue
                if 'Total Revenue' in fin_df.index:
                    rev_data = fin_df.loc['Total Revenue']
                    st.bar_chart(rev_data)
            else:
                st.warning("Financial data not available for this ticker.")

        # === TAB 5: NEWS ANALYSIS ===
        with tab5:
            st.subheader("🤖 News Sentiment Analysis")
            news_list = get_stock_news(ticker_input)
            
            if news_list:
                total_polarity = 0
                count = 0
                
                for item in news_list[:5]: # Analyze top 5 news
                    title = item['title']
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    total_polarity += polarity
                    count += 1
                    
                    # Display News Item
                    with st.expander(f"{title} (Score: {polarity:.2f})"):
                        st.write(f"Source: {item['publisher']}")
                        st.write(f"[Read Article]({item['link']})")
                
                if count > 0:
                    avg_sentiment = total_polarity / count
                    st.divider()
                    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
                    if avg_sentiment > 0.1:
                        st.success("Overall Sentiment: BULLISH 🐂")
                    elif avg_sentiment < -0.1:
                        st.error("Overall Sentiment: BEARISH 🐻")
                    else:
                        st.info("Overall Sentiment: NEUTRAL 😐")
            else:
                st.info("No recent news found.")
