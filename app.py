import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize App with a dark theme (CYBORG)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # Needed for deployment

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Get 1 year history for default view
        hist = stock.history(period="1y")
        return stock, hist, stock.info
    except:
        return None, None, None

# ==========================================
# APP LAYOUT (The "Frontend")
# ==========================================
app.layout = dbc.Container([
    
    # --- Header ---
    dbc.Row([
        dbc.Col(html.H1("📈 StockPro Analytics", className="text-center text-primary mb-4"), width=12)
    ], className="mt-3"),

    # --- Inputs ---
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker-input", placeholder="Enter Ticker (e.g. AAPL)", type="text", value="AAPL"),
        ], width=8),
        dbc.Col([
            dbc.Button("Analyze", id="analyze-btn", color="primary", className="w-100"),
        ], width=4),
    ], className="mb-4"),

    # --- Loading Spinner & Content ---
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="main-content")
    ),

], fluid=True)

# ==========================================
# CALLBACKS (The "Backend" Logic)
# ==========================================
@app.callback(
    Output("main-content", "children"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    prevent_initial_call=True
)
def update_dashboard(n_clicks, ticker):
    if not ticker:
        return dbc.Alert("Please enter a ticker symbol", color="warning")
    
    stock, history, info = get_stock_data(ticker)
    
    if history.empty:
        return dbc.Alert(f"Could not find data for {ticker}", color="danger")

    current_price = history['Close'].iloc[-1]
    
    # --- 1. Valuation Logic ---
    pe_ratio = info.get('trailingPE', 0)
    eps = info.get('trailingEps', 0)
    bvps = info.get('bookValue', 0)
    
    # Graham Number
    graham = 0
    if eps and bvps and eps > 0 and bvps > 0:
        graham = np.sqrt(22.5 * eps * bvps)
    
    # Simple Valuation Badge
    if graham > current_price:
        verdict = "UNDERVALUED"
        color = "success"
    else:
        verdict = "OVERVALUED"
        color = "danger"

    # --- 2. Create Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'],
                                 low=history['Low'], close=history['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))

    # --- 3. Build the Tab Layout ---
    content = html.Div([
        # Top Metrics Cards
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"${current_price:.2f}", className="card-title"),
                    html.P("Current Price", className="card-text text-muted")
                ])
            ], color="dark", inverse=True), width=3),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{info.get('sector', 'N/A')}", className="card-title"),
                    html.P("Sector", className="card-text text-muted")
                ])
            ], color="dark", inverse=True), width=3),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(f"{verdict}", className=f"text-{color}"),
                    html.P(f"Graham Value: ${graham:.2f}", className="card-text text-muted")
                ])
            ], color="dark", inverse=True), width=6),
        ], className="mb-4"),

        # Tabs
        dbc.Tabs([
            # Tab 1: Charts
            dbc.Tab(dcc.Graph(figure=fig), label="Pro Charts"),
            
            # Tab 2: Financials
            dbc.Tab(dbc.Card(dbc.CardBody([
                html.H5("Company Summary", className="card-title"),
                html.P(info.get('longBusinessSummary', 'No summary available.'))
            ]), className="mt-3"), label="Profile"),
            
            # Tab 3: Valuation
            dbc.Tab(dbc.Card(dbc.CardBody([
                html.H4("Graham Number Calculation"),
                html.Hr(),
                html.P(f"EPS: {eps}"),
                html.P(f"Book Value: {bvps}"),
                html.P(f"Formula: Sqrt(22.5 * {eps} * {bvps})"),
                html.H3(f"= ${graham:.2f}")
            ]), className="mt-3"), label="Valuation Logic")
        ])
    ])
    
    return content

# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
