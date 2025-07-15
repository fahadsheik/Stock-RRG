import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os # Import os to handle file paths

# ---------------------- CONFIGURATION ---------------------- #
st.set_page_config(page_title="Indian Market RRG Tool", layout="wide")

# --- MODIFIED: Define colors for dark theme ---
# Colors for the markers themselves (brighter for dark background)
QUADRANT_COLORS = {
    "Leading": "rgba(0, 255, 0, 0.8)",      # Bright Green
    "Weakening": "rgba(255, 165, 0, 0.8)",  # Orange
    "Lagging": "rgba(255, 0, 0, 0.8)",      # Red
    "Improving": "rgba(0, 191, 255, 0.8)"   # Deep Sky Blue
}
# Background colors for the chart quadrants (subtle for dark mode)
QUADRANT_BG_COLORS = {
    "Leading": "rgba(0, 255, 0, 0.1)",
    "Weakening": "rgba(255, 165, 0, 0.1)",
    "Lagging": "rgba(255, 0, 0, 0.1)",
    "Improving": "rgba(0, 191, 255, 0.1)"
}
# Background colors for the data table rows (more opaque for readability)
TABLE_BG_COLORS = {
    "Leading": "rgba(0, 255, 0, 0.2)",
    "Weakening": "rgba(255, 165, 0, 0.2)",
    "Lagging": "rgba(255, 0, 0, 0.2)",
    "Improving": "rgba(0, 191, 255, 0.2)"
}

# ---------------------- DATA LOADING & PREP ---------------------- #

# --- NEW: Define a dictionary for the available datasets ---
# Use raw strings (r"...") to handle backslashes in Windows paths correctly.
DATASET_PATHS = {
    "Nifty MidSmallCap 400": r"C:\Users\FAHAD\Downloads\stock market file\ind_niftymidsmallcap400list.csv",
    "Nifty 500 Momentum 50": r"C:\Users\FAHAD\Downloads\stock market file\ind_nifty500Momentum50_list.csv",
    "Nifty SmallCap 250": r"C:\Users\FAHAD\Downloads\stock market file\ind_niftysmallcap250list.csv"
}

@st.cache_data
def load_indian_tickers(file_path):
    """Loads and prepares Indian stock tickers from a CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found. Please check the file path.")
        return []
    try:
        # Tickers are expected to be in a column named 'Symbol'
        df = pd.read_csv(file_path)
        tickers = df['Symbol'].dropna().unique().tolist()
        # Append '.NS' for Yahoo Finance compatibility for Indian stocks
        ns_tickers = [f"{ticker}.NS" for ticker in tickers]
        return ns_tickers
    except KeyError:
        st.error(f"Error: The CSV file at '{file_path}' must contain a column named 'Symbol'.")
        return []

# ---------------------- HELPER & CALCULATION FUNCTIONS ---------------------- #

@st.cache_data
def download_data(tickers, benchmark, period, interval):
    """Downloads historical price data from Yahoo Finance."""
    # Download close prices for all selected tickers and the benchmark
    data = yf.download(tickers + [benchmark], period=period, interval=interval, auto_adjust=True)['Close']
    return data.dropna(axis=1, how='all')

def calculate_rrg_coordinates(data, benchmark, lookback=21):
    """
    Calculates JdK RS-Ratio and JdK RS-Momentum.
    RS-Ratio is normalized around 100.
    RS-Momentum is the rate of change of RS-Ratio, centered around 0.
    """
    # 1. Calculate price ratio relative to the benchmark
    price_ratio = data.drop(columns=benchmark).div(data[benchmark], axis=0)

    # 2. Calculate JdK RS-Ratio (normalized ratio centered at 100)
    rs_ratio = 100 + ((price_ratio - price_ratio.mean()) / price_ratio.std())

    # 3. Calculate JdK RS-Momentum (rate of change of RS-Ratio, centered at 0)
    rs_momentum = rs_ratio.diff(lookback)

    return rs_ratio.iloc[-1], rs_momentum.iloc[-1]


def get_quadrant(rs_ratio, rs_momentum):
    """
    Determines the RRG quadrant based on the corrected logic.
    - LEADING:   Momentum > 0, Ratio > 100
    - IMPROVING: Momentum > 0, Ratio < 100
    - LAGGING:   Momentum < 0, Ratio < 100
    - WEAKENING: Momentum < 0, Ratio > 100
    """
    if rs_momentum > 0 and rs_ratio > 100:
        return "Leading"
    if rs_momentum > 0 and rs_ratio < 100:
        return "Improving"
    if rs_momentum < 0 and rs_ratio < 100:
        return "Lagging"
    if rs_momentum < 0 and rs_ratio > 100:
        return "Weakening"
    return "N/A"

@st.cache_data
def get_historical_trail(data, benchmark, lookback=21, trail_length=5):
    """Calculates the historical trail for each ticker on the RRG chart."""
    trails = {}
    # Iterate backwards from the most recent data to generate the trail path
    for i in range(trail_length, -1, -1):
        # Create a subset of data ending at the historical point
        subset = data.iloc[:-(i)] if i > 0 else data
        if len(subset) < lookback + 5:  # Ensure enough data for calculation
            continue
        
        ratio, mom = calculate_rrg_coordinates(subset, benchmark, lookback)
        for ticker in ratio.index:
            if ticker not in trails:
                trails[ticker] = {'x': [], 'y': []}
            trails[ticker]['x'].append(ratio[ticker])
            trails[ticker]['y'].append(mom[ticker])
    return trails

# ---------------------- UI & VISUALIZATION ---------------------- #

# --- Sidebar for User Inputs ---
st.sidebar.title("📊 RRG Settings")
st.sidebar.markdown("Configure the parameters for the Relative Rotation Graph.")

# --- NEW: Sidebar option to select the dataset ---
selected_dataset_name = st.sidebar.selectbox(
    "Select Stock Universe",
    options=list(DATASET_PATHS.keys()),
    index=0 # Default to the first file
)
selected_csv_path = DATASET_PATHS[selected_dataset_name]
available_tickers = load_indian_tickers(selected_csv_path)


benchmark_ticker = st.sidebar.text_input(
    "Benchmark Ticker",
    "^NSEI" # Default to NIFTY 50 for Indian market
)

# --- MODIFIED: Ticker selection now has two parts ---
st.sidebar.markdown("### Select Stocks")
# Part 1: Multiselect from the chosen file
preselected_tickers = st.sidebar.multiselect(
    "Select from List",
    options=available_tickers,
    default=available_tickers[:10] if available_tickers else [] # Default to first 10
)

# Part 2: Text area for custom tickers
custom_tickers_str = st.sidebar.text_area(
    "Add Custom Tickers (comma-separated)",
    "RELIANCE, TCS, INFY",
    help="Enter stock symbols like 'RELIANCE' or 'BAJFINANCE'. The '.NS' suffix will be added automatically."
)

# --- NEW: Logic to process and combine all selected tickers ---
final_selected_tickers = set(preselected_tickers) # Use a set to handle duplicates

if custom_tickers_str:
    # Split by comma, remove whitespace, and convert to uppercase
    custom_tickers_list = [ticker.strip().upper() for ticker in custom_tickers_str.split(',')]
    # Add '.NS' suffix if it's not already there
    processed_custom_tickers = [
        f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        for ticker in custom_tickers_list if ticker
    ]
    final_selected_tickers.update(processed_custom_tickers)


period = st.sidebar.selectbox(
    "Data Period",
    ["3mo", "6mo", "1y", "2y", "5y", "ytd"],
    index=2 # Default to '1y'
)

interval = st.sidebar.selectbox(
    "Data Interval",
    ["1d", "1wk", "1mo"],
    index=0 # Default to '1d'
)

lookback_period = st.sidebar.slider(
    "Momentum Lookback (Days)",
    min_value=5, max_value=50, value=21 # Default to 21
)

show_trail = st.sidebar.checkbox("Show Historical Trail", value=True)
trail_length = 5 # Fixed trail length for visual clarity


# --- Main App ---
st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")
st.markdown("""
This tool visualizes the relative strength and momentum of Indian stocks against a benchmark. The chart is divided into four quadrants based on performance relative to the benchmark:
- **<span style='color:green;'>Leading</span>**: Strong relative strength and positive momentum.
- **<span style='color:blue;'>Improving</span>**: Weak relative strength but improving momentum.
- **<span style='color:red;'>Lagging</span>**: Weak relative strength and negative momentum.
- **<span style='color:orange;'>Weakening</span>**: Strong relative strength but weakening momentum.
""", unsafe_allow_html=True)


# --- Data Processing and Chart Generation ---
if not final_selected_tickers:
    st.warning("Please select at least one stock from the sidebar.")
elif not benchmark_ticker:
    st.warning("Please enter a benchmark ticker.")
elif not available_tickers and not custom_tickers_str:
     st.error("Ticker list could not be loaded and no custom tickers were entered. Check the CSV file path and format.")
else:
    with st.spinner("Downloading data and generating RRG..."):
        # Convert set to list for yfinance
        all_tickers_to_download = list(final_selected_tickers)
        raw_data = download_data(all_tickers_to_download, benchmark_ticker, period, interval)

        if raw_data.empty or benchmark_ticker not in raw_data.columns:
            st.error(f"Could not download data for the benchmark '{benchmark_ticker}'. Please check the ticker.")
        elif len(raw_data.columns) <= 1:
            st.error("Could not download valid data for any of the selected stocks. Please check the tickers.")
        else:
            
            # --- NEW: Top 10 Returns Table ---
            st.subheader(f"Top 10 Performers (Last 21 Trading Periods)")
            if len(raw_data) >= 22:
                returns_21d = (raw_data.iloc[-1] / raw_data.iloc[-22] - 1) * 100
                returns_21d = returns_21d.drop(benchmark_ticker, errors='ignore')
                top_10 = returns_21d.sort_values(ascending=False).head(10).reset_index()
                top_10.columns = ['Ticker', 'Return (%)']
                top_10['Return (%)'] = top_10['Return (%)'].map('{:.2f}%'.format)
                st.table(top_10)
            else:
                st.warning(f"Not enough data to calculate 21-period returns. Downloaded data has {len(raw_data)} points. Please select a longer 'Data Period'.")
            
            # Calculate RRG coordinates for the main plot
            rs_ratio, rs_momentum = calculate_rrg_coordinates(raw_data, benchmark_ticker, lookback_period)

            # Prepare data for plotting
            plot_data = pd.DataFrame({
                'Ticker': rs_ratio.index,
                'RS-Ratio': rs_ratio.values,
                'RS-Momentum': rs_momentum.values
            })
            plot_data['Quadrant'] = plot_data.apply(lambda row: get_quadrant(row['RS-Ratio'], row['RS-Momentum']), axis=1)
            plot_data = plot_data.dropna()

            # Create Plotly RRG Chart
            fig = go.Figure()

            # Determine plot boundaries dynamically
            x_min, x_max = plot_data['RS-Ratio'].min() - 1, plot_data['RS-Ratio'].max() + 1
            y_min, y_max = plot_data['RS-Momentum'].min() - 0.5, plot_data['RS-Momentum'].max() + 0.5
            
            # Add quadrant background colors
            fig.add_shape(type="rect", xref="x", yref="y", x0=100, y0=0, x1=x_max+5, y1=y_max+5, fillcolor=QUADRANT_BG_COLORS['Leading'], layer="below", line_width=0)
            fig.add_shape(type="rect", xref="x", yref="y", x0=100, y0=y_min-5, x1=x_max+5, y1=0, fillcolor=QUADRANT_BG_COLORS['Weakening'], layer="below", line_width=0)
            fig.add_shape(type="rect", xref="x", yref="y", x0=x_min-5, y0=y_min-5, x1=100, y1=0, fillcolor=QUADRANT_BG_COLORS['Lagging'], layer="below", line_width=0)
            fig.add_shape(type="rect", xref="x", yref="y", x0=x_min-5, y0=0, x1=100, y1=y_max+5, fillcolor=QUADRANT_BG_COLORS['Improving'], layer="below", line_width=0)

            # Plot scatter points for each quadrant
            for quadrant, color in QUADRANT_COLORS.items():
                quad_data = plot_data[plot_data['Quadrant'] == quadrant]
                fig.add_trace(go.Scatter(
                    x=quad_data['RS-Ratio'],
                    y=quad_data['RS-Momentum'],
                    text=quad_data['Ticker'].str.replace(".NS", ""), # Clean ticker names for display
                    mode='markers+text',
                    textposition='top right',
                    marker=dict(color=color, size=12, line=dict(width=1, color='DarkSlateGrey')),
                    name=quadrant,
                    textfont=dict(color='white') # Ensure text is white
                ))

            # Add historical trails if enabled
            if show_trail:
                trails = get_historical_trail(raw_data, benchmark_ticker, lookback_period, trail_length)
                for ticker, trail_data in trails.items():
                    if ticker in plot_data['Ticker'].values:
                        final_quadrant = plot_data.loc[plot_data['Ticker'] == ticker, 'Quadrant'].iloc[0]
                        trail_color = QUADRANT_COLORS.get(final_quadrant, 'grey')
                        fig.add_trace(go.Scatter(
                            x=trail_data['x'],
                            y=trail_data['y'],
                            mode='lines',
                            line=dict(color=trail_color, width=1, dash='dot'),
                            hoverinfo='none',
                            showlegend=False
                        ))

            # --- MODIFIED: Center lines for dark theme ---
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")
            fig.add_vline(x=100, line_width=1, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)")

            # --- MODIFIED: Update layout for dark theme ---
            fig.update_layout(
                title=f'RRG of Selected Stocks vs. {benchmark_ticker} ({datetime.now().strftime("%Y-%m-%d")})',
                xaxis_title='JdK RS-Ratio (Relative Strength)',
                yaxis_title='JdK RS-Momentum',
                xaxis=dict(
                    range=[x_min, x_max],
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                height=700,
                legend_title_font_color="white",
                legend_title="Quadrants",
                plot_bgcolor='#1E1E1E',  # Dark background for the plot area
                paper_bgcolor='#1E1E1E', # Dark background for the entire chart
                font_color='white'       # White text for titles, axes, etc.
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- MODIFIED: Display data table with dark theme friendly colors ---
            st.subheader("RRG Data")
            display_df = plot_data[['Ticker', 'RS-Ratio', 'RS-Momentum', 'Quadrant']].copy()
            display_df['RS-Ratio'] = display_df['RS-Ratio'].round(2)
            display_df['RS-Momentum'] = display_df['RS-Momentum'].round(2)
            st.dataframe(display_df.style.apply(
                lambda row: [f'background-color: {TABLE_BG_COLORS.get(row.Quadrant, "transparent")}' for c in row], axis=1
            ), use_container_width=True)