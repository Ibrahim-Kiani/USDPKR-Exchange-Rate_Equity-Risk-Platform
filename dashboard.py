import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Markets Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    # Load the prediction data
    usdpkr_data = pd.read_csv('data/models/usdpkr_stacked_full_predictions.csv')
    kse_data = pd.read_csv('data/models/kse_lasso_full_predictions.csv')
    
    # Convert date columns
    usdpkr_data['Date'] = pd.to_datetime(usdpkr_data['Date'])
    kse_data['Date'] = pd.to_datetime(kse_data['Date'])
    
    return usdpkr_data, kse_data

# Model metrics
usdpkr_metrics = {'R_squared': 0.688884, 'MSE': 577.717, 'MAE': 14.448}
kse_metrics = {'R_squared': 0.700925, 'MSE': 7546243, 'MAE': 2120.692}

# Main dashboard
def main():
    # Title
    st.markdown('<h1 class="main-header">📈 Financial Markets Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Predictive Analytics for USD/PKR Exchange Rate & KSE100 Index")
    
    # Load data
    try:
        usdpkr_data, kse_data = load_data()
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'usdpkr_stacked_full_predictions.csv' and 'kse_lasso_full_predictions.csv' are in the same directory.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Date range selector
    min_date = min(usdpkr_data['Date'].min(), kse_data['Date'].min())
    max_date = max(usdpkr_data['Date'].max(), kse_data['Date'].max())
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Market selector
    market_view = st.sidebar.selectbox(
        "Select Market View",
        ["Overview", "USD/PKR Analysis", "KSE100 Analysis", "Comparative Analysis"]
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        usdpkr_filtered = usdpkr_data[
            (usdpkr_data['Date'] >= pd.to_datetime(date_range[0])) & 
            (usdpkr_data['Date'] <= pd.to_datetime(date_range[1]))
        ]
        kse_filtered = kse_data[
            (kse_data['Date'] >= pd.to_datetime(date_range[0])) & 
            (kse_data['Date'] <= pd.to_datetime(date_range[1]))
        ]
    else:
        usdpkr_filtered = usdpkr_data
        kse_filtered = kse_data
    
    # Main content based on selection
    if market_view == "Overview":
        show_overview(usdpkr_filtered, kse_filtered)
    elif market_view == "USD/PKR Analysis":
        show_usdpkr_analysis(usdpkr_filtered)
    elif market_view == "KSE100 Analysis":
        show_kse_analysis(kse_filtered)
    elif market_view == "Comparative Analysis":
        show_comparative_analysis(usdpkr_filtered, kse_filtered)

def show_overview(usdpkr_data, kse_data):
    st.markdown('<div class="section-header">🎯 Model Performance Overview</div>', unsafe_allow_html=True)
    
    # Performance metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### USD/PKR Exchange Rate Model")
        st.metric("R² Score", f"{usdpkr_metrics['R_squared']:.3f}", "68.9% accuracy")
        st.metric("Mean Squared Error", f"{usdpkr_metrics['MSE']:.2f}")
        st.metric("Mean Absolute Error", f"{usdpkr_metrics['MAE']:.2f}")
    
    with col2:
        st.markdown("#### KSE100 Index Model")
        st.metric("R² Score", f"{kse_metrics['R_squared']:.3f}", "70.1% accuracy")
        st.metric("Mean Squared Error", f"{kse_metrics['MSE']:,.0f}")
        st.metric("Mean Absolute Error", f"{kse_metrics['MAE']:.2f}")
    
    # Quick insights
    st.markdown('<div class="section-header">📈 Market Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # USD/PKR trend
        fig_usd = go.Figure()
        fig_usd.add_trace(go.Scatter(
            x=usdpkr_data['Date'],
            y=usdpkr_data['Actual'],
            mode='lines',
            name='Actual USD/PKR',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_usd.add_trace(go.Scatter(
            x=usdpkr_data['Date'],
            y=usdpkr_data['Predicted'],
            mode='lines',
            name='Predicted USD/PKR',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_usd.update_layout(
            title="USD/PKR Exchange Rate Trends",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_usd, use_container_width=True)
    
    with col2:
        # KSE100 trend
        fig_kse = go.Figure()
        fig_kse.add_trace(go.Scatter(
            x=kse_data['Date'],
            y=kse_data['Actual'],
            mode='lines',
            name='Actual KSE100',
            line=dict(color='#2ca02c', width=2)
        ))
        fig_kse.add_trace(go.Scatter(
            x=kse_data['Date'],
            y=kse_data['Predicted'],
            mode='lines',
            name='Predicted KSE100',
            line=dict(color='#d62728', width=2)
        ))
        fig_kse.update_layout(
            title="KSE100 Index Trends",
            xaxis_title="Date",
            yaxis_title="Index Value",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_kse, use_container_width=True)

def show_usdpkr_analysis(usdpkr_data):
    st.markdown('<div class="section-header">💱 USD/PKR Exchange Rate Analysis</div>', unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_rate = usdpkr_data['Actual'].iloc[-1]
    predicted_rate = usdpkr_data['Predicted'].iloc[-1]
    rate_change = ((current_rate - usdpkr_data['Actual'].iloc[-30]) / usdpkr_data['Actual'].iloc[-30]) * 100 if len(usdpkr_data) > 30 else 0
    
    with col1:
        st.metric("Current Rate", f"PKR {current_rate:.2f}")
    with col2:
        st.metric("Predicted Rate", f"PKR {predicted_rate:.2f}")
    with col3:
        st.metric("30-Day Change", f"{rate_change:.2f}%")
    with col4:
        prediction_error = abs(current_rate - predicted_rate)
        st.metric("Prediction Error", f"PKR {prediction_error:.2f}")
    
    # Detailed chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Exchange Rate Comparison", "Prediction Error Over Time"),
        row_heights=[0.7, 0.3]
    )
    
    # Main comparison chart
    fig.add_trace(
        go.Scatter(
            x=usdpkr_data['Date'],
            y=usdpkr_data['Actual'],
            mode='lines',
            name='Actual Rate',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=usdpkr_data['Date'],
            y=usdpkr_data['Predicted'],
            mode='lines',
            name='Predicted Rate',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    # Error chart
    error = usdpkr_data['Actual'] - usdpkr_data['Predicted']
    fig.add_trace(
        go.Scatter(
            x=usdpkr_data['Date'],
            y=error,
            mode='lines',
            name='Prediction Error',
            line=dict(color='#d62728', width=1),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text="USD/PKR Exchange Rate Prediction Analysis",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Exchange Rate (PKR)", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Statistical Summary")
        stats = usdpkr_data['Actual'].describe()
        for stat, value in stats.items():
            st.text(f"{stat.title()}: {value:.2f}")
    
    with col2:
        st.markdown("#### Model Performance")
        st.text(f"R² Score: {usdpkr_metrics['R_squared']:.4f}")
        st.text(f"MSE: {usdpkr_metrics['MSE']:.2f}")
        st.text(f"MAE: {usdpkr_metrics['MAE']:.2f}")
        st.text(f"MAPE: {(abs(error).mean() / usdpkr_data['Actual'].mean() * 100):.2f}%")

def show_kse_analysis(kse_data):
    st.markdown('<div class="section-header">📊 KSE100 Index Analysis</div>', unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_index = kse_data['Actual'].iloc[-1]
    predicted_index = kse_data['Predicted'].iloc[-1]
    index_change = ((current_index - kse_data['Actual'].iloc[-30]) / kse_data['Actual'].iloc[-30]) * 100 if len(kse_data) > 30 else 0
    
    with col1:
        st.metric("Current Index", f"{current_index:,.0f}")
    with col2:
        st.metric("Predicted Index", f"{predicted_index:,.0f}")
    with col3:
        st.metric("30-Day Change", f"{index_change:.2f}%")
    with col4:
        prediction_error = abs(current_index - predicted_index)
        st.metric("Prediction Error", f"{prediction_error:,.0f}")
    
    # Detailed chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("KSE100 Index Comparison", "Prediction Error Over Time"),
        row_heights=[0.7, 0.3]
    )
    
    # Main comparison chart
    fig.add_trace(
        go.Scatter(
            x=kse_data['Date'],
            y=kse_data['Actual'],
            mode='lines',
            name='Actual Index',
            line=dict(color='#2ca02c', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=kse_data['Date'],
            y=kse_data['Predicted'],
            mode='lines',
            name='Predicted Index',
            line=dict(color='#d62728', width=2)
        ),
        row=1, col=1
    )
    
    # Error chart
    error = kse_data['Actual'] - kse_data['Predicted']
    fig.add_trace(
        go.Scatter(
            x=kse_data['Date'],
            y=error,
            mode='lines',
            name='Prediction Error',
            line=dict(color='#ff7f0e', width=1),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text="KSE100 Index Prediction Analysis",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Index Value", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Statistical Summary")
        stats = kse_data['Actual'].describe()
        for stat, value in stats.items():
            st.text(f"{stat.title()}: {value:,.0f}")
    
    with col2:
        st.markdown("#### Model Performance")
        st.text(f"R² Score: {kse_metrics['R_squared']:.4f}")
        st.text(f"MSE: {kse_metrics['MSE']:,}")
        st.text(f"MAE: {kse_metrics['MAE']:,.0f}")
        st.text(f"MAPE: {(abs(error).mean() / kse_data['Actual'].mean() * 100):.2f}%")

def show_comparative_analysis(usdpkr_data, kse_data):
    st.markdown('<div class="section-header">🔄 Comparative Market Analysis</div>', unsafe_allow_html=True)
    
    # Correlation analysis (if dates align)
    merged_data = pd.merge(usdpkr_data, kse_data, on='Date', suffixes=('_usd', '_kse'))
    
    if not merged_data.empty:
        correlation = merged_data['Actual_usd'].corr(merged_data['Actual_kse'])
        st.info(f"Correlation between USD/PKR and KSE100: {correlation:.3f}")
    
    # Side-by-side comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("USD/PKR Actual vs Predicted", "KSE100 Actual vs Predicted",
                       "USD/PKR Prediction Accuracy", "KSE100 Prediction Accuracy"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # USD/PKR comparison
    fig.add_trace(
        go.Scatter(x=usdpkr_data['Date'], y=usdpkr_data['Actual'], 
                  name='USD/PKR Actual', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=usdpkr_data['Date'], y=usdpkr_data['Predicted'], 
                  name='USD/PKR Predicted', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    # KSE100 comparison
    fig.add_trace(
        go.Scatter(x=kse_data['Date'], y=kse_data['Actual'], 
                  name='KSE100 Actual', line=dict(color='#2ca02c')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=kse_data['Date'], y=kse_data['Predicted'], 
                  name='KSE100 Predicted', line=dict(color='#d62728')),
        row=1, col=2
    )
    
    # Scatter plots for accuracy
    fig.add_trace(
        go.Scatter(x=usdpkr_data['Actual'], y=usdpkr_data['Predicted'],
                  mode='markers', name='USD/PKR Accuracy',
                  marker=dict(color='#1f77b4', opacity=0.6)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=kse_data['Actual'], y=kse_data['Predicted'],
                  mode='markers', name='KSE100 Accuracy',
                  marker=dict(color='#2ca02c', opacity=0.6)),
        row=2, col=2
    )
    
    # Add perfect prediction line
    usd_range = [usdpkr_data['Actual'].min(), usdpkr_data['Actual'].max()]
    kse_range = [kse_data['Actual'].min(), kse_data['Actual'].max()]
    
    fig.add_trace(
        go.Scatter(x=usd_range, y=usd_range, mode='lines',
                  name='Perfect Prediction', line=dict(dash='dash', color='red')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=kse_range, y=kse_range, mode='lines',
                  name='Perfect Prediction', line=dict(dash='dash', color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Comparative Market Analysis")
    fig.update_xaxes(title_text="Actual Value", row=2, col=1)
    fig.update_xaxes(title_text="Actual Value", row=2, col=2)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison table
    st.markdown("#### Model Performance Comparison")
    comparison_df = pd.DataFrame({
        'Metric': ['R² Score', 'MSE', 'MAE'],
        'USD/PKR Model': [f"{usdpkr_metrics['R_squared']:.4f}", 
                         f"{usdpkr_metrics['MSE']:.2f}", 
                         f"{usdpkr_metrics['MAE']:.2f}"],
        'KSE100 Model': [f"{kse_metrics['R_squared']:.4f}", 
                        f"{kse_metrics['MSE']:,.0f}", 
                        f"{kse_metrics['MAE']:.2f}"]
    })
    st.table(comparison_df)
    
    # Feature importance explanation
    st.markdown("#### Model Features")
    st.markdown("""
    **Key Economic Indicators Used:**
    - **Gold Prices**: Precious metal as safe haven asset
    - **Forex Reserves**: Foreign exchange reserves indicator
    - **Inflation (YoY)**: Year-over-year inflation rate
    - **Oil Prices**: Crude oil price impact
    - **Interest Rates**: Monetary policy indicator
    - **Lagged Variables**: Historical values for trend analysis
    - **Rolling Statistics**: Moving averages and volatilities
    """)

if __name__ == "__main__":
    main()