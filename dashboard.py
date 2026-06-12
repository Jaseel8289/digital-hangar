import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# PAGE CONFIGURATION & UI OVERRIDES
# ==========================================
st.set_page_config(layout="wide", page_title="Digital Hangar v3.5")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("Digital Hangar v3.5: Comprehensive Prognostics Suite")
st.write("Multi-Model Competitive Benchmarking and Thermodynamic Diagnostic Environment | Powered by NASA C-MAPSS Telemetry")

# ==========================================
# 1. MODEL CONFIGURATION 
# ==========================================
MODEL_DICTIONARY = {
    "Physics-Informed XGBoost (v2.0)": {
        "file": "physics_xgb_model.pkl", "type": "physics", "color": "red",
        "features": "3 Columns: Time Cycles, HPC Efficiency (η), EGT Margin",
        "rmse": "28.67", "mae": "21.10", "r2": "78.4%",
        "weightage": "Ultra-Light (Physics Engine + Minimal Trees)",
        "desc": "The primary operational framework. Completely abandons statistical guessing by translating raw telemetry into pure thermodynamics (Brayton Cycle) prior to ingestion. Because the input features represent the true physical root cause of engine failure, the algorithm achieves superior accuracy on unseen data with minimal computational overhead."
    },
    "Random Forest Regressor (Baseline)": {
        "file": "turbofan_god_model.pkl", "type": "raw", "color": "orange",
        "features": "22 Columns: Time Cycles + All 21 Raw Sensor Streams",
        "rmse": "41.20", "mae": "31.45", "r2": "54.2%",
        "weightage": "Heavy (100 Deep Trees, Unfiltered High-Dimensionality)",
        "desc": "A foundational statistical baseline. Evaluates all available high-noise data to identify statistical correlations. Susceptible to overfitting, resulting in a highly jagged and 'steppy' prediction curve."
    },
    "Basic Gradient Boosting (Raw Sensors)": {
        "file": "watchdog.pkl", "type": "raw", "color": "green",
        "features": "22 Columns: Time Cycles + All 21 Raw Sensor Streams",
        "rmse": "38.50", "mae": "29.12", "r2": "61.8%",
        "weightage": "Moderate (Sequential Boosting, Full Dimensionality)",
        "desc": "An architectural upgrade utilizing sequential error correction. Smooths out predictions compared to the Random Forest, but absolute accuracy remains fundamentally constrained by un-smoothed sensor noise."
    },
    "XGBoost (Feature Selection)": {
        "file": "xgb_greedy.pkl", "type": "engineered", "color": "purple",
        "features": "6 Columns: Time Cycles, s_11/s_15/s_2 rolling averages, Custom Interaction Ratios",
        "rmse": "34.10", "mae": "25.80", "r2": "69.5%",
        "weightage": "Light (Reduced Dimensionality, 6 Features)",
        "desc": "A refined ensemble leveraging 5-cycle rolling averages and engineered cross-products (e.g., pseudo-efficiency) to isolate degradation trends. Eliminating 18 high-noise sensor streams resulted in a significant accuracy improvement."
    },
    "XGBoost (Hyperparameter Optimized)": {
        "file": "xgb_tuned.pkl", "type": "engineered", "color": "magenta",
        "features": "6 Columns: Time Cycles, s_11/s_15/s_2 rolling averages, Custom Interaction Ratios",
        "rmse": "32.40", "mae": "24.35", "r2": "72.1%",
        "weightage": "Optimized (Shallow Depth, Controlled Learning Rate)",
        "desc": "The empirical limit of purely statistical approaches on this dataset. Utilizing constrained tree depth and minimized learning rates prevents the memorization of anomalies, yielding a stable degradation curve."
    }
}

# ==========================================
# 2. LOAD ASSETS (Cached for Performance)
# ==========================================
@st.cache_resource
def load_models():
    loaded_models = {}
    missing_models = []
    for name, config in MODEL_DICTIONARY.items():
        if os.path.exists(config["file"]):
            with open(config["file"], 'rb') as f:
                loaded_models[name] = pickle.load(f)
        else:
            missing_models.append(name)
    return loaded_models, missing_models

@st.cache_data
def load_data():
    df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None)
    sensor_names = [f's_{i}' for i in range(1, 22)]
    df.columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + sensor_names
    return df

loaded_models, missing_models = load_models()
raw_df = load_data()

if missing_models:
    st.sidebar.warning(f"Missing model archives: {', '.join(missing_models)}")

# ==========================================
# 3. DATA STREAM PROCESSING PIPELINE
# ==========================================
@st.cache_data
def process_data(df):
    # Lane 1: Thermodynamics
    gamma = 1.4
    actual_rise = df['s_3'] - df['s_1']
    pressure_ratio = df['s_7'] / df['s_5']
    ideal_rise = df['s_1'] * ((pressure_ratio ** ((gamma - 1) / gamma)) - 1)
    df['hpc_efficiency'] = np.where(actual_rise > 0, ideal_rise / actual_rise, 0)
    df['egt_margin'] = 1440.0 - df['s_4']
    df['hpc_efficiency_smooth'] = df.groupby('unit_number')['hpc_efficiency'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['egt_margin_smooth'] = df.groupby('unit_number')['egt_margin'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Lane 2: Signal Smoothing and Interactivity
    df['s_11_roll5'] = df.groupby('unit_number')['s_11'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['s_15_roll5'] = df.groupby('unit_number')['s_15'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['s_2_roll5'] = df.groupby('unit_number')['s_2'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['pseudo_efficiency'] = df['s_11_roll5'] / df['s_2_roll5']
    df['bypass_pressure_interaction'] = df['s_15_roll5'] / df['s_11_roll5']
    return df

processed_df = process_data(raw_df)

# ==========================================
# 4. SIDEBAR TIMELINE MANAGER & ENVIRONMENT
# ==========================================
st.sidebar.header("Fleet Telemetry Controls")
selected_engine = st.sidebar.selectbox("Select Aircraft Engine ID", processed_df['unit_number'].unique())

engine_data = processed_df[processed_df['unit_number'] == selected_engine].copy()
max_cycles = int(engine_data['time_cycles'].max())
engine_data['True_RUL'] = max_cycles - engine_data['time_cycles']

current_cycle = st.sidebar.slider("Timeline Position (Flight Cycles)", 1, max_cycles, int(max_cycles/2))
visible_data = engine_data[engine_data['time_cycles'] <= current_cycle].copy()
current_snapshot = visible_data.iloc[-1]

st.sidebar.markdown("---")
st.sidebar.subheader("Active Flight Envelope")
st.sidebar.write(f"**Altitude Parameter:** `{current_snapshot['op_setting_1']:.4f}`")
st.sidebar.write(f"**Mach Parameter:** `{current_snapshot['op_setting_2']:.4f}`")
st.sidebar.write(f"**Throttle (TRA):** `{current_snapshot['op_setting_3']:.2f}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Provenance")
st.sidebar.caption("**Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)")
st.sidebar.caption("**Active Stream:** `train_FD001` (Simulating Full Run-to-Failure)")
st.sidebar.caption("**Validation Criteria:** Blind RMSE via `test_FD001`")

st.sidebar.markdown("---")
st.sidebar.info("Active Development (v4.0): This suite is currently being upgraded to incorporate TensorFlow sequence modeling (LSTMs) and advanced gas-path thermodynamics.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Developer Links")
st.sidebar.markdown("[View Source Code on GitHub](https://github.com/Jaseel8289/digital-hangar/tree/main)")

# Ingestion structures for inferences
physics_features = visible_data[['time_cycles', 'hpc_efficiency_smooth', 'egt_margin_smooth']]
raw_features = visible_data[['time_cycles'] + [f's_{i}' for i in range(1, 22)]]
engineered_features = visible_data[['time_cycles', 's_11_roll5', 's_15_roll5', 's_2_roll5', 'pseudo_efficiency', 'bypass_pressure_interaction']]

predictions_dict = {}
for name, model in loaded_models.items():
    m_type = MODEL_DICTIONARY[name]["type"]
    if m_type == "physics":
        predictions_dict[name] = model.predict(physics_features)
    elif m_type == "engineered":
        predictions_dict[name] = model.predict(engineered_features)
    else:
        predictions_dict[name] = model.predict(raw_features)

# ==========================================
# 5. NAVIGATION STRUCTURE
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "Live Operations & Diagnostics", 
    "Comparative Benchmarking", 
    "Model Specifications"
])

with tab1:
    st.subheader("Instantaneous Aerothermal Telemetry")
    col1, col2, col3 = st.columns(3)
    
    # Calculate Deltas for the metrics
    if len(visible_data) > 1:
        prev_hpc = visible_data.iloc[-2]['hpc_efficiency_smooth']
        prev_egt = visible_data.iloc[-2]['egt_margin_smooth']
        delta_hpc = f"{(current_snapshot['hpc_efficiency_smooth'] - prev_hpc)*100:.3f}%"
        delta_egt = f"{current_snapshot['egt_margin_smooth'] - prev_egt:.2f} °R"
    else:
        delta_hpc = None
        delta_egt = None

    col1.metric("HPC Efficiency (η)", f"{current_snapshot['hpc_efficiency_smooth']*100:.2f}%", delta=delta_hpc)
    col2.metric("EGT Safety Margin", f"{current_snapshot['egt_margin_smooth']:.1f} °R", delta=delta_egt)
    
    if "Physics-Informed XGBoost (v2.0)" in loaded_models:
        rul_pred = max(0, float(predictions_dict["Physics-Informed XGBoost (v2.0)"][-1]))
        if rul_pred <= 30:
            col3.error(f"MAINTENANCE REQUIRED: Overhaul within {int(rul_pred)} flights")
        else:
            col3.metric("Predicted Lifespan Remaining", f"{int(rul_pred)} Flights")
        
    st.write("---")
    st.subheader("Physical Parameter Tracking History")
    
    # Interactive Plotly Dual-Axis Chart
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(go.Scatter(x=visible_data['time_cycles'], y=visible_data['hpc_efficiency_smooth'], 
                              name="Efficiency (η)", line=dict(color='#1f77b4', width=3)), secondary_y=False)
    
    fig1.add_trace(go.Scatter(x=visible_data['time_cycles'], y=visible_data['egt_margin_smooth'], 
                              name="EGT Margin (°R)", line=dict(color='#d62728', width=3)), secondary_y=True)
    
    # Adjusted legend position to the top-left to avoid overlap with zoom tools
    fig1.update_layout(height=450, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig1.update_xaxes(title_text="Recorded Flight Cycles")
    fig1.update_yaxes(title_text="Efficiency (η)", secondary_y=False, color="#1f77b4")
    fig1.update_yaxes(title_text="EGT Margin (°R)", secondary_y=True, color="#d62728")
    
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Comparative Prognostic Alignment")
    available_models = list(loaded_models.keys())
    selected_models = st.multiselect(
        "Toggle models on/off for graphical intersection analysis:", 
        options=available_models, 
        default=available_models
    )
    st.write("---")
    
    # Interactive Plotly Benchmarking Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=visible_data['time_cycles'], y=visible_data['True_RUL'], 
                              name='True RUL (Actual Life Window)', 
                              line=dict(color='white', width=3, dash='dash')))
    
    for name in selected_models:
        color = MODEL_DICTIONARY[name]["color"]
        # Convert standard colors to slightly softer hex codes for better dark-mode viewing
        color_map = {"red": "#ff4b4b", "orange": "#ffa421", "green": "#21c354", "purple": "#803df5", "magenta": "#ff2b8f"}
        plot_color = color_map.get(color, color)
        
        fig2.add_trace(go.Scatter(x=visible_data['time_cycles'], y=predictions_dict[name], 
                                  name=name, mode='lines', line=dict(color=plot_color, width=2), opacity=0.8))
    
    fig2.update_layout(height=550, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                       xaxis_title="Accumulated Flight Cycles", yaxis_title="Remaining Useful Life Estimate (Flights)",
                       legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"))
    fig2.update_xaxes(range=[0, max_cycles + 5])
    fig2.update_yaxes(rangemode="tozero")
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Detailed Model Architecture Matrix")
    st.write("All predictive models were trained on the **NASA C-MAPSS FD001 Training Set** (Run-to-failure degradation trajectories). Validation scores reflect predictive accuracy measured against the unobserved **FD001 Blind Test Set**.")
    st.write("---")
    
    for name, config in MODEL_DICTIONARY.items():
        box = st.container()
        col_meta, col_details = st.columns([1, 3])
        
        with col_meta:
            if name == "Physics-Informed XGBoost (v2.0)":
                st.success(f"**{name} (Deployed)**")
            else:
                st.info(f"**{name}**")
            st.markdown(f"**Computational Weight:**<br>{config['weightage']}", unsafe_allow_html=True)
            
        with col_details:
            err1, err2, err3 = st.columns(3)
            err1.metric("Blind RMSE", config["rmse"])
            err2.metric("MAE (Mean Absolute Error)", config["mae"])
            err3.metric("R² Score", config["r2"])
            
            st.markdown(f"**Structural Architecture:** `{config['type'].upper()}` pipeline")
            st.markdown(f"**Ingested Vector Scope:** {config['features']}")
            st.markdown(f"**Performance Verdict:** {config['desc']}")
        st.write("---")
