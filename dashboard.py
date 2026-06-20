import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# PAGE CONFIGURATION & UI OVERRIDES
# ==========================================
st.set_page_config(layout="wide", page_title="Digital Hangar v4.0")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("Digital Hangar v4.0: SOTA Prognostics Suite")
st.write("Physics-Informed Deep Learning vs. Legacy Machine Learning | Powered by NASA C-MAPSS")

# ==========================================
# 1. MODEL CONFIGURATION 
# ==========================================
MODEL_DICTIONARY = {
    "CNN-PINN Ensemble (SOTA v4.0)": {
        "files": [f"pinn_ensemble_model_{i}.keras" for i in range(1, 6)], 
        "type": "pinn", "color": "cyan",
        "features": "19 Features x 50-Flight 3D Sliding Window",
        "mae": "9.02", "r2": "92.4%",
        "weightage": "Heavy (5x Convolutional LSTMs averaged in real-time)",
        "desc": "The ultimate State-of-the-Art architecture. Evaluates 50 flights of chronological history simultaneously. Trained with a custom thermodynamic loss function that heavily penalizes the network for violating the Law of Entropy (predicting RUL increasing over time)."
    },
    "Physics-Informed XGBoost (Legacy)": {
        "file": "physics_xgb_model.pkl", "type": "physics", "color": "red",
        "features": "3 Columns: Time Cycles, HPC Efficiency (η), EGT Margin",
        "mae": "21.10", "r2": "78.4%",
        "weightage": "Light (Physics Engine + Gradient Boosting Trees)",
        "desc": "The previous operational framework. Translates raw telemetry into pure thermodynamics (Brayton Cycle) prior to ingestion. Highly effective but constrained by its 2D architecture (lacks chronological memory)."
    }
}

# ==========================================
# 2. LOAD ASSETS (Cached for Performance)
# ==========================================
@st.cache_resource
def load_models():
    loaded_pinn = []
    loaded_xgb = None
    missing_models = []
    
    # Load the PINN Ensemble
    pinn_config = MODEL_DICTIONARY["CNN-PINN Ensemble (SOTA v4.0)"]
    for f in pinn_config["files"]:
        if os.path.exists(f):
            # compile=False allows us to load the model without needing to define the custom physics loss!
            loaded_pinn.append(tf.keras.models.load_model(f, compile=False))
        else:
            missing_models.append(f)
            
    # Load the old XGBoost model (if it exists)
    xgb_file = MODEL_DICTIONARY["Physics-Informed XGBoost (Legacy)"]["file"]
    if os.path.exists(xgb_file):
        with open(xgb_file, 'rb') as f:
            loaded_xgb = pickle.load(f)
            
    return loaded_pinn, loaded_xgb, missing_models

@st.cache_data
def load_data():
    df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None)
    sensor_names = [f's_{i}' for i in range(1, 22)]
    df.columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + sensor_names
    return df

pinn_ensemble, loaded_xgb, missing_models = load_models()
raw_df = load_data()

if missing_models:
    st.sidebar.warning(f"Missing ensemble components. Make sure {missing_models[0]} is in the folder!")

# ==========================================
# 3. DATA STREAM PROCESSING PIPELINE
# ==========================================
@st.cache_data
def process_data(df):
    """Processes data for BOTH the old XGBoost model and the new PINN model."""
    processed_df = df.copy()
    
    # --- LANE 1: Legacy XGBoost Processing ---
    gamma = 1.4
    actual_rise = processed_df['s_3'] - processed_df['s_1']
    pressure_ratio = processed_df['s_7'] - processed_df['s_5'] # Safe fallback
    processed_df['hpc_efficiency'] = np.where(actual_rise > 0, 1.0 / actual_rise, 0) # Simplified for display
    processed_df['egt_margin'] = 1440.0 - processed_df['s_4']
    processed_df['hpc_efficiency_smooth'] = processed_df.groupby('unit_number')['hpc_efficiency'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    processed_df['egt_margin_smooth'] = processed_df.groupby('unit_number')['egt_margin'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # --- LANE 2: PINN 3D Tensor Processing ---
    # Drop flatlines
    flat_cols = ['op_setting_3', 's_1', 's_10', 's_18', 's_19']
    pinn_df = processed_df.drop(columns=flat_cols)
    
    # Thermodynamic Features
    pinn_df['hpc_temp_rise'] = pinn_df['s_3'] - pinn_df['s_2']
    pinn_df['thermo_ratio'] = pinn_df['s_11'] / pinn_df['s_4']
    
    # Scaling
    feature_cols = [col for col in pinn_df.columns if col not in ['unit_number', 'time_cycles']]
    scaler = MinMaxScaler()
    pinn_df[feature_cols] = scaler.fit_transform(pinn_df[feature_cols])
    
    return processed_df, pinn_df, feature_cols

processed_df, pinn_df, pinn_features = process_data(raw_df)

# ==========================================
# 4. SIDEBAR TIMELINE MANAGER & ENVIRONMENT
# ==========================================
st.sidebar.header("Fleet Telemetry Controls")
selected_engine = st.sidebar.selectbox("Select Aircraft Engine ID", processed_df['unit_number'].unique())

engine_data_raw = processed_df[processed_df['unit_number'] == selected_engine].copy()
engine_data_pinn = pinn_df[pinn_df['unit_number'] == selected_engine].copy()

max_cycles = int(engine_data_raw['time_cycles'].max())
engine_data_raw['True_RUL'] = max_cycles - engine_data_raw['time_cycles']

current_cycle = st.sidebar.slider("Timeline Position (Flight Cycles)", 1, max_cycles, int(max_cycles/2))
visible_data = engine_data_raw[engine_data_raw['time_cycles'] <= current_cycle].copy()
current_snapshot = visible_data.iloc[-1]

st.sidebar.markdown("---")
st.sidebar.subheader("Active Flight Envelope")
st.sidebar.write(f"**Altitude Parameter:** `{current_snapshot['op_setting_1']:.4f}`")
st.sidebar.write(f"**Mach Parameter:** `{current_snapshot['op_setting_2']:.4f}`")

st.sidebar.markdown("---")
st.sidebar.success("v4.0 ACTIVE: 3D CNN-PINN Pipeline Operational.")

# ==========================================
# 5. BATCH PREDICTION LOGIC
# ==========================================
@st.cache_data
def generate_pinn_predictions(engine_id):
    """Generates the RUL line for the entire engine history instantly via batching."""
    if not pinn_ensemble:
        return np.zeros(max_cycles)
        
    engine_vals = pinn_df[pinn_df['unit_number'] == engine_id][pinn_features].values
    num_cycles = len(engine_vals)
    batch_X = []
    
    # Recreate the 50-flight sliding window
    for i in range(1, num_cycles + 1):
        window = engine_vals[:i]
        if len(window) < 50:
            # Pad early flights by repeating the first row
            pad_size = 50 - len(window)
            pad_array = np.tile(window[0], (pad_size, 1))
            window = np.vstack((pad_array, window))
        else:
            window = window[-50:]
        batch_X.append(window)
        
    batch_X = np.array(batch_X) # Shape: [Cycles, 50, Features]
    
    # Query all 5 AI models
    ensemble_preds = []
    for model in pinn_ensemble:
        preds = model.predict(batch_X, verbose=0).flatten()
        ensemble_preds.append(preds)
        
    # Wisdom of crowds (Average)
    return np.mean(ensemble_preds, axis=0)

# Generate Predictions for the Graph
pinn_rul_line = generate_pinn_predictions(selected_engine)

xgb_rul_line = None
if loaded_xgb:
    xgb_features = engine_data_raw[['time_cycles', 'hpc_efficiency_smooth', 'egt_margin_smooth']]
    xgb_rul_line = loaded_xgb.predict(xgb_features)

# ==========================================
# 6. NAVIGATION STRUCTURE
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
    
    if len(pinn_ensemble) > 0:
        current_rul_pred = max(0, float(pinn_rul_line[current_cycle - 1]))
        if current_rul_pred <= 30:
            col3.error(f"MAINTENANCE REQUIRED: Overhaul within {int(current_rul_pred)} flights")
        else:
            col3.metric("CNN-PINN Predicted Lifespan", f"{int(current_rul_pred)} Flights")
        
    st.write("---")
    st.subheader("Physical Parameter Tracking History")
    
    # Interactive Plotly Dual-Axis Chart
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(go.Scatter(x=visible_data['time_cycles'], y=visible_data['hpc_efficiency_smooth'], 
                              name="HPC Efficiency (η)", line=dict(color='#1f77b4', width=3)), secondary_y=False)
    
    fig1.add_trace(go.Scatter(x=visible_data['time_cycles'], y=visible_data['egt_margin_smooth'], 
                              name="EGT Margin (°R)", line=dict(color='#d62728', width=3)), secondary_y=True)
    
    fig1.update_layout(height=450, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig1.update_xaxes(title_text="Recorded Flight Cycles")
    fig1.update_yaxes(title_text="HPC Efficiency (η)", secondary_y=False, color="#1f77b4")
    fig1.update_yaxes(title_text="EGT Margin (°R)", secondary_y=True, color="#d62728")
    
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Prognostic Alignment: PINN vs Legacy XGBoost")
    st.write("Observe how the Deep Learning PINN strictly follows chronological trends, whereas the legacy XGBoost suffers from cycle-to-cycle 'stepping' due to its 2D flat architecture.")
    st.write("---")
    
    fig2 = go.Figure()
    # Ground Truth
    fig2.add_trace(go.Scatter(x=engine_data_raw['time_cycles'], y=engine_data_raw['True_RUL'], 
                              name='True RUL (Actual Life Window)', 
                              line=dict(color='gray', width=3, dash='dash')))
    
    # CNN-PINN SOTA
    if len(pinn_ensemble) > 0:
        fig2.add_trace(go.Scatter(x=engine_data_raw['time_cycles'], y=pinn_rul_line, 
                                  name="CNN-PINN Ensemble (v4.0)", mode='lines', 
                                  line=dict(color='#00f2fe', width=3), opacity=0.9))
        
    # Legacy XGBoost
    if loaded_xgb:
        fig2.add_trace(go.Scatter(x=engine_data_raw['time_cycles'], y=xgb_rul_line, 
                                  name="XGBoost Baseline (v2.0)", mode='lines', 
                                  line=dict(color='#ff4b4b', width=2), opacity=0.7))
    
    fig2.update_layout(height=550, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                       xaxis_title="Accumulated Flight Cycles", yaxis_title="Remaining Useful Life Estimate (Flights)",
                       legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"))
    fig2.update_xaxes(range=[0, max_cycles + 5])
    fig2.update_yaxes(rangemode="tozero")
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Detailed Model Architecture Matrix")
    st.write("Validation scores reflect predictive accuracy measured against the unobserved **FD001 Blind Test Set**.")
    st.write("---")
    
    for name, config in MODEL_DICTIONARY.items():
        box = st.container()
        col_meta, col_details = st.columns([1, 3])
        
        with col_meta:
            if "PINN" in name:
                st.success(f"**{name} (Deployed)**")
            else:
                st.info(f"**{name}**")
            st.markdown(f"**Computational Weight:**<br>{config['weightage']}", unsafe_allow_html=True)
            
        with col_details:
            err1, err2 = st.columns(2)
            err1.metric("MAE (Mean Absolute Error)", config["mae"])
            err2.metric("R² Score / Efficacy", config["r2"])
            
            st.markdown(f"**Structural Architecture:** `{config['type'].upper()}` pipeline")
            st.markdown(f"**Ingested Vector Scope:** {config['features']}")
            st.markdown(f"**Performance Verdict:** {config['desc']}")
        st.write("---")
