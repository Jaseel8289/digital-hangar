import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

# ==========================================
# PAGE CONFIGURATION & UI OVERRIDES
# ==========================================
st.set_page_config(layout="wide", page_title="Digital Hangar v5.0")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("Digital Hangar v5.0: Environment-Aware Prognostics")
st.write("Condition-Based Normalization (CBN) & Universal PINN Evaluation")

# ==========================================
# 1. DATA & MODEL PIPELINE (Cached)
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'universal_pinn_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

@st.cache_data
def process_cbn_data(filename):
    # Load raw complex environment data
    cols = ['unit', 'cycles', 'alt', 'mach', 'tra'] + [f's_{i}' for i in range(1, 22)]
    try:
        # FIXED: Swapped delim_whitespace=True for sep=r'\s+' for Streamlit Cloud compatibility
        raw_df = pd.read_csv(filename, sep=r'\s+', names=cols)
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return None, None, None
        
    # Thermodynamic Feature Engineering
    raw_df['hpc_temp_rise'] = raw_df['s_3'] - raw_df['s_2']
    raw_df['thermo_ratio'] = raw_df['s_11'] / raw_df['s_4']
    
    # Smart Feature Selection
    stds = raw_df.std()
    flat_cols = stds[stds == 0].index
    raw_df.drop(columns=flat_cols, inplace=True)
    sensor_cols = [c for c in raw_df.columns if c.startswith('s_') or c in ['hpc_temp_rise', 'thermo_ratio']]
    
    # 1. K-Means Clustering for Flight Regimes
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    raw_df['regime'] = kmeans.fit_predict(raw_df[['alt', 'mach', 'tra']])
    
    # 2. Condition-Based Normalization (CBN)
    norm_df = raw_df.copy()
    for col in sensor_cols:
        norm_df[col] = norm_df.groupby('regime')[col].transform(
            lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten() if x.max() != x.min() else 0
        )
        
    # Ground Truth RUL
    max_cycles = raw_df.groupby('unit')['cycles'].max()
    raw_df['True_RUL'] = raw_df['unit'].map(max_cycles) - raw_df['cycles']
    norm_df['True_RUL'] = raw_df['True_RUL']
    
    return raw_df, norm_df, sensor_cols

# ==========================================
# 2. SIDEBAR CONTROLS & DATASET SELECTION
# ==========================================
st.sidebar.header("Fleet Telemetry Controls")

dataset_choice = st.sidebar.selectbox(
    "Select NASA C-MAPSS Dataset",
    ["FD002", "FD004"],
    index=1 # Default to FD004
)
dataset_filename = f"train_{dataset_choice}.txt"

model = load_model()
raw_df, norm_df, sensor_cols = process_cbn_data(dataset_filename)

if raw_df is None:
    st.error(f"Dataset {dataset_filename} not found. Please ensure it is in the directory.")
    st.stop()

if model is None:
    st.sidebar.warning("Model 'universal_pinn_model.keras' not found. Train the universal model first.")

selected_engine = st.sidebar.selectbox("Select Aircraft Engine ID", raw_df['unit'].unique())

engine_raw = raw_df[raw_df['unit'] == selected_engine].copy()
engine_norm = norm_df[norm_df['unit'] == selected_engine].copy()
max_cycles = int(engine_raw['cycles'].max())

current_cycle = st.sidebar.slider("Timeline Position (Flight Cycles)", 1, max_cycles, int(max_cycles/2))

# ==========================================
# 3. DATASET DESCRIPTIONS
# ==========================================
with st.expander("View Dataset Operating Conditions and Fault Modes"):
    st.markdown("""
    **FD002:** 6 Operating Conditions (Dynamic Flight Envelope), 1 Fault Mode (HPC Degradation)
    **FD004:** 6 Operating Conditions (Dynamic Flight Envelope), 2 Fault Modes (HPC and Fan Degradation)
    
    The neural network evaluating this data was trained exclusively on FD004, enabling it to generalize across all sub-variants by processing the highest degree of operational complexity.
    """)

st.info("Instruction: Move the Timeline Position slider in the sidebar to observe how Condition-Based Normalization clarifies the degradation trend compared to the raw sensor volatility.")

# ==========================================
# 4. TABBED NAVIGATION
# ==========================================
tab1, tab2 = st.tabs(["Telemetry & Normalization", "RUL Prediction"])

with tab1:
    st.subheader("1. Active Flight Envelope Tracker")
    current_snapshot = engine_raw[engine_raw['cycles'] == current_cycle].iloc[0]
    current_regime = int(current_snapshot['regime'])

    # Assigning colors/labels to regimes purely for visualization
    regime_labels = {
        0: ("Cruise", "blue"), 1: ("Takeoff", "red"), 2: ("Climb", "orange"), 
        3: ("Descent", "green"), 4: ("Hold", "purple"), 5: ("Approach", "cyan")
    }
    regime_name, regime_color = regime_labels.get(current_regime % 6, ("Unknown", "gray"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Altitude Parameter", f"{current_snapshot['alt']:.2f}")
    col2.metric("Mach Number", f"{current_snapshot['mach']:.2f}")
    col3.metric("Throttle Angle (TRA)", f"{current_snapshot['tra']:.2f}")
    col4.markdown(f"### Regime {current_regime}: <span style='color:{regime_color}'>{regime_name}</span>", unsafe_allow_html=True)
    st.write("---")

    st.subheader("2. Condition-Based Normalization")
    st.write("By clustering the environment into 6 operational regimes and scaling the thermodynamic parameters within those specific regimes, the pipeline extracts the underlying degradation trend from the environmental noise.")

    visible_raw = engine_raw[engine_raw['cycles'] <= current_cycle]
    visible_norm = engine_norm[engine_norm['cycles'] <= current_cycle]

    fig_cbn = make_subplots(rows=1, cols=2, subplot_titles=("Raw Telemetry", "CBN Normalized Tensor (Model Input)"))

    # Raw Sensor 4
    fig_cbn.add_trace(go.Scatter(x=visible_raw['cycles'], y=visible_raw['s_4'], 
                                 name="Raw HPC Temp", line=dict(color='#ff4b4b', width=2)), row=1, col=1)

    # Normalized Sensor 4
    fig_cbn.add_trace(go.Scatter(x=visible_norm['cycles'], y=visible_norm['s_4'], 
                                 name="Normalized HPC Temp", line=dict(color='#00f2fe', width=2)), row=1, col=2)

    fig_cbn.update_layout(height=450, template="plotly_dark", showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    fig_cbn.update_xaxes(title_text="Flight Cycles")
    fig_cbn.update_yaxes(title_text="Temperature (°R)", row=1, col=1)
    fig_cbn.update_yaxes(title_text="Scaled Metric [0-1]", row=1, col=2)
    st.plotly_chart(fig_cbn, use_container_width=True)

with tab2:
    st.subheader("Universal PINN Prognostic Output")

    @st.cache_data
    def generate_universal_predictions(engine_id, dataset_name):
        if model is None: return np.zeros(max_cycles)
        
        engine_vals = norm_df[norm_df['unit'] == engine_id][sensor_cols].values
        batch_X = []
        
        for i in range(1, len(engine_vals) + 1):
            window = engine_vals[:i]
            if len(window) < 50:
                pad_array = np.tile(window[0], (50 - len(window), 1))
                window = np.vstack((pad_array, window))
            else:
                window = window[-50:]
            batch_X.append(window)
            
        return model.predict(np.array(batch_X), verbose=0).flatten()

    pinn_rul_line = generate_universal_predictions(selected_engine, dataset_choice)

    # Final RUL Metric Boxes (Placed prominently at the top of the tab)
    if model is not None:
        current_rul = max(0, float(pinn_rul_line[current_cycle - 1]))
        predicted_lifespan = current_cycle + current_rul
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            if current_rul <= 30:
                st.error(f"MAINTENANCE REQUIRED: Overhaul within {int(current_rul)} flights")
            else:
                st.metric("Predicted Remaining Useful Life (RUL)", f"{int(current_rul)} Flights")
                
        with m_col2:
            st.metric("CNN-PINN Predicted Total Lifespan", f"{int(predicted_lifespan)} Flights")
            
        st.write("---")

    fig_rul = go.Figure()

    # Ground Truth
    fig_rul.add_trace(go.Scatter(x=engine_raw['cycles'], y=engine_raw['True_RUL'], 
                                 name='True RUL (Actual Life Window)', 
                                 line=dict(color='gray', width=3, dash='dash')))

    # PINN Prediction
    if model is not None:
        fig_rul.add_trace(go.Scatter(x=engine_raw['cycles'], y=pinn_rul_line, 
                                     name="Universal PINN Output", mode='lines', 
                                     line=dict(color='#00f2fe', width=3), opacity=0.9))

    fig_rul.update_layout(height=550, template="plotly_dark", hovermode="x unified",
                          xaxis_title="Accumulated Flight Cycles", yaxis_title="Remaining Useful Life Estimate (Flights)",
                          legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"))
    fig_rul.update_xaxes(range=[0, max_cycles + 5])
    fig_rul.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig_rul, use_container_width=True)
