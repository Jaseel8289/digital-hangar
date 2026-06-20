# Digital Hangar v4.0: Turbofan Prognostics and MLOps Suite

An engineering-focused predictive maintenance (PdM) dashboard built to monitor, benchmark, and evaluate multi-generation Machine Learning pipelines against simulated run-to-failure jet engine telemetry.
Live Web Application: https://digital-hangar-turbofan-prognostics.streamlit.app/

## Data Source and Provenance
- Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- Flight Regime: Sea-level operational profiles (FD001)
- Training Input: train_FD001.txt (Run-to-failure degradation trajectories)
- Validation Target: Blind Remaining Useful Life (RUL) metrics via test_FD001

## v4.0 (Active Development)
Current and future architecture updates include:
- Deep Learning Integration: Transitioning from static tree-based models to temporal sequence modeling using TensorFlow/Keras (LSTMs) to capture the exact velocity of engine degradation over time.
- Enhanced Physics Engines: Expanding the thermodynamic lane to include deeper gas path analysis and Virtual Sensor approximations for unreadable high-temperature turbine states.
- Dimensionality Reduction: Automating data compression using Principal Component Analysis (PCA) to mathematically eliminate sensor noise prior to Neural Network ingestion.
- 
## System Architecture and Pipeline Lanes
The suite ingests raw data and processes it through three distinct inference lanes:
1. Thermodynamic Lane (Deployed): Calculates isentropic compressor efficiency via the Brayton Cycle and maps thermal breakdown via Exhaust Gas Temperature (EGT) Margins.
2. Feature Engineering Lane: Processes high-variance sensor inputs utilizing 5-cycle rolling window smoothing and custom interaction ratios.
3. Raw Telemetry Lane: Feeds high-dimensional, unfiltered sensor matrices to historical baseline architectures.

## Model Matrix and Benchmarking Results
- Physics-Informed XGBoost (v2.0): RMSE 28.67 (Ultra-lightweight edge deployment profile)
- XGBoost (Hyperparameter Optimized): RMSE 32.40
- XGBoost (Feature Selection): RMSE 34.10
- Basic Gradient Boosting (Raw Sensors): RMSE 38.50
- Random Forest Regressor (Baseline): RMSE 41.20

## How To Run Locally
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Launch the suite: streamlit run dashboard.py
