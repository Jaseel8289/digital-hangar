Digital Hangar v5.0: Turbofan Prognostics & MLOps Suite

An engineering-focused predictive maintenance (PdM) dashboard built to monitor, benchmark, and evaluate multi-generation machine learning and deep learning pipelines against simulated, highly dynamic run-to-failure jet engine telemetry.

Live Web Application (v4.0 Benchmarking, PINN deployment): https://digital-hangar-turbofan-prognostics.streamlit.app/
Live Web Application (v5.0 EnvironmentÁware Neural Network): https://pinn-degradation-model.streamlit.app/

Data Source and Provenance

Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

Flight Regimes: Highly dynamic operational envelopes encompassing up to 6 distinct flight phases (Takeoff, Climb, Cruise, Descent, Hold, Approach).

Fault Modes: Single (HPC Degradation) and Multi-Fault (HPC & Fan Degradation) scenarios.

Validation Target: Continuous Remaining Useful Life (RUL) trajectory evaluation.

Core Innovations (v4.0 & v5.0 Architecture)

The architecture has transitioned from legacy tree-based machine learning to advanced, physics-informed deep learning to handle extreme environmental volatility:

Physics-Informed Neural Networks (PINN): Integrates custom thermodynamic loss functions that heavily penalize the neural network during backpropagation for predicting physically impossible scenarios (e.g., an increasing RUL).

3D Chronological Tensors: Transitions from flat 2D tabular data to 50-flight 3D sliding windows, granting the neural network deep chronological "memory" of the degradation trajectory.

Condition-Based Normalization (CBN): Utilizes K-Means clustering to autonomously categorize operational spaces, scaling thermodynamic sensor data strictly within its specific operational cluster in an attempt to mathematically flatten environmental noise.

The Model Matrix: Evolution of Predictive Architectures

The suite tracks the evolutionary progression of prognostic modeling from pure statistics to physics-guided deep learning:

Phase I: Statistical Baselines

Random Forest Regressor: A foundational statistical baseline (100 Deep Trees). Evaluates all available high-noise data to identify statistical correlations. Susceptible to overfitting and highly jagged predictions.

Basic Gradient Boosting: Utilizes sequential error correction on raw sensor streams to smooth out predictions, but accuracy remains fundamentally constrained by un-smoothed sensor noise.

Phase II: Feature-Engineered Machine Learning

XGBoost (Feature Selection): A refined ensemble leveraging 5-cycle rolling averages to isolate degradation trends, eliminating 18 high-noise sensor streams.

XGBoost (Hyperparameter Optimized): The empirical limit of purely statistical approaches on this dataset, utilizing constrained tree depth and minimized learning rates to prevent anomaly memorization.

Phase III: Physics-Informed Machine Learning

Physics-Informed XGBoost (v2.0): Abandons raw statistics by converting telemetry into pure thermodynamic parameters (Isentropic Efficiency $\eta$ and EGT Margin) via the Brayton Cycle prior to ingestion, achieving superior accuracy with an ultra-lightweight edge profile.

Phase IV: Deep Learning & The Altitude Illusion

CNN-PINN Ensemble (v4.0): Introduces 3D chronological sliding windows and custom physics-loss functions to achieve State-of-the-Art accuracy in static, sea-level environments (FD001/FD003). Employs 5 distinct Convolutional-LSTMs averaged in real-time to successfully predict RUL trajectories.

Universal CBN Pipeline (v5.0): Integrates Condition-Based Normalization and Unsupervised K-Means clustering to force a single deep learning architecture to generalize across highly complex, dynamic flight envelopes.

Architectural Limitations: While v5.0 successfully categorizes the 6 distinct flight regimes, relying on a single "jack-of-all-trades" model exposes structural limitations. Rapid transitions between flight regimes within the 50-flight sliding window cause non-monotonic prediction swings. This dashboard explicitly demonstrates the boundary limitations of single-model architectures when applied to highly dynamic operational environments.

How To Run Locally

Clone the repository.

Install dependencies: pip install -r requirements.txt

Launch the v5.0 MLOps suite: streamlit run <app_name>.py
