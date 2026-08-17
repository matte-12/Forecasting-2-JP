# LightTimesNet — Efficient 2D Time-Series Forecasting
LightTimesNet is a lightweight redesign of TimesNet for efficient
multi-horizon time-series forecasting.
The project investigates whether the dynamic FFT-based period
extraction used by TimesNet can be replaced by domain-knowledge-driven
static spatial patching, reducing computational overhead while
maintaining competitive forecasting accuracy.
## Key Idea
TimesNet dynamically extracts dominant periods using FFT and reshapes
1D time series into 2D representations.
LightTimesNet replaces this process with a deterministic
[frequency, period] reshape based on known periodic structure.
This enables:
- fully vectorized tensor construction
- reduced period-extraction overhead
- simpler hardware execution
- lightweight spatial backbones
- improved accuracy–efficiency trade-offs
## Models
| Model | Representation |
|---|---|
| DLinear | 1D |
| CausalTCN | 1D |
| TimesNet | Dynamic 2D |
| Fixed-Period TimesNet | Static 2D |
| LightTimesNet | Static 2D |
| LightTimesNet + Depthwise | Static 2D |
## Datasets
Experiments use standard forecasting benchmarks:
| Dataset | Frequency | Horizons |
|---|---|---|
| ETTh1 | 1-hour | 24, 48, 96 |
| ETTm1 | 15-minute | 24, 48, 96 |
| Electricity | 1-hour | 24, 48, 96 |
## Results

The experiments demonstrate that deterministic period folding can
substantially reduce the computational cost of TimesNet while retaining
competitive forecasting performance.

| Comparison | Accuracy | Parameters | Training |
|---|---:|---:|---:|
| ETTh1 H=24 — DLinear | MSE 0.337 | 4.7K | 4.68 s |
| ETTh1 H=24 — LightTimesNet-DW | **MSE 0.326** | **4.2K** | 10.15 s |
| Electricity H=48 — TimesNet | **MSE 0.293** | 169K | 51.9 s |
| Electricity H=48 — LightTimesNet-Group | MSE 0.302 | **28.9K (-83%)** | **20.7 s (-60%)** |

The ETTh1 experiment shows that LightTimesNet-DW can slightly outperform
a strong lightweight 1D baseline while using fewer parameters.

On Electricity, the Group-based LightTimesNet variant reduces the model
size by approximately 83% and training time by 60% compared with
TimesNet, at the cost of only a 3.1% increase in MSE. Demystifying_2D_Time_Series_Forecasting__A_Targeted_Redesign_of_TimesNet_for_High_Frequency_Data.pdf

The optimal lightweight backbone is dataset-dependent: Depthwise
convolution performs best on ETTh1, whereas Group convolution provides
a stronger accuracy–complexity compromise on Electricity. Demystifying_2D_Time_Series_Forecasting__A_Targeted_Redesign_of_TimesNet_for_High_Frequency_Data.pdf
## Analysis
The repository includes experiments on:
- dynamic vs. fixed-period modeling
- 1D vs. 2D temporal representations
- top-k period selection
- number of TimesBlocks
- alternative spatial backbones
- depthwise convolutions
- parameter count and computational cost
- accuracy–efficiency Pareto analysis
## Getting Started
### Requirements
- Python 3.9+
- PyTorch
- NumPy
- Pandas
- PyYAML
- Matplotlib
### Installation
```bash
pip install -r requirements.txt
```

Run an Experiment
```
python run_pipeline.py --config configs/etth1_24.yaml
```
Other experiment configurations are available under configs/.

Repository Structure
```
.
├── configs/              # Experiment configurations
├── src/
│   ├── models_1d.py      # 1D baselines
│   ├── models_2d.py      # TimesNet and 2D models
│   ├── models_light.py   # LightTimesNet
│   ├── timesnet_original.py
│   ├── fixed_period_inception.py
│   └── runners/          # Experiment runners
├── src/standalone_files/ # Analysis and evaluation scripts
├── run_pipeline.py
├── requirements.txt
└── ...
```
Technical Focus

Time-Series Forecasting · TimesNet · PyTorch · FFT · 1D/2D
Representations · Spatial Patching · CNNs · Depthwise Convolutions ·
Ablation Studies · Pareto Analysis · Model Efficiency

Reference

Based on:

Wu et al., “TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis.”