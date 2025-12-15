#!/bin/bash

# ReRAM-based RNN Accelerator Setup Script
# This script creates the correct project structure

echo "=================================================="
echo "Setting up ReRAM RNN Accelerator Implementation"
echo "=================================================="

# Navigate to DNN_NeuroSim_V2.1 directory
cd ~/DNN_NeuroSim_V2.1

# Create new RNN_Inference directory
echo "[1] Creating RNN_Inference directory..."
mkdir -p RNN_Inference/results
mkdir -p RNN_Inference/configs

cd RNN_Inference

# Create main Python files
echo "[2] Creating Python implementation files..."

# Create __init__.py for package
touch __init__.py

# These files will be created manually with the code provided
echo "   - rnn_inference.py (create manually)"
echo "   - hardware_simulator.py (create manually)"
echo "   - run_experiments.py (create manually)"

# Create config file
echo "[3] Creating configuration file..."
cat > configs/rnn_config.cfg << 'EOF'
# ReRAM-based RNN Accelerator Configuration
# Based on IEEE TVLSI 2018 Paper

[Hardware]
process_node = 28
temperature_k = 300
device_type = RRAM

[ReRAM_Device]
ron_ohm = 1e6
roff_ohm = 1e8
read_voltage_v = 0.5
write_voltage_v = 2.0
read_pulse_ns = 5
write_pulse_ns = 50
bits_per_cell = 2
read_noise_std = 0.15

[Crossbar]
array_size = 128
num_arrays = 128
cell_area_f2 = 10

[Peripherals]
adc_bits = 8
adc_type = SAR
adc_power_mw = 3.1
adc_latency_ns = 0.83
dac_bits = 1
dac_latency_ns = 0.1

[SFU]
method = chebyshev
num_intervals = 10
power_mw = 15.8

[Multiplier]
width = 128
power_mw = 6.8

[Buffer]
type = eDRAM
size_kb = 64
power_mw = 2.2

[Pipeline]
stages = 3
enable = true

[RNN]
type = LSTM
hidden_size = 128
input_size = 28
num_layers = 1
sequence_length = 100

[Optimization]
clock_freq_hz = 1e9
batch_size = 1
precision_bits = 8
EOF

# Create README
echo "[4] Creating README..."
cat > README.md << 'EOF'
# ReRAM-Based RNN Accelerator

Implementation of "ReRAM-Based Processing-in-Memory Architecture for Recurrent Neural Network Acceleration" (IEEE TVLSI 2018)

## Project Structure

```
RNN_Inference/
├── rnn_inference.py          # Main inference simulator
├── hardware_simulator.py     # Hardware component models
├── run_experiments.py        # Experimental suite
├── configs/
│   └── rnn_config.cfg       # Configuration file
├── results/                  # Output plots and data
└── README.md                # This file
```

## Setup

```bash
# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy pandas matplotlib scipy

# Run experiments
python3 run_experiments.py
```

## Files to Create

You need to manually create these files with the provided code:

1. **rnn_inference.py** - Main RNN/LSTM simulator
2. **hardware_simulator.py** - ReRAM crossbar, SFU, ADC models  
3. **run_experiments.py** - Complete experimental suite

Copy the code from the artifacts provided by Claude.

## Quick Test

```bash
# Test hardware simulator
python3 hardware_simulator.py

# Run main inference
python3 rnn_inference.py

# Run all experiments
python3 run_experiments.py
```
EOF

# Create requirements.txt
echo "[5] Creating requirements.txt..."
cat > requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
results/*.png
*.log
.vscode/
.idea/
EOF

echo ""
echo "=================================================="
echo "✅ Directory structure created successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Navigate to: cd ~/DNN_NeuroSim_V2.1/RNN_Inference"
echo "2. Create the three main Python files:"
echo "   - rnn_inference.py"
echo "   - hardware_simulator.py"  
echo "   - run_experiments.py"
echo "3. Copy the code from Claude's artifacts"
echo "4. Run: python3 run_experiments.py"
echo ""
echo "Current location: $(pwd)"
echo "=================================================="

# List the created structure
echo ""
echo "Created structure:"
tree -L 2 . 2>/dev/null || find . -maxdepth 2 -print

cd ..