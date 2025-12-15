import torch
import torch.nn as nn
import numpy as np
import os
import sys

class ReRAM_LSTM(nn.Module):
    """
    ReRAM-based LSTM implementation matching the paper specifications
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ReRAM_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with specifications from paper
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Quantization parameters (8-bit as per paper Section V-C)
        self.weight_bits = 8
        self.activation_bits = 8
        
    def quantize_weights(self):
        """Quantize weights to 8-bit as described in paper"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Symmetric quantization
                max_val = torch.max(torch.abs(param.data))
                scale = (2 ** (self.weight_bits - 1) - 1) / max_val
                param.data = torch.round(param.data * scale) / scale
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out

class ReRAM_RNN_Simulator:
    """
    Simulator for ReRAM-based RNN acceleration
    Based on paper specifications
    """
    def __init__(self, config_file='NeuroSIM/RNN_LSTM_config.cfg'):
        self.config = self.load_config(config_file)
        self.setup_hardware_params()
        
    def load_config(self, config_file):
        """Load configuration from file"""
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    if ':' in line and not line.startswith('-'):
                        key, value = line.strip().split(':', 1)
                        config[key.strip()] = value.strip()
        return config
    
    def setup_hardware_params(self):
        """Setup hardware parameters based on paper specifications"""
        # Table II - Power and Area parameters
        self.crossbar_power_per_array = 4.8e-3  # 4.8 mW per array
        self.sfu_power = 15.8e-3  # 15.8 mW
        self.multiplier_power = 6.8e-3  # 6.8 mW
        self.adc_power = 3.1e-3  # 3.1 mW per ADC
        self.buffer_power = 2.2e-3  # 2.2 mW
        
        # Crossbar specifications
        self.array_size = 128  # 128x128 as per paper
        self.num_arrays = 128  # Total arrays in system
        self.device_resistance = 1e6  # 1 MΩ
        self.read_noise_std = 0.15  # Standard deviation < 0.2
        
        # Timing parameters (Table III)
        self.read_latency = 5e-9  # 5 ns
        self.write_latency = 50e-9  # 50 ns
        self.adc_latency = 0.83e-9  # 0.83 ns
        self.dac_latency = 0.1e-9  # 0.1 ns
        
    def calculate_energy(self, model, input_data):
        """
        Calculate energy consumption for RNN inference
        Based on paper's energy model
        """
        # Count operations
        total_ops = 0
        matmul_ops = 0
        activation_ops = 0
        elementwise_ops = 0
        
        # For LSTM: 4 matrix multiplications per timestep
        # (input gate, forget gate, output gate, cell gate)
        seq_length = input_data.size(1)
        hidden_size = model.hidden_size
        input_size = model.input_size
        
        # Matrix-vector operations per timestep
        matmul_ops_per_step = 4 * (input_size + hidden_size) * hidden_size
        matmul_ops = matmul_ops_per_step * seq_length
        
        # Activation functions (sigmoid x3, tanh x2)
        activation_ops = 5 * hidden_size * seq_length
        
        # Element-wise multiplications (3 per timestep)
        elementwise_ops = 3 * hidden_size * seq_length
        
        total_ops = matmul_ops + activation_ops + elementwise_ops
        
        # Calculate energy based on operations
        # ReRAM crossbar energy (dominant component)
        cycles_for_matmul = (seq_length * 16)  # 16 cycles for 16-bit or 8 for 8-bit
        crossbar_energy = (self.crossbar_power_per_array * 
                          self.num_arrays * cycles_for_matmul * self.read_latency)
        
        # SFU energy
        sfu_cycles = seq_length * 5  # 5 activation functions per step
        sfu_energy = self.sfu_power * sfu_cycles * 1e-9  # Assume 1ns per operation
        
        # Multiplier energy
        mult_cycles = seq_length * 3
        mult_energy = self.multiplier_power * mult_cycles * 1e-9
        
        # ADC energy (one per array)
        adc_energy = self.adc_power * cycles_for_matmul * self.adc_latency
        
        total_energy = crossbar_energy + sfu_energy + mult_energy + adc_energy
        
        return {
            'total_energy_J': total_energy,
            'total_ops': total_ops,
            'matmul_ops': matmul_ops,
            'energy_per_op': total_energy / total_ops,
            'crossbar_energy': crossbar_energy,
            'sfu_energy': sfu_energy,
            'multiplier_energy': mult_energy,
            'adc_energy': adc_energy
        }
    
    def calculate_throughput(self, model, input_data):
        """
        Calculate system throughput
        Based on paper's pipeline design (Fig. 7)
        """
        seq_length = input_data.size(1)
        
        # Pipeline stages: matmul -> SFU -> multiplier
        # With 3-stage pipeline, effective cycle time is max of stage times
        
        matmul_cycles = 16  # 16 cycles for 16-bit input (or 8 for 8-bit)
        sfu_cycles = 5  # 5 activation functions
        mult_cycles = 3  # 3 element-wise multiplications
        
        cycle_time = max(matmul_cycles, sfu_cycles, mult_cycles) * 1e-9  # Convert to seconds
        
        # Total time with pipeline
        total_time = seq_length * cycle_time
        
        # Operations per second
        total_ops = self.calculate_energy(model, input_data)['total_ops']
        throughput = total_ops / total_time / 1e9  # GOPS
        
        return {
            'throughput_GOPS': throughput,
            'total_time_s': total_time,
            'cycle_time_ns': cycle_time * 1e9
        }
    
    def simulate_with_noise(self, model, input_data, noise_std=0.15):
        """
        Simulate device variation effects
        Section V-D of paper
        """
        model.eval()
        with torch.no_grad():
            # Clean inference
            clean_output = model(input_data)
            
            # Add Gaussian noise to weights (Equation 10)
            noisy_model = type(model)(model.input_size, model.hidden_size)
            noisy_model.load_state_dict(model.state_dict())
            
            for name, param in noisy_model.named_parameters():
                if 'weight' in name:
                    noise = torch.randn_like(param) * noise_std
                    param.data = param.data * (1 + noise)
            
            noisy_output = noisy_model(input_data)
            
            # Calculate error
            mse = torch.mean((clean_output - noisy_output) ** 2)
            relative_error = mse / torch.mean(clean_output ** 2)
            
        return {
            'clean_output': clean_output,
            'noisy_output': noisy_output,
            'mse': mse.item(),
            'relative_error': relative_error.item()
        }

def main():
    print("=" * 80)
    print("ReRAM-based RNN Accelerator Simulation")
    print("Based on: IEEE TVLSI 2018 Paper")
    print("=" * 80)
    
    # Model configuration (matching paper's benchmark - Table IV)
    input_size = 28
    hidden_size = 128
    seq_length = 100
    batch_size = 1
    
    # Create model
    print("\n[1] Creating LSTM model...")
    model = ReRAM_LSTM(input_size, hidden_size)
    
    # Quantize weights to 8-bit
    print("[2] Quantizing weights to 8-bit...")
    model.quantize_weights()
    
    # Generate sample input
    print("[3] Generating sample input data...")
    input_data = torch.randn(batch_size, seq_length, input_size)
    
    # Initialize simulator
    print("[4] Initializing ReRAM simulator...")
    simulator = ReRAM_RNN_Simulator()
    
    # Calculate energy
    print("\n[5] Calculating energy consumption...")
    energy_results = simulator.calculate_energy(model, input_data)
    print(f"  Total Energy: {energy_results['total_energy_J']*1e6:.2f} µJ")
    print(f"  Total Operations: {energy_results['total_ops']/1e9:.2f} GOP")
    print(f"  Energy Efficiency: {1/(energy_results['energy_per_op']*1e12):.2f} TOPS/W")
    print(f"  Breakdown:")
    print(f"    - Crossbar: {energy_results['crossbar_energy']*1e6:.2f} µJ")
    print(f"    - SFU: {energy_results['sfu_energy']*1e6:.2f} µJ")
    print(f"    - Multiplier: {energy_results['multiplier_energy']*1e6:.2f} µJ")
    print(f"    - ADC: {energy_results['adc_energy']*1e6:.2f} µJ")
    
    # Calculate throughput
    print("\n[6] Calculating throughput...")
    throughput_results = simulator.calculate_throughput(model, input_data)
    print(f"  Throughput: {throughput_results['throughput_GOPS']:.2f} GOPS")
    print(f"  Total Time: {throughput_results['total_time_s']*1e6:.2f} µs")
    
    # Simulate with noise
    print("\n[7] Simulating device variation (σ=0.15)...")
    noise_results = simulator.simulate_with_noise(model, input_data, noise_std=0.15)
    print(f"  Mean Squared Error: {noise_results['mse']:.6f}")
    print(f"  Relative Error: {noise_results['relative_error']*100:.2f}%")
    
    # Compare with paper results (Table VI)
    print("\n" + "=" * 80)
    print("Comparison with Paper Results:")
    print("=" * 80)
    print(f"Paper reported: 177 GOPS/W")
    print(f"Our simulation: {1/(energy_results['energy_per_op']*1e9):.2f} GOPS/W")
    print("\nNote: Differences may arise from:")
    print("  - Simplified analog circuit models")
    print("  - Different device parameters")
    print("  - Abstracted pipeline implementation")
    print("=" * 80)

if __name__ == "__main__":
    main()