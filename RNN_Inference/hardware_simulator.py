"""
Hardware-level simulation modules for ReRAM-based RNN accelerator
Implements circuit-level details from the paper
"""

import numpy as np
import math

class ReRAMCrossbar:
    """
    ReRAM Crossbar Array Simulator
    Based on Section IV-B and Fig. 5 of the paper
    """
    def __init__(self, rows=128, cols=128, ron=1e6, roff=1e8, 
                 read_voltage=0.5, write_voltage=2.0):
        self.rows = rows
        self.cols = cols
        self.ron = ron  # ON resistance (1 MΩ)
        self.roff = roff  # OFF resistance (100 MΩ)
        self.read_voltage = read_voltage
        self.write_voltage = write_voltage
        
        # Initialize conductance matrix (storing weights)
        self.conductance = np.zeros((rows, cols))
        
        # Device parameters from Table I (ITR model)
        self.gap_min = 0.2e-9  # 0.2 nm
        self.gap_max = 1.7e-9  # 1.7 nm
        self.gap_init = 1.0e-9  # Initial gap
        
    def program_weights(self, weights, bits_per_cell=2):
        """
        Program weight matrix into ReRAM cells
        Section IV-A programming scheme
        """
        # Normalize weights to conductance range
        w_min, w_max = weights.min(), weights.max()
        
        # Map to conductance levels (2^bits_per_cell levels)
        levels = 2 ** bits_per_cell
        conductance_min = 1.0 / self.roff
        conductance_max = 1.0 / self.ron
        
        # Quantize and map weights
        weights_normalized = (weights - w_min) / (w_max - w_min)
        weights_quantized = np.round(weights_normalized * (levels - 1))
        
        self.conductance = (conductance_min + 
                           weights_quantized * (conductance_max - conductance_min) / (levels - 1))
        
        return weights_quantized
    
    def read_current(self, input_voltage):
        """
        Perform analog matrix-vector multiplication
        Based on Kirchhoff's law (Fig. 2c)
        """
        # Input should be row-wise voltage vector
        if len(input_voltage) != self.rows:
            raise ValueError(f"Input size mismatch: expected {self.rows}, got {len(input_voltage)}")
        
        # Current at each bitline: I = sum(V * G)
        output_current = np.dot(input_voltage, self.conductance)
        
        return output_current
    
    def add_read_noise(self, current, noise_std=0.15):
        """
        Add Gaussian read noise (Equation 10)
        Section V-D device variation
        """
        noise = np.random.normal(0, noise_std, current.shape)
        noisy_current = current * (1 + noise)
        return noisy_current
    
    def calculate_read_energy(self, num_reads=1):
        """
        Calculate read energy based on device parameters
        """
        # Energy per read = V^2 / R * t
        energy_per_cell = (self.read_voltage ** 2 / self.ron) * 5e-9  # 5ns read time
        total_energy = energy_per_cell * self.rows * self.cols * num_reads
        return total_energy
    
    def calculate_write_energy(self, num_writes=1):
        """
        Calculate write energy
        Table III: 50ns write pulse
        """
        energy_per_cell = (self.write_voltage ** 2 / self.ron) * 50e-9
        total_energy = energy_per_cell * num_writes
        return total_energy

class SpecialFunctionUnit:
    """
    SFU for nonlinear activation functions
    Section IV-C and Fig. 11
    """
    def __init__(self, method='chebyshev', num_intervals=10):
        self.method = method
        self.num_intervals = num_intervals
        self.coefficients = {}
        
        # Pre-compute Chebyshev coefficients for common functions
        self._precompute_coefficients()
        
    def _precompute_coefficients(self):
        """
        Compute Chebyshev polynomial coefficients
        For sigmoid and tanh functions
        """
        # Simplified coefficients (in practice, computed using Chebyshev expansion)
        self.coefficients['sigmoid'] = self._chebyshev_coeffs_sigmoid()
        self.coefficients['tanh'] = self._chebyshev_coeffs_tanh()
    
    def _chebyshev_coeffs_sigmoid(self, n=10):
        """Generate Chebyshev coefficients for sigmoid approximation"""
        # Simplified: using polynomial approximation
        # In real implementation, use scipy.interpolate or numerical methods
        return np.array([0.5, 0.25, -0.02, 0.001])  # Placeholder
    
    def _chebyshev_coeffs_tanh(self, n=10):
        """Generate Chebyshev coefficients for tanh approximation"""
        return np.array([0.0, 1.0, 0.0, -0.33])  # Placeholder
    
    def sigmoid(self, x):
        """Approximate sigmoid using Chebyshev polynomial"""
        if self.method == 'exact':
            return 1 / (1 + np.exp(-x))
        else:
            # Chebyshev approximation
            coeffs = self.coefficients['sigmoid']
            result = np.zeros_like(x)
            for i, c in enumerate(coeffs):
                result += c * (x ** i)
            return np.clip(result, 0, 1)
    
    def tanh(self, x):
        """Approximate tanh using Chebyshev polynomial"""
        if self.method == 'exact':
            return np.tanh(x)
        else:
            coeffs = self.coefficients['tanh']
            result = np.zeros_like(x)
            for i, c in enumerate(coeffs):
                result += c * (x ** i)
            return np.clip(result, -1, 1)
    
    def calculate_energy(self, num_operations):
        """
        SFU energy consumption (Table II)
        15.8 mW power consumption
        """
        # Assume 1ns per operation (simplified)
        energy = 15.8e-3 * num_operations * 1e-9
        return energy
    
    def calculate_latency(self, num_operations):
        """
        SFU latency (Table III)
        Approximately 1 ns per operation
        """
        return num_operations * 1e-9

class MultiplierArray:
    """
    Element-wise multiplier array for LSTM gates
    Section III-B-3
    """
    def __init__(self, width=128):
        self.width = width
        self.power = 6.8e-3  # 6.8 mW from Table II
    
    def multiply(self, a, b):
        """Element-wise multiplication"""
        return a * b
    
    def calculate_energy(self, num_multiplications):
        """
        Calculate energy for element-wise multiplications
        """
        # Assume 1ns per multiplication
        energy = self.power * num_multiplications * 1e-9
        return energy

class ADC:
    """
    8-bit SAR ADC
    Section IV-C-3
    """
    def __init__(self, bits=8, sampling_rate=1.2e9):
        self.bits = bits
        self.sampling_rate = sampling_rate
        self.power = 3.1e-3  # 3.1 mW from Table II
        self.latency = 0.83e-9  # 0.83 ns from Table III
        self.area = 0.0015  # 0.0015 mm^2
        
    def convert(self, analog_current):
        """
        Convert analog current to digital value
        """
        # Quantization levels
        levels = 2 ** self.bits
        
        # Normalize and quantize
        max_current = 1e-3  # Assume max 1mA
        normalized = np.clip(analog_current / max_current, 0, 1)
        digital = np.round(normalized * (levels - 1)).astype(int)
        
        return digital
    
    def calculate_energy(self, num_conversions):
        """ADC energy per conversion"""
        return self.power * num_conversions * self.latency

class DAC:
    """
    1-bit DAC for wordline drivers
    Section IV-C-2
    """
    def __init__(self, bits=1):
        self.bits = bits
        self.latency = 0.1e-9  # 0.1 ns
        
    def convert(self, digital_value):
        """Convert digital to analog voltage"""
        # For 1-bit: 0 -> 0V, 1 -> Vread
        if self.bits == 1:
            return 0.5 if digital_value else 0.0
        else:
            # Multi-bit DAC
            levels = 2 ** self.bits
            return digital_value / levels

class LocalBuffer:
    """
    eDRAM-based local buffer
    Section IV-C-5
    """
    def __init__(self, size_kb=64):
        self.size_kb = size_kb
        self.power = 2.2e-3  # 2.2 mW from Table II
        self.data = {}
        
    def read(self, address):
        return self.data.get(address, 0)
    
    def write(self, address, value):
        self.data[address] = value
    
    def calculate_energy(self, num_accesses):
        """Buffer access energy"""
        # Assume 1ns per access
        return self.power * num_accesses * 1e-9

class PipelineController:
    """
    3-stage pipeline controller
    Section III-C and Fig. 7
    """
    def __init__(self):
        self.stages = ['matmul', 'sfu', 'multiplier']
        self.stage_latencies = {
            'matmul': 16e-9,  # 16 cycles * 1ns
            'sfu': 5e-9,      # 5 activation functions
            'multiplier': 3e-9  # 3 element-wise ops
        }
    
    def calculate_pipeline_throughput(self, sequence_length):
        """
        Calculate throughput with 3-stage pipeline
        """
        # Pipeline allows parallel processing
        # Effective cycle time is max of stage times
        max_stage_time = max(self.stage_latencies.values())
        
        # Total time = initialization + (seq_length * cycle_time)
        total_time = max_stage_time * 2 + sequence_length * max_stage_time
        
        return sequence_length / total_time, total_time

class IntegratedSystem:
    """
    Complete integrated system simulator
    Combines all hardware components
    """
    def __init__(self, num_arrays=128, array_size=128):
        self.num_arrays = num_arrays
        self.array_size = array_size
        
        # Initialize components
        self.crossbars = [ReRAMCrossbar(array_size, array_size) 
                         for _ in range(num_arrays)]
        self.sfu = SpecialFunctionUnit()
        self.multiplier = MultiplierArray(array_size)
        self.adc = ADC()
        self.dac = DAC()
        self.buffer = LocalBuffer()
        self.pipeline = PipelineController()
        
    def simulate_lstm_step(self, input_vec, hidden_state, weights_i, weights_f, 
                          weights_o, weights_g):
        """
        Simulate one LSTM time step
        Implements equations (3)-(8) from paper
        """
        # Concatenate input and hidden state
        concat_input = np.concatenate([input_vec, hidden_state])
        
        # Stage 1: Matrix-vector multiplications in ReRAM crossbars
        # Input gate
        i_matmul = self.crossbars[0].read_current(concat_input @ weights_i)
        # Forget gate  
        f_matmul = self.crossbars[1].read_current(concat_input @ weights_f)
        # Output gate
        o_matmul = self.crossbars[2].read_current(concat_input @ weights_o)
        # Cell gate
        g_matmul = self.crossbars[3].read_current(concat_input @ weights_g)
        
        # ADC conversion
        i_digital = self.adc.convert(i_matmul)
        f_digital = self.adc.convert(f_matmul)
        o_digital = self.adc.convert(o_matmul)
        g_digital = self.adc.convert(g_matmul)
        
        # Stage 2: Activation functions in SFU
        i_gate = self.sfu.sigmoid(i_digital)
        f_gate = self.sfu.sigmoid(f_digital)
        o_gate = self.sfu.sigmoid(o_digital)
        g_gate = self.sfu.tanh(g_digital)
        
        # Stage 3: Element-wise multiplications
        # (Simplified - actual cell state update)
        c_new = f_gate * 0.5 + i_gate * g_gate  # Simplified
        h_new = o_gate * self.sfu.tanh(c_new)
        
        return h_new, c_new
    
    def calculate_total_energy(self, sequence_length):
        """Calculate total system energy for sequence"""
        # Crossbar energy
        crossbar_energy = sum([cb.calculate_read_energy(sequence_length * 16) 
                              for cb in self.crossbars[:4]])
        
        # SFU energy (5 activations per step)
        sfu_energy = self.sfu.calculate_energy(sequence_length * 5)
        
        # Multiplier energy (3 multiplications per step)
        mult_energy = self.multiplier.calculate_energy(sequence_length * 3)
        
        # ADC energy
        adc_energy = self.adc.calculate_energy(sequence_length * 4 * 16)
        
        # Buffer energy
        buffer_energy = self.buffer.calculate_energy(sequence_length * 10)
        
        return {
            'total': crossbar_energy + sfu_energy + mult_energy + adc_energy + buffer_energy,
            'crossbar': crossbar_energy,
            'sfu': sfu_energy,
            'multiplier': mult_energy,
            'adc': adc_energy,
            'buffer': buffer_energy
        }

# Example usage
if __name__ == "__main__":
    print("Hardware Simulator Module Test")
    print("=" * 50)
    
    # Test ReRAM crossbar
    crossbar = ReRAMCrossbar(rows=8, cols=8)
    weights = np.random.randn(8, 8)
    crossbar.program_weights(weights)
    
    input_v = np.random.rand(8)
    output_i = crossbar.read_current(input_v)
    print(f"Crossbar output current: {output_i[:4]}...")
    
    # Test integrated system
    system = IntegratedSystem(num_arrays=4, array_size=128)
    energy = system.calculate_total_energy(sequence_length=100)
    print(f"\nSystem energy for 100 timesteps:")
    print(f"  Total: {energy['total']*1e6:.2f} µJ")
    print(f"  Crossbar: {energy['crossbar']*1e6:.2f} µJ")
    print(f"  SFU: {energy['sfu']*1e6:.2f} µJ")
    print(f"  ADC: {energy['adc']*1e6:.2f} µJ")