#!/usr/bin/env python3
"""
Neural Network Training Benchmark
Compares Python, Haskell, and Feyn Language implementations
"""

import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NeuralNetworkBenchmark:
    def __init__(self):
        self.results = {
            'python': {},
            'haskell': {},
            'feyn': {}
        }
        
    def run_python_neural_network(self) -> Dict:
        """Run Python neural network benchmark."""
        print("Running Python neural network benchmark...")
        
        try:
            import sys
            sys.path.append('benchmarks')
            start_time = time.time()
            
            # Training data: y = 2x + 1 + noise
            X = np.array([1, 2, 3, 4, 5])
            y = np.array([3.1, 5.2, 7.1, 9.3, 11.2])
            
            # Initialize weights
            w = 0.0
            b = 0.0
            learning_rate = 0.01
            iterations = 1000
            
            loss_history = []
            
            for i in range(iterations):
                # Forward pass
                y_pred = w * X + b
                
                # Calculate loss (MSE)
                loss = np.mean((y_pred - y) ** 2)
                loss_history.append(loss)
                
                # Backward pass (manual gradients)
                dw = 2 * np.mean((y_pred - y) * X)
                db = 2 * np.mean(y_pred - y)
                
                # Update weights
                w -= learning_rate * dw
                b -= learning_rate * db
            
            execution_time = time.time() - start_time
            final_loss = loss_history[-1]
            
            print("Python neural network completed successfully")
            return {
                'iterations': iterations,
                'final_loss': final_loss,
                'time': execution_time,
                'loss_history': loss_history,
                'final_weights': (w, b)
            }
                
        except Exception as e:
            print(f"❌ Error running Python neural network: {e}")
            return {}
    
    def run_haskell_neural_network(self) -> Dict:
        """Run Haskell neural network benchmark and parse actual results."""
        print("Running Haskell neural network benchmark...")

        try:
            exe = 'benchmarks/monte_carlo_haskell.exe' if os.name == 'nt' else './benchmarks/monte_carlo_haskell'
            result = subprocess.run(
                [exe],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"❌ Haskell neural network failed: {result.stderr}")
                return {}

            # Prefer machine-readable line: "NN,<iters>,<final_loss>,<time>"
            final_loss, execution_time, iterations = None, None, 1000
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('NN,'):
                    try:
                        _, iters, loss, t = line.split(',')
                        iterations = int(iters)
                        final_loss = float(loss)
                        execution_time = float(t)
                    except Exception:
                        pass

            if final_loss is None or execution_time is None:
                print("❌ Failed to parse Haskell NN output")
                return {}

            print("Haskell neural network completed successfully")
            return {
                'iterations': 1000,
                'final_loss': final_loss,
                'time': execution_time,
                'loss_history': [],
            }

        except Exception as e:
            print(f"❌ Error running Haskell neural network: {e}")
            return {}
    
    def run_feyn_neural_network(self) -> Dict:
        """Run Feyn neural network GD via CLI and parse machine-readable output."""
        print("Running Feyn Language neural network benchmark...")

        try:
            trials = 5
            losses, times = [], []
            iters = 1000
            for _ in range(trials):
                r = subprocess.run(
                    ['cabal', 'run', 'feyn', '--', 'bench', 'nn', str(iters)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if r.returncode != 0:
                    print(f"❌ Feyn neural network failed: {r.stderr}")
                    continue
                # Expect: NN,<iters>,<final_loss>,<time>
                lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip().startswith('NN,')]
                if not lines:
                    continue
                _, iters_s, loss, t = lines[-1].split(',')
                iters = int(iters_s)
                losses.append(float(loss))
                times.append(float(t))
            import statistics as stats
            if not losses:
                return {}
            return {
                'iterations': iters,
                'final_loss': stats.mean(losses),
                'time': max(stats.mean(times), 1e-9),
                'loss_history': [],
            }

        except Exception as e:
            print(f"❌ Error running Feyn neural network: {e}")
            return {}
    
    def create_neural_network_visualizations(self):
        """Create separate visualizations for neural network benchmarks."""
        print("Creating neural network visualizations...")
        
        # 1. Training Loss Comparison
        plt.figure(figsize=(10, 6))
        for lang, color in [('python', 'blue'), ('haskell', 'green'), ('feyn', 'red')]:
            if lang in self.results and self.results[lang]:
                data = self.results[lang]
                iterations = range(len(data['loss_history']))
                plt.plot(iterations, data['loss_history'], color=color, linewidth=2, label=lang.title())
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.title('Neural Network: Training Loss Convergence')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('benchmarks/images/neural_network_loss_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Final Loss Comparison
        plt.figure(figsize=(8, 6))
        languages = []
        final_losses = []
        colors = []
        
        for lang, color in [('python', 'blue'), ('haskell', 'green'), ('feyn', 'red')]:
            if lang in self.results and self.results[lang]:
                languages.append(lang.title())
                final_losses.append(self.results[lang]['final_loss'])
                colors.append(color)
        
        bars = plt.bar(languages, final_losses, color=colors, alpha=0.7)
        plt.ylabel('Final Loss (MSE)')
        plt.title('Neural Network: Final Loss Comparison')
        plt.yscale('log')
        
        # Add value labels on bars (simplified positioning)
        for i, (bar, loss) in enumerate(zip(bars, final_losses)):
            height = bar.get_height()
            plt.text(i, height * 1.1, f'{loss:.6f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.savefig('benchmarks/images/neural_network_final_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Execution Time Comparison
        plt.figure(figsize=(8, 6))
        execution_times = []
        for lang in ['python', 'haskell', 'feyn']:
            if lang in self.results and self.results[lang]:
                execution_times.append(max(self.results[lang]['time'], 1e-6))  # Ensure minimum value
        
        bars = plt.bar(languages, execution_times, color=colors, alpha=0.7)
        plt.ylabel('Execution Time (seconds)')
        plt.title('Neural Network: Performance Comparison')
        
        for i, (bar, time_val) in enumerate(zip(bars, execution_times)):
            height = bar.get_height()
            plt.text(i, height + 0.001,
                    f'{time_val:.6f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.savefig('benchmarks/images/neural_network_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Speedup Analysis
        if 'python' in self.results and 'feyn' in self.results:
            plt.figure(figsize=(8, 6))
            python_time = self.results['python']['time']
            haskell_time = self.results['haskell']['time'] if 'haskell' in self.results else python_time
            feyn_time = self.results['feyn']['time']
            
            speedups = [python_time / feyn_time, haskell_time / feyn_time]
            speedup_languages = ['Python', 'Haskell']
            
            bars = plt.bar(speedup_languages, speedups, color=['gold', 'orange'], alpha=0.7)
            plt.ylabel('Speedup vs Feyn Language')
            plt.title('Neural Network: Feyn Language Speedup')
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
            
            for i, (bar, speedup) in enumerate(zip(bars, speedups)):
                height = bar.get_height()
                plt.text(i, height + 0.1,
                        f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
            
            plt.legend()
            plt.savefig('benchmarks/images/neural_network_speedup.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Neural network visualizations saved:")
        print("  - benchmarks/images/neural_network_loss_convergence.png")
        print("  - benchmarks/images/neural_network_final_loss.png") 
        print("  - benchmarks/images/neural_network_performance.png")
        print("  - benchmarks/images/neural_network_speedup.png")
    
    def run_neural_network_benchmarks(self):
        """Run all neural network benchmarks and create visualizations."""
        print("🚀 Starting Neural Network Training Benchmark Suite")
        print("=" * 60)
        
        # Create benchmarks directory if it doesn't exist
        os.makedirs('benchmarks', exist_ok=True)
        
        # Run all benchmarks
        self.results['python'] = self.run_python_neural_network()
        self.results['haskell'] = self.run_haskell_neural_network()
        self.results['feyn'] = self.run_feyn_neural_network()
        
        # Print summary
        print("\n📊 Neural Network Benchmark Summary")
        print("=" * 60)
        
        for lang in ['python', 'haskell', 'feyn']:
            if lang in self.results and self.results[lang]:
                data = self.results[lang]
                print(f"\n{lang.upper()} RESULTS:")
                print(f"  Final Loss: {data['final_loss']:.6f}")
                print(f"  Time: {data['time']:.4f}s")
                if 'final_weights' in data:
                    w, b = data['final_weights']
                    print(f"  Final Weights: w={w:.4f}, b={b:.4f}")
        
        # Create visualizations
        self.create_neural_network_visualizations()
        
        print("\n🎉 Neural network benchmark suite completed successfully!")
        print("📈 Check 'benchmarks/neural_network_comparison.png' for visualizations")

if __name__ == "__main__":
    benchmark = NeuralNetworkBenchmark()
    benchmark.run_neural_network_benchmarks()
