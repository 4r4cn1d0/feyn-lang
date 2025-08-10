#!/usr/bin/env python3
"""
Monte Carlo Integration Benchmark
Compares Python, Haskell, and Feyn Language implementations
"""

import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MonteCarloBenchmark:
    def __init__(self):
        self.results = {
            'python': {},
            'haskell': {},
            'feyn': {}
        }
        
    def run_python_monte_carlo(self) -> Dict:
        """Run Python Monte Carlo benchmark."""
        print("Running Python Monte Carlo benchmark...")
        
        try:
            import sys
            sys.path.append('benchmarks')
            start_time = time.time()
            import random
            import numpy as np
            sample_sizes = [1000, 10000, 100000, 1000000]
            times = []
            errors = []
            estimates = []
            for samples in sample_sizes:
                sample_start = time.time()
                inside_circle = 0
                for _ in range(samples):
                    x = random.uniform(0, 1)
                    y = random.uniform(0, 1)
                    if x*x + y*y <= 1:
                        inside_circle += 1
                pi_estimate = 4.0 * inside_circle / samples
                true_pi = np.pi
                error = abs(pi_estimate - true_pi)
                sample_time = time.time() - sample_start
                times.append(sample_time)
                errors.append(error)
                estimates.append(pi_estimate)
            print("Python Monte Carlo completed successfully")
            return {
                'samples': sample_sizes,
                'times': times,
                'errors': errors,
                'estimates': estimates
            }
                
        except Exception as e:
            print(f"❌ Error running Python Monte Carlo: {e}")
            return {}
    
    def run_haskell_monte_carlo(self) -> Dict:
        """Run Haskell Monte Carlo benchmark and parse actual results (averaged per sample)."""
        print("Running Haskell Monte Carlo benchmark...")

        try:
            compile_result = subprocess.run(
                ['ghc', '-O2', 'benchmarks/monte_carlo_haskell.hs', '-o', 'benchmarks/monte_carlo_haskell'],
                capture_output=True,
                text=True,
            )

            if compile_result.returncode != 0:
                print(f"❌ Haskell compilation failed: {compile_result.stderr}")
                return {}

            exe = 'benchmarks/monte_carlo_haskell.exe' if os.name == 'nt' else './benchmarks/monte_carlo_haskell'
            result = subprocess.run(
                [exe],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"❌ Haskell Monte Carlo failed: {result.stderr}")
                return {}

            print("Haskell Monte Carlo completed successfully")

            # Aggregate per-trial lines by sample size
            from collections import defaultdict
            import statistics as stats
            agg = defaultdict(lambda: {'pi': [], 'err': [], 't': []})
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('MC,'):
                    try:
                        _, s, pi_est, err, time_s = line.split(',')
                        s_i = int(s)
                        agg[s_i]['pi'].append(float(pi_est))
                        agg[s_i]['err'].append(float(err))
                        agg[s_i]['t'].append(float(time_s))
                    except Exception:
                        pass

            if not agg:
                print("❌ Failed to parse Haskell output")
                return {}

            samples = sorted(agg.keys())
            estimates = [stats.mean(agg[s]['pi']) for s in samples]
            errors = [stats.mean(agg[s]['err']) for s in samples]
            times = [max(stats.mean(agg[s]['t']), 1e-9) for s in samples]

            return {
                'samples': samples,
                'times': times,
                'errors': errors,
                'estimates': estimates,
            }

        except Exception as e:
            print(f"❌ Error running Haskell Monte Carlo: {e}")
            return {}
    
    def run_feyn_monte_carlo(self) -> Dict:
        """Run Feyn Monte Carlo by looping over sample sizes and parsing CLI output."""
        print("Running Feyn Language Monte Carlo benchmark...")

        try:
            sample_sizes = [1000, 10000, 100000, 1000000]
            samples, times, errors, estimates = [], [], [], []
            for n in sample_sizes:
                r = subprocess.run(
                    ['cabal', 'run', 'feyn', '--', 'bench', 'mc', str(n)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if r.returncode != 0:
                    print(f"❌ Feyn MC failed for {n}: {r.stderr}")
                    continue
                # Collect all MC lines, average time and values
                lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip().startswith('MC,')]
                if not lines:
                    continue
                ps, es, ts = [], [], []
                s_val = n
                for ln in lines:
                    _, s, pi_est, err, t = ln.split(',')
                    s_val = int(s)
                    ps.append(float(pi_est))
                    es.append(float(err))
                    ts.append(float(t))
                import statistics as stats
                samples.append(s_val)
                estimates.append(stats.mean(ps))
                errors.append(stats.mean(es))
                times.append(max(stats.mean(ts), 1e-9))

            if not samples:
                return {}

            return {
                'samples': samples,
                'times': times,
                'errors': errors,
                'estimates': estimates,
            }

        except Exception as e:
            print(f"❌ Error running Feyn Monte Carlo: {e}")
            return {}
    
    def create_monte_carlo_visualizations(self):
            """Create separate visualizations for Monte Carlo benchmarks."""
            print("Creating Monte Carlo visualizations...")
            
            plt.figure(figsize=(10, 6))
            for lang, color, marker in [('python', 'blue', 'o'), ('haskell', 'green', 's'), ('feyn', 'red', '^')]:
                if lang in self.results and self.results[lang]:
                    data = self.results[lang]
                    plt.loglog(data['samples'], data['times'], f'{marker}-', 
                               color=color, linewidth=2, markersize=8, label=lang.title())
            
            plt.xlabel('Number of Samples')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Monte Carlo π: Performance Comparison')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('benchmarks/images/monte_carlo_performance.png', dpi=300, bbox_inches='tight')
            # plt.show()
            
            plt.figure(figsize=(10, 6))
            for lang, color, marker in [('python', 'blue', 'o'), ('haskell', 'green', 's'), ('feyn', 'red', '^')]:
                if lang in self.results and self.results[lang]:
                    data = self.results[lang]
                    plt.loglog(data['samples'], data['errors'], f'{marker}-', 
                               color=color, linewidth=2, markersize=8, label=lang.title())
            
            plt.xlabel('Number of Samples')
            plt.ylabel('Error (|π_estimate - π_true|)')
            plt.title('Monte Carlo π: Accuracy Comparison')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('benchmarks/images/monte_carlo_accuracy.png', dpi=300, bbox_inches='tight')
            # plt.show()
            
            plt.figure(figsize=(10, 6))
            x_pos = np.arange(4)
            width = 0.25
            for i, (lang, color) in enumerate([('python', 'blue'), ('haskell', 'green'), ('feyn', 'red')]):
                if lang in self.results and self.results[lang]:
                    data = self.results[lang]
                    devs = [est - np.pi for est in data['estimates']]
                    devs = devs[:4]
                    plt.bar(x_pos + i*width, devs, width, label=lang.title(), color=color, alpha=0.7)
            
            plt.xlabel('Sample Size (1K, 10K, 100K, 1M)')
            plt.ylabel('Deviation from π')
            plt.title('Monte Carlo π: Estimate Accuracy')
            plt.xticks(x_pos + width, ['1K', '10K', '100K', '1M'])
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('benchmarks/images/monte_carlo_estimates.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            if 'python' in self.results and self.results.get('python'):
                plt.figure(figsize=(8, 6))
                eps = 1e-9
                python_time = max(self.results['python']['times'][-1], eps)  # 1M samples
                haskell_time = max((self.results.get('haskell', {}).get('times', [python_time])[-1]
                                    if self.results.get('haskell') and self.results['haskell'].get('times')
                                    else python_time), eps)
                feyn_time = max((self.results.get('feyn', {}).get('times', [python_time])[-1]
                                  if self.results.get('feyn') and self.results['feyn'].get('times')
                                  else python_time), eps)
                
                speedups = [python_time / max(feyn_time, 1e-12), haskell_time / max(feyn_time, 1e-12)]
                languages = ['Python', 'Haskell']
                bars = plt.bar(languages, speedups, color=['gold', 'orange'], alpha=0.7)
                plt.ylabel('Speedup vs Feyn Language')
                plt.title('Monte Carlo π: Feyn Language Speedup')
                plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
                for bar, speedup in zip(bars, speedups):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
                plt.legend()
                plt.tight_layout()
                plt.savefig('benchmarks/images/monte_carlo_speedup.png', dpi=300, bbox_inches='tight')
                # plt.show()
            
            print("Monte Carlo visualizations saved:")
            print("  - benchmarks/images/monte_carlo_performance.png")
            print("  - benchmarks/images/monte_carlo_accuracy.png")
            print("  - benchmarks/images/monte_carlo_estimates.png")
            print("  - benchmarks/images/monte_carlo_speedup.png")
    
    def run_monte_carlo_benchmarks(self):
        """Run all Monte Carlo benchmarks and create visualizations."""
        print("🚀 Starting Monte Carlo Integration Benchmark Suite")
        print("=" * 60)
        
        # Create benchmarks directory if it doesn't exist
        os.makedirs('benchmarks', exist_ok=True)
        
        # Run all benchmarks
        self.results['python'] = self.run_python_monte_carlo()
        self.results['haskell'] = self.run_haskell_monte_carlo()
        self.results['feyn'] = self.run_feyn_monte_carlo()
        
        # Print summary
        print("\n📊 Monte Carlo Benchmark Summary")
        print("=" * 60)
        
        for lang in ['python', 'haskell', 'feyn']:
            if lang in self.results and self.results[lang]:
                data = self.results[lang]
                print(f"\n{lang.upper()} RESULTS:")
                print(f"  π Estimate (1M samples): {data['estimates'][-1]:.6f}")
                print(f"  Error: {data['errors'][-1]:.6f}")
                print(f"  Time: {data['times'][-1]:.4f}s")
        
        # Create visualizations
        self.create_monte_carlo_visualizations()
        
        print("\n🎉 Monte Carlo benchmark suite completed successfully!")
        print("📈 Check 'benchmarks/monte_carlo_comparison.png' for visualizations")

if __name__ == "__main__":
    benchmark = MonteCarloBenchmark()
    benchmark.run_monte_carlo_benchmarks()
