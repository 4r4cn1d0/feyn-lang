import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
import random

def monte_carlo_pi_python(samples: int) -> Tuple[float, float, float]:
    """
    Calculate π using Monte Carlo integration in Python.
    Returns: (estimated_pi, error, execution_time)
    """
    start_time = time.time()
    
    # Count points inside the quarter circle
    inside_circle = 0
    total_points = samples
    
    for _ in range(total_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Check if point is inside quarter circle (x² + y² ≤ 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    # Calculate π estimate
    pi_estimate = 4.0 * inside_circle / total_points
    
    # Calculate error
    true_pi = np.pi
    error = abs(pi_estimate - true_pi)
    
    execution_time = time.time() - start_time
    
    return pi_estimate, error, execution_time

def neural_network_python(iterations: int) -> Tuple[float, float, List[float]]:
    """
    Simple neural network training in Python.
    Returns: (final_loss, execution_time, loss_history)
    """
    start_time = time.time()
    
    # Simple neural network: y = 2x + 1 + noise
    # Training data
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([3.1, 5.2, 7.1, 9.3, 11.2])  # 2x + 1 + noise
    
    # Initialize weights
    w = 0.0
    b = 0.0
    learning_rate = 0.01
    
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
    
    return final_loss, execution_time, loss_history

def run_python_benchmarks():
    """Run all Python benchmarks with visualization."""
    print("🐍 Python Benchmarks")
    print("=" * 50)
    
    # Monte Carlo Integration
    print("\n📊 Monte Carlo Integration (π calculation)")
    print("-" * 40)
    
    sample_sizes = [1000, 10000, 100000, 1000000]
    pi_results = []
    
    for samples in sample_sizes:
        pi_est, error, time_taken = monte_carlo_pi_python(samples)
        pi_results.append({
            'samples': samples,
            'estimate': pi_est,
            'error': error,
            'time': time_taken
        })
        print(f"Samples: {samples:,} | π ≈ {pi_est:.6f} | Error: {error:.6f} | Time: {time_taken:.4f}s")
    
    # Neural Network Training
    print("\n🧠 Neural Network Training")
    print("-" * 40)
    
    iterations = 1000
    final_loss, time_taken, loss_history = neural_network_python(iterations)
    print(f"Iterations: {iterations} | Final Loss: {final_loss:.6f} | Time: {time_taken:.4f}s")
    
    # Create visualizations
    create_python_plots(pi_results, loss_history)
    
    return pi_results, final_loss, time_taken, loss_history

def create_python_plots(pi_results, loss_history):
    """Create visualization plots for Python results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Monte Carlo π accuracy plot
    samples = [r['samples'] for r in pi_results]
    errors = [r['error'] for r in pi_results]
    times = [r['time'] for r in pi_results]
    
    ax1.loglog(samples, errors, 'bo-', linewidth=2, markersize=8, label='Python')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Error (|π_estimate - π_true|)')
    ax1.set_title('Monte Carlo π: Accuracy vs Samples')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Neural Network loss plot
    iterations = range(len(loss_history))
    ax2.plot(iterations, loss_history, 'b-', linewidth=2, label='Python')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Neural Network Training Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('benchmarks/python_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_python_benchmarks()
