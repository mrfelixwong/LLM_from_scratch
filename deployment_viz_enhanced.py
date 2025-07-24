# Enhanced deployment strategy visualization
import matplotlib.pyplot as plt
import numpy as np

def visualize_deployment_tradeoffs(benchmark_results):
    """
    Visualize deployment strategy trade-offs with enhanced formatting.
    
    Args:
        benchmark_results: Dict with strategy names as keys and metrics as values
    """
    strategies = list(benchmark_results.keys())
    latencies = [benchmark_results[s]['avg_latency_ms'] for s in strategies]
    throughputs = [benchmark_results[s]['throughput_req_per_sec'] for s in strategies]
    
    # Enhanced color scheme
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(strategies)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Latency comparison with improved styling
    bars1 = ax1.bar(strategies, latencies, color=colors, alpha=0.8, 
                    edgecolor='white', linewidth=2)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency by Deployment Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(latencies) * 1.15)
    
    # Add value labels on bars
    for bar, latency in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(latencies) * 0.02,
                f'{latency:.1f}ms', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Throughput comparison with improved styling
    bars2 = ax2.bar(strategies, throughputs, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=2)
    ax2.set_ylabel('Throughput (requests/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput by Deployment Strategy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(throughputs) * 1.15)
    
    # Add value labels on bars
    for bar, throughput in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(throughputs) * 0.02,
                f'{throughput:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    # Rotate x-axis labels if needed
    if any(len(s) > 8 for s in strategies):
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Enhanced insights with performance ratios
    best_latency_idx = np.argmin(latencies)
    best_throughput_idx = np.argmax(throughputs)
    
    print("\n🎯 DEPLOYMENT STRATEGY ANALYSIS:")
    print(f"• Best latency: {strategies[best_latency_idx]} ({latencies[best_latency_idx]:.1f}ms)")
    print(f"• Best throughput: {strategies[best_throughput_idx]} ({throughputs[best_throughput_idx]:.1f} req/s)")
    
    # Calculate efficiency ratios
    for i, strategy in enumerate(strategies):
        efficiency = throughputs[i] / latencies[i]  # req/s per ms
        print(f"• {strategy}: {efficiency:.2f} efficiency ratio (throughput/latency)")
    
    print("\n💡 PRODUCTION RECOMMENDATIONS:")
    if 'cached' in [s.lower() for s in strategies]:
        print("• Use caching for repeated queries (10-100x speedup)")
    if 'batched' in [s.lower() for s in strategies]:
        print("• Implement batching for high-load scenarios")
    if 'streaming' in [s.lower() for s in strategies]:
        print("• Enable streaming for better user experience")
    print("• Monitor memory usage and GPU utilization")
    print("• Consider hybrid approach: batching + caching + streaming")

# Example usage with sample data
if __name__ == "__main__":
    # Sample benchmark results
    sample_results = {
        'Single': {'avg_latency_ms': 150.0, 'throughput_req_per_sec': 6.7},
        'Batched': {'avg_latency_ms': 200.0, 'throughput_req_per_sec': 25.0},
        'Cached': {'avg_latency_ms': 15.0, 'throughput_req_per_sec': 66.7},
        'Streaming': {'avg_latency_ms': 180.0, 'throughput_req_per_sec': 5.6}
    }
    
    visualize_deployment_tradeoffs(sample_results)