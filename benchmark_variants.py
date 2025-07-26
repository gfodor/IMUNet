#!/usr/bin/env python3
"""
Benchmark script to compare different IMUNet optimization variants
"""

import os
import time
import numpy as np
import tensorflow as tf
from optimized_imunet import OptimizedIMUNet
import pandas as pd

def benchmark_inference(model, input_shape, num_runs=100, warmup_runs=10):
    """Benchmark model inference time"""
    
    # Create dummy input
    dummy_input = tf.random.normal([1] + list(input_shape))
    
    # Warm up
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input, training=False)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95)
    }

def estimate_mobile_performance(desktop_time_ms, model_params):
    """
    Estimate mobile performance based on desktop benchmarks
    Rough estimation based on typical mobile vs desktop performance ratios
    """
    
    # Typical mobile slowdown factors
    # These are rough estimates and will vary significantly by device
    mobile_factors = {
        'low_end': {'cpu': 8, 'gpu': 4},      # Budget Android phones
        'mid_range': {'cpu': 4, 'gpu': 2.5},  # Mid-range phones (Adreno 6xx)
        'high_end': {'cpu': 2, 'gpu': 1.5},   # Flagship phones (Adreno 7xx+)
    }
    
    # Parameter-based additional scaling (larger models run relatively slower on mobile)
    param_factor = max(1.0, (model_params / 1e6) ** 0.3)  # Mild scaling with model size
    
    estimates = {}
    for device_class, factors in mobile_factors.items():
        estimates[device_class] = {
            'cpu_ms': desktop_time_ms * factors['cpu'] * param_factor,
            'gpu_ms': desktop_time_ms * factors['gpu'] * param_factor,
            'gpu_optimized_ms': desktop_time_ms * factors['gpu'] * param_factor * 0.6  # With quantization
        }
    
    return estimates

def benchmark_all_variants():
    """Benchmark all optimization variants"""
    
    variants = ['ultra_fast', 'fast', 'balanced', 'accurate']
    results = []
    
    print("IMUNet Optimization Variants Benchmark")
    print("=" * 70)
    print()
    
    for variant in variants:
        print(f"Benchmarking {variant} variant...")
        
        # Create model
        imunet = OptimizedIMUNet(variant)
        model = imunet.build_model()
        info = imunet.get_model_info()
        
        # Benchmark on desktop
        input_shape = (info['config']['window_size'], info['config']['input_channels'])
        desktop_results = benchmark_inference(model, input_shape)
        
        # Estimate mobile performance
        mobile_estimates = estimate_mobile_performance(
            desktop_results['mean_ms'], 
            info['total_params']
        )
        
        # Store results
        result = {
            'variant': variant,
            'params': info['total_params'],
            'window_size': info['config']['window_size'],
            'max_channels': info['config']['base_channels'] * max(info['config']['channel_multipliers']),
            'desktop_ms': desktop_results['mean_ms'],
            'desktop_std_ms': desktop_results['std_ms'],
            **{f'mobile_{k}_{m}': v for k, v_dict in mobile_estimates.items() for m, v in v_dict.items()}
        }
        
        results.append(result)
        
        # Print summary for this variant
        print(f"  Parameters: {info['total_params']:,}")
        print(f"  Desktop inference: {desktop_results['mean_ms']:.2f} ± {desktop_results['std_ms']:.2f} ms")
        print(f"  Estimated mobile GPU (high-end): {mobile_estimates['high_end']['gpu_optimized_ms']:.1f} ms")
        print()
    
    return results

def create_comparison_table(results):
    """Create a comparison table of all variants"""
    
    df = pd.DataFrame(results)
    
    print("\nDetailed Comparison Table")
    print("=" * 120)
    
    # Performance comparison
    print("\nDesktop Performance:")
    print(f"{'Variant':<12} {'Params':<10} {'Window':<8} {'Channels':<10} {'Desktop (ms)':<15}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        print(f"{row['variant']:<12} {row['params']:<10,} {row['window_size']:<8} "
              f"{row['max_channels']:<10} {row['desktop_ms']:<15.2f}")
    
    # Mobile estimates
    print("\nMobile Performance Estimates (ms):")
    print(f"{'Variant':<12} {'High-end GPU':<12} {'Mid-range GPU':<14} {'Low-end GPU':<12} {'Speedup vs Accurate':<18}")
    print("-" * 75)
    
    accurate_time = df[df['variant'] == 'accurate']['mobile_high_end_gpu_optimized_ms'].iloc[0]
    
    for _, row in df.iterrows():
        high_end = row['mobile_high_end_gpu_optimized_ms']
        mid_range = row['mobile_mid_range_gpu_optimized_ms']
        low_end = row['mobile_low_end_gpu_optimized_ms']
        speedup = accurate_time / high_end
        
        print(f"{row['variant']:<12} {high_end:<12.1f} {mid_range:<14.1f} "
              f"{low_end:<12.1f} {speedup:<18.1f}x")
    
    print("\nNotes:")
    print("- Desktop times measured on this machine")
    print("- Mobile estimates are rough approximations")
    print("- GPU optimized includes quantization benefits")
    print("- Actual performance will vary by device and framework optimization")

def save_results(results, filename='benchmark_results.csv'):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

def memory_usage_analysis():
    """Analyze memory usage of different variants"""
    
    print("\nMemory Usage Analysis")
    print("=" * 50)
    
    variants = ['ultra_fast', 'fast', 'balanced', 'accurate']
    
    print(f"{'Variant':<12} {'Model Size (MB)':<15} {'Params':<10} {'Reduction vs Accurate':<20}")
    print("-" * 60)
    
    accurate_params = OptimizedIMUNet('accurate').get_model_info()['total_params']
    
    for variant in variants:
        imunet = OptimizedIMUNet(variant)
        info = imunet.get_model_info()
        
        # Estimate model size (4 bytes per float32 parameter)
        model_size_mb = (info['total_params'] * 4) / (1024 * 1024)
        reduction_factor = accurate_params / info['total_params']
        
        print(f"{variant:<12} {model_size_mb:<15.2f} {info['total_params']:<10,} {reduction_factor:<20.1f}x")

def main():
    """Main benchmark function"""
    
    print("Starting comprehensive IMUNet optimization benchmark...")
    print("This will test inference performance of all variants.\n")
    
    # Run benchmarks
    results = benchmark_all_variants()
    
    # Create comparison table
    create_comparison_table(results)
    
    # Memory analysis
    memory_usage_analysis()
    
    # Save results
    save_results(results)
    
    print("\nRecommendations:")
    print("- For real-time applications (>30 FPS): Use 'ultra_fast' variant")
    print("- For balanced performance: Use 'fast' variant")
    print("- For highest accuracy: Use 'balanced' or 'accurate' variant")
    print("- Always test on your target mobile device for actual performance")

if __name__ == "__main__":
    main()