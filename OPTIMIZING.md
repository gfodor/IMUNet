# IMUNet Optimization Guide for Mobile Deployment

## Overview

This guide provides concrete recommendations for optimizing IMUNet for mobile deployment on Adreno GPUs using TensorFlow Lite. Based on analysis of the current architecture and mobile inference research, significant speedups (3-10x) are achievable through architectural and quantization optimizations.

## Current Model Analysis

### Baseline IMUNet Architecture
- **Parameters**: ~3.66M
- **Input**: 6 channels × 200 timesteps
- **Architecture**: 6 conv blocks with channel progression: 64→128→256→512→1024
- **Estimated FLOPs**: ~18-23M
- **Expected Adreno latency**: 15-25ms (unoptimized)

## Optimization Strategies

### 1. Architecture Optimization (3-4x speedup)

#### Channel Width Reduction
**Current**: 64→128→256→512→1024 channels  
**Optimized**: 32→64→128→256→512 channels

**Benefits**:
- 75% parameter reduction
- 3-4x inference speedup
- Minimal accuracy loss for IMU tasks

#### Network Depth Reduction
**Remove**: Final conv block (conv_7_*)  
**Benefits**:
- 25% fewer operations
- Reduced memory footprint

#### Input Window Optimization
**Current**: 200 timesteps  
**Optimized**: 100-128 timesteps

**Benefits**:
- 40% fewer operations
- Still captures essential temporal patterns for IMU
- Reduces input data requirements

### 2. TensorFlow Lite Optimization (2-3x speedup)

#### INT8 Quantization
- **Speed**: 2-3x faster on Adreno
- **Memory**: 4x reduction
- **Accuracy loss**: Typically <2% for IMU models

#### GPU Delegate
- **NNAPI integration**: Native Adreno acceleration
- **Expected speedup**: 3x vs CPU-only

### 3. Mobile-Specific Optimizations

#### Remove Custom Components
- **Custom noise layer**: Simplify for mobile optimization
- **Complex residual connections**: Use standard patterns

#### Optimized Activation Functions
- **ReLU instead of ELU**: Better mobile GPU support
- **Fused operations**: BatchNorm + Activation fusion

## Architecture Variants

### Variant Configurations

| Variant | Channels | Blocks | Window | Params | Speed | Use Case |
|---------|----------|--------|--------|--------|-------|----------|
| **Ultra-Fast** | 16→32→64→128 | 4 | 64 | ~200K | 10x faster | Real-time (>100Hz) |
| **Fast** | 32→64→128→256 | 5 | 100 | ~800K | 5x faster | High-frequency (50Hz) |
| **Balanced** | 32→64→128→256→512 | 5 | 128 | ~1.5M | 3x faster | Standard (20Hz) |
| **Accurate** | 64→128→256→512 | 6 | 200 | ~2.5M | 2x faster | High-precision |

### Recommended Starting Point: **Fast Variant**
- Good balance of speed and accuracy
- 5x speedup over baseline
- Suitable for most mobile applications

## Implementation Guide

### 1. Model Configuration
```python
# Fast variant configuration
config = {
    'base_channels': 32,
    'channel_multipliers': [1, 2, 4, 8],  # 32, 64, 128, 256
    'num_blocks': 5,
    'window_size': 100,
    'input_channels': 6,
    'num_classes': 3,  # For 3D velocity
    'dropout_rate': 0.3
}
```

### 2. Training Strategy
1. **Start with Fast variant** for baseline
2. **Train with augmentation**: Random horizontal rotation, noise injection
3. **Use smaller learning rate**: 5e-5 for smaller models
4. **Early stopping**: Monitor validation loss
5. **Quantization-aware training**: For best INT8 performance

### 3. TensorFlow Lite Conversion
```python
# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()
```

### 4. Performance Validation
Test on target device with:
- **Latency benchmarking**: Using TensorFlow Lite benchmark tool
- **Accuracy validation**: Compare against original model
- **Memory usage**: Monitor peak memory during inference

## Expected Performance Gains

### Latency Improvements (Modern Adreno GPU)
- **Baseline**: 15-25ms
- **Architecture optimization**: 5-8ms
- **+ TensorFlow Lite**: 3-5ms  
- **+ INT8 quantization**: 1-3ms
- **+ GPU delegate**: <2ms

### Memory Reductions
- **Model size**: 75% smaller (Fast variant)
- **Runtime memory**: 60% reduction
- **Storage**: 4x smaller with quantization

## Accuracy Trade-offs

Based on research and similar architectures:
- **Fast variant**: 90-95% of original accuracy
- **Ultra-fast variant**: 85-90% of original accuracy
- **With quantization**: Additional 1-2% accuracy loss

For IMU navigation tasks, these trade-offs are typically acceptable given the significant speed improvements.

## Testing Protocol

### 1. Baseline Establishment
1. Train original model on your dataset
2. Record accuracy metrics (ATE, RTE)
3. Measure inference time on target device

### 2. Variant Testing
1. Train each variant with same hyperparameters
2. Compare accuracy vs baseline
3. Measure inference time and memory usage
4. Test with quantization

### 3. Production Validation
1. Test with real IMU data streams
2. Validate in actual application scenarios
3. Monitor for numerical stability issues

## Next Steps

1. **Implement parameterized training script** (see `train_optimized.py`)
2. **Train Fast variant** as starting point
3. **Convert to TensorFlow Lite** with quantization
4. **Benchmark on target device**
5. **Iteratively optimize** based on results

The provided training script allows easy experimentation with all these variants through command-line parameters.