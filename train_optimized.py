#!/usr/bin/env python3
"""
Optimized IMUNet Training Script with Parameterized Architecture Variants
Supports easy experimentation with different model configurations
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import glob
from optimized_imunet import OptimizedIMUNet, create_optimized_model

# Set memory growth for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_imu_data(data_dir):
    """Load IMU data from text files"""
    acce_file = os.path.join(data_dir, 'acce.txt')
    gyro_file = os.path.join(data_dir, 'gyro.txt')
    
    # Read accelerometer data
    acce_data = []
    with open(acce_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = float(parts[0])
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                acce_data.append([timestamp, ax, ay, az])
    
    # Read gyroscope data
    gyro_data = []
    with open(gyro_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = float(parts[0])
                gx, gy, gz = float(parts[1]), float(parts[2]), float(parts[3])
                gyro_data.append([timestamp, gx, gy, gz])
    
    return np.array(acce_data), np.array(gyro_data)


def create_windows(acce_data, gyro_data, window_size=200, step_size=10):
    """Create windowed data for training"""
    # Align timestamps and create combined data
    acce_df = pd.DataFrame(acce_data, columns=['timestamp', 'ax', 'ay', 'az'])
    gyro_df = pd.DataFrame(gyro_data, columns=['timestamp', 'gx', 'gy', 'gz'])
    
    # Merge on nearest timestamps
    combined = pd.merge_asof(acce_df.sort_values('timestamp'), 
                            gyro_df.sort_values('timestamp'), 
                            on='timestamp', direction='nearest')
    
    # Create features (6 channels: gx, gy, gz, ax, ay, az)
    features = combined[['gx', 'gy', 'gz', 'ax', 'ay', 'az']].values
    
    # Create sliding windows
    windows = []
    targets = []
    
    for i in range(0, len(features) - window_size, step_size):
        window = features[i:i+window_size]
        if window.shape[0] == window_size:
            windows.append(window)  # Shape: (window_size, 6)
            # Create dummy target (velocity) - in real scenario this would be computed from pose
            if window_size >= 128:
                target = np.array([0.1, 0.1, 0.05])  # Dummy vx, vy, vz
            else:
                target = np.array([0.2, 0.2, 0.1])   # Adjusted for shorter windows
            targets.append(target)
    
    return np.array(windows), np.array(targets)


def create_data_augmentation():
    """Create data augmentation for IMU data"""
    def augment_imu(x, y):
        # Add noise
        noise = tf.random.normal(tf.shape(x), stddev=0.01)
        x_aug = x + noise
        
        # Random scaling
        scale = tf.random.uniform([], 0.9, 1.1)
        x_aug = x_aug * scale
        
        return x_aug, y
    
    return augment_imu


def prepare_datasets(data_dirs, config, validation_split=0.2, batch_size=32):
    """Prepare training and validation datasets"""
    all_windows = []
    all_targets = []
    
    window_size = config['window_size']
    step_size = max(1, window_size // 20)  # Adaptive step size
    
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            print(f"Loading data from {data_dir}")
            try:
                acce_data, gyro_data = load_imu_data(data_dir)
                windows, targets = create_windows(acce_data, gyro_data, window_size, step_size)
                all_windows.append(windows)
                all_targets.append(targets)
                print(f"Created {len(windows)} windows from {os.path.basename(data_dir)}")
            except Exception as e:
                print(f"Error loading {data_dir}: {e}")
    
    if not all_windows:
        raise ValueError("No data found! Please ensure IMU data is in the specified directory")
    
    # Combine all data
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_targets, axis=0)
    
    print(f"Total dataset: {X.shape[0]} samples")
    print(f"Input shape: {X.shape[1:]}")
    print(f"Output shape: {y.shape[1:]}")
    
    # Split data
    split_idx = int((1 - validation_split) * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32)))
    
    # Apply augmentation to training data
    augment_fn = create_data_augmentation()
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


def create_callbacks(output_dir, config):
    """Create training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, 'best_model.h5')
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ))
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ))
    
    # Learning rate reduction
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ))
    
    # TensorBoard logging
    log_dir = os.path.join(output_dir, 'logs')
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    ))
    
    return callbacks


def train_model(model, train_ds, val_ds, config, output_dir):
    """Train the model"""
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=config.get('learning_rate', 1e-4))
    model.compile(
        optimizer=optimizer,
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError(), metrics.RootMeanSquaredError()]
    )
    
    # Create callbacks
    callbacks = create_callbacks(output_dir, config)
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.get('epochs', 100),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def convert_to_tflite(model, output_dir, quantize=True):
    """Convert model to TensorFlow Lite"""
    
    # Standard conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Enable optimization and quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For INT8 quantization, you would need representative dataset
        # converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, 'model_quantized.tflite')
    else:
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, 'model.tflite')
    
    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
    
    return tflite_path


def benchmark_model(model, config):
    """Benchmark model performance"""
    # Create dummy input
    dummy_input = tf.random.normal([1, config['window_size'], config['input_channels']])
    
    # Warm up
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    import time
    times = []
    for _ in range(100):
        start = time.time()
        _ = model(dummy_input)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print(f"Inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    
    return avg_time


def save_config(config, output_dir):
    """Save configuration to JSON file"""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimized IMUNet Training Script')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='imu_data',
                        help='Directory containing IMU data (default: imu_data)')
    parser.add_argument('--output_dir', type=str, default='models/optimized',
                        help='Directory to save trained model (default: models/optimized)')
    
    # Model arguments
    parser.add_argument('--variant', type=str, default='fast',
                        choices=['ultra_fast', 'fast', 'balanced', 'accurate'],
                        help='Model variant preset (default: fast)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='Override window size for the model')
    parser.add_argument('--base_channels', type=int, default=None,
                        help='Override base channel count')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes (default: 3 for vx,vy,vz)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Conversion arguments
    parser.add_argument('--convert_tflite', action='store_true',
                        help='Convert model to TensorFlow Lite after training')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply quantization during TFLite conversion')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark model inference time')
    
    args = parser.parse_args()
    
    print("Optimized IMUNet Training Script")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model configuration
    imunet = OptimizedIMUNet(args.variant)
    config = imunet.config.copy()
    
    # Override config with command line arguments
    if args.window_size is not None:
        config['window_size'] = args.window_size
    if args.base_channels is not None:
        config['base_channels'] = args.base_channels
    config['num_classes'] = args.num_classes
    
    # Add training configuration
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split
    })
    
    # Save configuration
    save_config(config, args.output_dir)
    
    # Create model
    imunet_optimized = OptimizedIMUNet(config)
    model = imunet_optimized.build_model()
    
    # Print model info
    info = imunet_optimized.get_model_info()
    print(f"\nModel Configuration: {args.variant}")
    print(f"Parameters: {info['total_params']:,}")
    print(f"Input shape: {info['input_shape']}")
    print(f"Output shape: {info['output_shape']}")
    print(f"Window size: {config['window_size']}")
    print(f"Max channels: {config['base_channels'] * max(config['channel_multipliers'])}")
    
    model.summary()
    
    # Prepare data
    data_dirs = glob.glob(f'{args.data_dir}/*') if os.path.exists(args.data_dir) else []
    if not data_dirs:
        print(f"No data found in {args.data_dir}. Please ensure IMU data is available.")
        return
    
    train_ds, val_ds = prepare_datasets(
        data_dirs, config, 
        validation_split=args.validation_split,
        batch_size=args.batch_size
    )
    
    # Train model
    print("\nStarting training...")
    history = train_model(model, train_ds, val_ds, config, args.output_dir)
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        model = keras.models.load_model(best_model_path)
        print(f"Loaded best model from {best_model_path}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\nBenchmarking model performance...")
        benchmark_model(model, config)
    
    # Convert to TensorFlow Lite if requested
    if args.convert_tflite:
        print("\nConverting to TensorFlow Lite...")
        tflite_path = convert_to_tflite(model, args.output_dir, quantize=args.quantize)
        
        if args.benchmark:
            print("Benchmarking TensorFlow Lite model...")
            # Load and benchmark TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Benchmark TFLite inference
            dummy_input = np.random.normal(0, 1, input_details[0]['shape']).astype(np.float32)
            
            import time
            times = []
            for _ in range(100):
                start = time.time()
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            print(f"TensorFlow Lite inference time: {avg_time:.2f} ms")
    
    print(f"\nTraining completed! Models saved to {args.output_dir}")
    print("\nTo experiment with different variants, try:")
    print("  python train_optimized.py --variant ultra_fast --window_size 64")
    print("  python train_optimized.py --variant balanced --convert_tflite --quantize")


if __name__ == '__main__':
    main()