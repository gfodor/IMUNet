#!/usr/bin/env python3
"""
Simple training script for IMUNet using the collected IMU data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import glob
import argparse
from RONIN_keras.IMUNet import IMUNet

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
    
    # Create features (6 channels: ax, ay, az, gx, gy, gz)
    features = combined[['gx', 'gy', 'gz', 'ax', 'ay', 'az']].values
    
    # Create sliding windows
    windows = []
    targets = []
    
    for i in range(0, len(features) - window_size, step_size):
        window = features[i:i+window_size]
        if window.shape[0] == window_size:
            windows.append(window.T)  # Transpose to (6, 200)
            # Create dummy target (velocity) - in real scenario this would be computed from pose
            target = np.array([0.1, 0.1])  # Dummy vx, vy
            targets.append(target)
    
    return np.array(windows), np.array(targets)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IMUNet Simple Training Script')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--data_dir', type=str, default='imu_data',
                        help='Directory containing IMU data (default: imu_data)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model (default: models)')
    
    args = parser.parse_args()
    
    print("IMUNet Simple Training Script")
    print("=============================")
    print(f"Training configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print()
    
    # Load IMU data
    data_dirs = glob.glob(f'{args.data_dir}/*')
    all_windows = []
    all_targets = []
    
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            print(f"Loading data from {data_dir}")
            try:
                acce_data, gyro_data = load_imu_data(data_dir)
                windows, targets = create_windows(acce_data, gyro_data)
                all_windows.append(windows)
                all_targets.append(targets)
                print(f"Created {len(windows)} windows from {os.path.basename(data_dir)}")
            except Exception as e:
                print(f"Error loading {data_dir}: {e}")
    
    if not all_windows:
        print("No data found! Please ensure IMU data is in imu_data/ directory")
        return
    
    # Combine all data
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_targets, axis=0)
    
    print(f"Total dataset: {X.shape[0]} samples")
    print(f"Input shape: {X.shape[1:]}")
    print(f"Output shape: {y.shape[1:]}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    model = IMUNet(input_shape=(6, 200), num_classes=2)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    print("Model created successfully!")
    model.summary()
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'imunet_simple.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()