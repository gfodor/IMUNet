#!/usr/bin/env python3
"""
Simple PyTorch training script for IMUNet using the collected IMU data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import argparse
import sys
sys.path.append('RONIN_torch')
from IMUNet import IMUNet


class IMUDataset(Dataset):
    """Custom Dataset for IMU data"""
    
    def __init__(self, windows, targets):
        self.windows = torch.FloatTensor(windows)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]


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
            windows.append(window.T)  # Transpose to (6, 200)
            # Create dummy target (velocity) - in real scenario this would be computed from pose
            target = np.array([0.1, 0.1])  # Dummy vx, vy
            targets.append(target)
    
    return np.array(windows), np.array(targets)


def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    """Train the PyTorch model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IMUNet PyTorch Simple Training Script')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--data_dir', type=str, default='imu_data',
                        help='Directory containing IMU data (default: imu_data)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model (default: models)')
    
    args = parser.parse_args()
    
    print("IMUNet PyTorch Simple Training Script")
    print("=====================================")
    print(f"Training configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create datasets and data loaders
    train_dataset = IMUDataset(X_train, y_train)
    val_dataset = IMUDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = IMUNet(
        num_classes=2, 
        input_size=(1, 6, 200), 
        sampling_rate=200, 
        num_T=32, 
        num_S=64, 
        hidden=64, 
        dropout_rate=0.5
    )
    
    print("Model created successfully!")
    print(f"Model has {model.get_num_params()} parameters")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        args.epochs, args.lr, device
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'imunet_simple_torch.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'num_classes': 2,
            'input_size': (1, 6, 200),
            'sampling_rate': 200,
            'num_T': 32,
            'num_S': 64,
            'hidden': 64,
            'dropout_rate': 0.5
        }
    }, model_path)
    print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()