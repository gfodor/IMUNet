# IMUNet Training Guide for ARCore Data

This document provides a comprehensive guide for training IMUNet on ARCore-collected data for inertial navigation and velocity estimation.

## Overview

This setup trains the IMUNet neural network (from the paper "IMUNet: Efficient Neural Networks for IMU-based Human Motion Estimation") on data collected via ARCore-enabled Android devices. The model learns to predict 2D velocity from 6-axis IMU data (3-axis gyroscope + 3-axis accelerometer).

## Data Pipeline Architecture

### 1. Raw Data Collection (ARCore App)
- **Input**: ARCore-enabled Android app collects sensor data
- **Files generated per session**:
  - `acce.txt`: Raw accelerometer data (timestamp, x, y, z)
  - `gyro.txt`: Raw gyroscope data (timestamp, x, y, z)  
  - `pose.txt`: ARCore pose estimates (timestamp, position xyz, orientation xyzw)
  - `orientation.txt`: Device rotation vector (timestamp, x, y, z, w)
  - Other files: `gravity.txt`, `linacce.txt`, `magnet.txt` (not used for training)

### 2. Data Processing Pipeline (`process_arcore_data.py`)

#### Input Data Structure
```
imu_data/
├── device_name/
│   └── timestamp_folder/
│       ├── acce.txt
│       ├── gyro.txt
│       ├── pose.txt
│       ├── orientation.txt
│       └── [other sensor files]
```

#### Processing Steps

1. **Data Loading & Validation**
   - Loads raw sensor data from text files
   - Validates required files exist
   - Applies front/back trimming (120 samples each) to remove startup/shutdown artifacts

2. **Coordinate System Transformations**
   - **Pose quaternions**: Converts from ARCore [x,y,z,w] → standard [w,x,y,z] format
   - **Position coordinates**: Transforms ARCore coordinates to navigation frame:
     - `pos_x = pose_x`
     - `pos_y = -pose_z` (flip Z to Y)
     - `pos_z = pose_y` (Y becomes Z)

3. **Temporal Resampling** 
   - **Target rate**: 200Hz (uniform sampling)
   - **Method**: Cubic spline interpolation for smooth temporal alignment
   - Handles different sensor rates (gyro/accel ~400-500Hz, pose ~30Hz, orientation ~200Hz)

4. **Feature Engineering**
   - **IMU data**: Linear interpolation to 200Hz timeline
   - **Quaternion data**: SLERP (spherical linear interpolation) for rotation data
   - **Position tracking**: Smooth trajectory interpolation

5. **Output Generation**
   Creates `processed/data.csv` with columns:
   ```
   time,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,
   pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z
   ```

### 3. Training Data Format (`ProposedSequence` class)

#### Feature Extraction
The `ProposedSequence` class in `utils.py` converts processed CSV data into training features:

```python
# Input: 18-column CSV data
# Output: 6-dimensional features (global frame IMU data)

# Raw IMU data (device frame)
gyro = [gyro_x, gyro_y, gyro_z]  # rad/s
acce = [acce_x, acce_y, acce_z]  # m/s²

# Transform to global frame using orientation
gyro_global = orientation * gyro * orientation.conj()
acce_global = orientation * acce * orientation.conj()

# Final features: [gyro_global_xyz, acce_global_xyz]
features = concatenate([gyro_global, acce_global])  # Shape: (N, 6)
```

#### Target Generation
```python
# Velocity targets computed from position differences
dt = timestamps[i+1] - timestamps[i]
velocity_2d = (position[i+1] - position[i]) / dt
targets = velocity_2d[:2]  # Only X,Y velocity, Shape: (N-1, 2)
```

### 4. Neural Network Architecture (`IMUNet.py`)

#### Model Structure
- **Input**: 6-channel IMU data (3 gyro + 3 accel) × 200 timesteps
- **Architecture**: Deep Separable Convolution network
  - Initial conv block: 6 → 64 channels
  - 7 DSConv blocks with skip connections
  - Channel progression: 64 → 64 → 128 → 256 → 512 → 1024
  - Global feature extraction + fully connected output
- **Output**: 2D velocity estimate [vx, vy] in m/s

#### Key Components
```python
class DSConv(nn.Module):
    """Depthwise Separable Convolution with residual connections"""
    - Depthwise convolution (grouped conv)
    - Pointwise convolution (1×1 conv) 
    - Batch normalization + ELU activation
    - Residual connection with optional downsampling

class CustomLayer(nn.Module):
    """Learnable noise compensation layer"""
    - Learnable parameters W and B
    - Applies: output = input_features - W * raw_input + B
```

## Training Configuration

### Dataset Setup
```
Datasets/proposed/
├── samsung_s9tab/           # Symlink to imu_data/samsung_s9tab/
├── fold/                    # Symlink to imu_data/fold/
├── samsung_crap/            # Symlink to imu_data/samsung_crap/
├── pixel_p8/                # Symlink to imu_data/pixel_p8/
├── list_train.txt           # Training sequences (80% split)
└── list_test.txt            # Test sequences (20% split)
```

### Training Parameters
```bash
# Standard training configuration
--dataset proposed           # Use ProposedSequence data loader
--arch IMUNet               # Use IMUNet architecture  
--batch_size 32             # Batch size (adjust for memory)
--epochs 50                 # Training epochs
--lr 1e-4                   # Learning rate (Adam optimizer)
--window_size 200           # Input sequence length
--step_size 10              # Sliding window step
```

### Loss Function & Optimization
- **Loss**: Mean Squared Error (MSE) on velocity predictions
- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=10)
- **Device**: Auto-detection (CUDA > MPS > CPU)

## Data Characteristics

### Current Dataset Stats
```
Total sequences: 4
Total duration: 19.08 minutes
Sample rate: 200Hz uniform

Training set (3 sequences):
- samsung_s9tab/20250726085048: 296.71s (59,342 samples)
- fold/20250727061722: 299.19s (59,837 samples)  
- samsung_crap/20250727050037: 252.83s (50,565 samples)

Test set (1 sequence):
- pixel_p8/20250726084345: 296.36s (59,271 samples)
```

### Data Quality Metrics
- **Temporal consistency**: All sequences resampled to 200Hz
- **Coordinate alignment**: Proper frame transformations applied
- **Motion diversity**: Multiple devices and movement patterns
- **Ground truth**: ARCore SLAM provides position/orientation references

## Running Training

### Basic Training Command
```bash
cd /Users/gfodor/portal/IMUNet/RONIN_torch

python main.py \
  --mode train \
  --dataset proposed \
  --arch IMUNet \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4
```

### Alternative Configurations

#### Fast Testing (Small epochs)
```bash
python main.py --mode train --dataset proposed --arch IMUNet \
  --batch_size 16 --epochs 10 --lr 1e-4
```

#### CPU-Only Training  
```bash
python main.py --mode train --dataset proposed --arch IMUNet \
  --batch_size 16 --epochs 50 --lr 1e-4 --cpu
```

#### Custom Paths
```bash
python main.py --mode train --dataset proposed --arch IMUNet \
  --train_list /path/to/custom_train.txt \
  --val_list /path/to/custom_val.txt \
  --root_dir /path/to/data \
  --out_dir /path/to/output
```

### Training Outputs

Training creates the following outputs in `Train_out/IMUNet/proposed/`:

```
Train_out/IMUNet/proposed/
├── checkpoints/
│   ├── checkpoint_best.pt      # Best validation model
│   └── checkpoint_latest.pt    # Latest epoch model
├── logs/                       # TensorBoard logs
│   └── events.out.tfevents.*
└── config.json                 # Training configuration
```

## Model Evaluation

### Test Mode
```bash
python main.py \
  --mode test \
  --dataset proposed \
  --arch IMUNet \
  --model_path Train_out/IMUNet/proposed/checkpoints/checkpoint_best.pt
```

### Evaluation Metrics
- **MSE Loss**: Mean squared error on velocity predictions
- **ATE**: Absolute Trajectory Error (position accuracy)
- **RTE**: Relative Trajectory Error (local consistency)

### Output Analysis
Test mode generates:
- Trajectory plots (predicted vs ground truth)
- Per-sequence error statistics
- CSV files with detailed results
- Visualization plots saved as PNG files

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths in list files match directory structure
   - Ensure processed CSV files exist in each sequence folder
   - Verify symbolic links are correctly created

2. **Memory Issues**
   - Reduce batch size (try 16 or 8)
   - Use CPU mode if GPU memory insufficient
   - Monitor system memory during data loading

3. **Training Convergence**
   - Check learning rate (try 5e-5 or 2e-4)
   - Verify data normalization and scaling
   - Monitor validation loss curves

4. **Performance Issues**
   - Use MPS acceleration on Mac (auto-detected)
   - Enable multiprocessing for data loading
   - Consider data caching for repeated training

### Data Quality Checks

```python
# Verify data loading
from RONIN_torch.utils import ProposedSequence
seq = ProposedSequence('/path/to/sequence')
print(f"Features: {seq.get_feature().shape}")
print(f"Targets: {seq.get_target().shape}")
```

### Debugging Tools

```bash
# Test single sequence processing
python test_arcore_data.py

# Debug data pipeline
python debug_samsung_crap.py

# Validate processed data
python -c "
import pandas as pd
df = pd.read_csv('path/to/processed/data.csv')
print(df.describe())
"
```

## Extending the System

### Adding New Sequences
1. Place new data in `imu_data/device_name/timestamp/`
2. Run `python process_arcore_data.py`
3. Update `list_train.txt` and `list_test.txt` with new sequences
4. Re-run training

### Modifying Architecture
- Edit `IMUNet.py` for model changes
- Adjust `feature_dim`, `target_dim` in `ProposedSequence` class
- Update training parameters accordingly

### Custom Data Processing
- Modify `process_arcore_data.py` for different sensor configurations
- Adjust coordinate transformations for different reference frames
- Customize resampling rates or window sizes

## References

- **IMUNet Paper**: [Add paper reference when available]
- **ARCore Documentation**: https://developers.google.com/ar
- **RONIN Dataset**: http://ronin.cs.sfu.ca/
- **PyTorch Documentation**: https://pytorch.org/docs/