# IMUNet Training Guide

This guide explains how to train the IMUNet neural network for inertial navigation using IMU sensor data.

## Overview

IMUNet is a neural network architecture designed for processing IMU (Inertial Measurement Unit) data to perform inertial navigation. The network takes accelerometer and gyroscope readings as input and predicts velocity vectors.

## Environment Setup

### Prerequisites

- Python 3.11+
- Conda package manager
- CUDA-capable GPU (optional, but recommended)

### 1. Create and Activate Conda Environment

The project uses a conda environment called `imunet`:

```bash
# If environment doesn't exist, create it:
conda create -n imunet python=3.11

# Activate the environment
conda activate imunet
```

### 2. Install Dependencies

Install required Python packages:

```bash
pip install tensorflow torch torchvision numpy scipy pandas matplotlib h5py quaternion numba tensorboardX
```

## Project Structure

```
IMUNet/
├── RONIN_keras/          # Keras/TensorFlow implementation
│   ├── main.py          # Main training script
│   ├── IMUNet.py        # IMUNet model definition
│   ├── utils.py         # Data loading utilities
│   └── quaternion.py    # Quaternion math utilities
├── RONIN_torch/          # PyTorch implementation
├── Datasets/             # Dataset storage
├── imu_data/            # Your collected IMU data
├── train_simple.py      # Simple training script
└── models/              # Saved model outputs
```

## Data Format

### IMU Data Structure

The training expects IMU data in the following format:

```
imu_data/
├── YYYYMMDDHHMMSS/      # Timestamp-based folder
│   ├── acce.txt         # Accelerometer data
│   ├── gyro.txt         # Gyroscope data
│   ├── pose.txt         # Ground truth pose (optional)
│   └── ...              # Other sensor data
```

### Data File Format

**Accelerometer (`acce.txt`):**
```
# Created at [timestamp]
[timestamp_ns] [ax] [ay] [az]
[timestamp_ns] [ax] [ay] [az]
...
```

**Gyroscope (`gyro.txt`):**
```
# Created at [timestamp]
[timestamp_ns] [gx] [gy] [gz]
[timestamp_ns] [gx] [gy] [gz]
...
```

Where:
- `timestamp_ns`: Nanosecond timestamp
- `ax, ay, az`: Acceleration in m/s² for X, Y, Z axes
- `gx, gy, gz`: Angular velocity in rad/s for X, Y, Z axes

## Training Methods

### Method 1: Simple Training (Recommended for beginners)

#### Keras/TensorFlow Implementation

Use the simplified Keras training script with your collected IMU data:

```bash
cd /path/to/IMUNet
python train_simple.py
```

#### PyTorch Implementation

Use the simplified PyTorch training script with your collected IMU data:

```bash
cd /path/to/IMUNet
python train_simple_torch.py
```

#### Command Line Options

Both simple training scripts (Keras and PyTorch) support several command line arguments:

**Keras/TensorFlow script (`train_simple.py`):**
```bash
python train_simple.py [OPTIONS]

Options:
  --epochs EPOCHS           Number of training epochs (default: 5)
  --batch_size BATCH_SIZE   Batch size for training (default: 16)
  --data_dir DATA_DIR       Directory containing IMU data (default: imu_data)
  --output_dir OUTPUT_DIR   Directory to save trained model (default: models)
  -h, --help               Show help message and exit
```

**PyTorch script (`train_simple_torch.py`):**
```bash
python train_simple_torch.py [OPTIONS]

Options:
  --epochs EPOCHS           Number of training epochs (default: 5)
  --batch_size BATCH_SIZE   Batch size for training (default: 16)
  --lr LR                   Learning rate (default: 1e-4)
  --data_dir DATA_DIR       Directory containing IMU data (default: imu_data)
  --output_dir OUTPUT_DIR   Directory to save trained model (default: models)
  -h, --help               Show help message and exit
```

#### Usage Examples

**Keras/TensorFlow:**
```bash
# Train for 10 epochs with default settings
python train_simple.py --epochs 10

# Train with larger batch size and custom directories
python train_simple.py --epochs 20 --batch_size 32 --data_dir my_imu_data --output_dir my_models

# Quick test run with 1 epoch
python train_simple.py --epochs 1
```

**PyTorch:**
```bash
# Train for 10 epochs with default settings
python train_simple_torch.py --epochs 10

# Train with custom learning rate and batch size
python train_simple_torch.py --epochs 20 --batch_size 32 --lr 5e-4

# Quick test run with 1 epoch
python train_simple_torch.py --epochs 1

# Train with custom directories
python train_simple_torch.py --data_dir my_imu_data --output_dir my_models
```

Both scripts:
- Automatically load data from the specified IMU data directory
- Create sliding windows from the sensor data
- Train the IMUNet model for the specified number of epochs
- Save the trained model to the specified output directory

**Output files:**
- Keras: `models/imunet_simple.h5` (HDF5 format)
- PyTorch: `models/imunet_simple_torch.pt` (PyTorch checkpoint with training history)

#### Framework Comparison

| Feature | Keras/TensorFlow | PyTorch |
|---------|------------------|---------|
| **Model File** | `.h5` format | `.pt` checkpoint |
| **Model Size** | ~12.8 MB | ~14.0 MB |
| **Parameters** | 3,352,690 | 3,661,618 |
| **Training Speed** | Faster (optimized) | Slightly slower |
| **Memory Usage** | Lower | Higher |
| **Deployment** | TensorFlow Lite, TF Serving | TorchScript, ONNX |
| **Debugging** | Good | Excellent |
| **Research** | Production-ready | Research-friendly |

**Recommendation:** Use Keras for production deployment, PyTorch for research and experimentation.

### Method 2: Advanced Training with Datasets

For training with research datasets (RONIN, RIDI, etc.):

```bash
cd RONIN_keras

# Train with different architectures
python main.py --dataset proposed --mode train --arch IMUNet --epochs 100 --batch_size 32

# Available architectures:
# --arch ResNet          # ResNet18 baseline
# --arch MobileNet       # MobileNetV1
# --arch MobileNetV2     # MobileNetV2
# --arch MnasNet         # MnasNet
# --arch EfficientNet    # EfficientNetB0
# --arch IMUNet          # Proposed IMUNet architecture

# Available datasets:
# --dataset ronin        # RONIN dataset
# --dataset ridi         # RIDI dataset
# --dataset proposed     # Proposed dataset
# --dataset oxiod        # OxIOD dataset
# --dataset px4          # PX4 dataset
```

### Training Parameters

Key training parameters you can adjust:

```bash
python main.py \
    --dataset proposed \
    --mode train \
    --arch IMUNet \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-4 \
    --window_size 200 \
    --step_size 10 \
    --out_dir ./outputs
```

Parameters explanation:
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 1e-4)
- `--window_size`: Input sequence length (default: 200)
- `--step_size`: Sliding window step size (default: 10)
- `--out_dir`: Output directory for models and logs

## Model Architecture

### IMUNet Architecture

The IMUNet model consists of:

1. **Input Layer**: Accepts 6-channel IMU data (3 gyro + 3 accel) with sequence length 200
2. **Initial Conv Block**: 7×1 convolution with 64 filters, stride 2
3. **Depthwise Separable Blocks**: Series of residual blocks with depthwise separable convolutions
4. **Feature Progression**: 64 → 128 → 256 → 512 → 1024 channels
5. **Global Average Pooling**: Reduces spatial dimensions
6. **Dense Layers**: Final classification/regression layers
7. **Output**: 2D velocity vector (vx, vy) or 3D for some datasets

### Model Inputs and Outputs

#### Input Format
- **Shape**: `(batch_size, 6, 200)`
- **Data Type**: Float32
- **Channels**: 6 channels representing:
  1. `gx`: Gyroscope X-axis (angular velocity in rad/s)
  2. `gy`: Gyroscope Y-axis (angular velocity in rad/s) 
  3. `gz`: Gyroscope Z-axis (angular velocity in rad/s)
  4. `ax`: Accelerometer X-axis (acceleration in m/s²)
  5. `ay`: Accelerometer Y-axis (acceleration in m/s²)
  6. `az`: Accelerometer Z-axis (acceleration in m/s²)
- **Sequence Length**: 200 time steps (representing ~1-2 seconds at typical IMU sampling rates)
- **Data Organization**: Each sample is a sliding window of IMU measurements

#### Input Data Preprocessing
The input data undergoes the following preprocessing:
1. **Temporal Alignment**: Gyroscope and accelerometer data are synchronized by timestamp
2. **Windowing**: Data is segmented into overlapping windows of 200 samples
3. **Normalization**: Raw sensor values are used without explicit normalization (model learns appropriate scaling)
4. **Channel Ordering**: Data is arranged as `[gx, gy, gz, ax, ay, az]` for each time step

#### Output Format
- **Shape**: `(batch_size, 2)` for 2D motion or `(batch_size, 3)` for 3D motion
- **Data Type**: Float32
- **Meaning**: 
  - **2D Output**: `[vx, vy]` - velocity components in X and Y directions (m/s)
  - **3D Output**: `[vx, vy, vz]` - velocity components in X, Y, and Z directions (m/s)
- **Coordinate Frame**: Global coordinate frame (world coordinates)
- **Units**: Meters per second (m/s)

#### Data Flow Example
```python
# Input: IMU window
input_shape = (batch_size, 6, 200)
# where each time step contains: [gx, gy, gz, ax, ay, az]

# Example input tensor for one sample:
sample_input = np.array([
    # Time step 0: [gx,   gy,   gz,   ax,   ay,   az  ]
                  [0.1,  0.05, -0.02, 0.8,  9.8,  0.2],
    # Time step 1: [gx,   gy,   gz,   ax,   ay,   az  ]
                  [0.12, 0.04, -0.01, 0.9,  9.7,  0.3],
    # ... (198 more time steps)
])

# Output: Velocity prediction
output_shape = (batch_size, 2)  # [vx, vy]
# Example output: [0.5, -0.2] means moving at 0.5 m/s in X, -0.2 m/s in Y
```

#### Training Target Generation
For supervised learning, ground truth velocity targets are typically:
1. **From Motion Capture**: High-precision position tracking differentiated to get velocity
2. **From Visual-Inertial SLAM**: Camera + IMU fusion systems providing pose estimates
3. **From GPS**: Differentiated GPS positions (lower accuracy, outdoor only)
4. **From Simulation**: Perfect ground truth from physics simulators

The model learns to map IMU sensor patterns to these velocity targets.

### Model Summary

```
Total params: 3,352,690 (12.79 MB)
Trainable params: 3,334,130 (12.72 MB)
Non-trainable params: 18,560 (72.50 KB)
```

## Testing

### Test Trained Model

```bash
cd RONIN_keras

python main.py \
    --mode test \
    --arch IMUNet \
    --dataset proposed \
    --model_path ./outputs/IMUNet.h5 \
    --test_list ./test_data_list.txt \
    --out_dir ./test_results
```

### Evaluation Metrics

The training outputs several metrics:
- **MSE Loss**: Mean squared error between predicted and true velocities
- **MAE**: Mean absolute error
- **ATE**: Absolute Trajectory Error (for trajectory reconstruction)
- **RTE**: Relative Trajectory Error

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python main.py --batch_size 16
   ```

2. **Module Import Errors**
   ```bash
   # Ensure you're in the correct conda environment
   conda activate imunet
   
   # Reinstall packages if needed
   pip install --force-reinstall tensorflow
   ```

3. **Data Loading Errors**
   - Ensure IMU data files exist in correct format
   - Check file permissions
   - Verify timestamp alignment between acce.txt and gyro.txt

4. **Quaternion Import Issues**
   - The project includes a custom quaternion.py for compatibility
   - If issues persist, check the quaternion module installation

### Performance Tips

1. **GPU Acceleration**
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory Optimization**
   - Use smaller batch sizes for limited memory
   - Reduce window size if needed
   - Use mixed precision training:
   ```python
   # Add to training script
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   ```

3. **Data Pipeline Optimization**
   - Use data caching (`--cache_path`)
   - Prefetch data for faster loading
   - Use multiple workers for data loading

## Output Files

After training, you'll find:

```
outputs/
├── IMUNet.h5           # Trained Keras model
├── IMUNet.ckpt         # Training checkpoint
├── IMUNet.tflite       # TensorFlow Lite model
├── train_IMUNet.npy    # Training loss history
├── val_IMUNet.npy      # Validation loss history
└── config.json         # Training configuration
```

## Advanced Usage

### Custom Data Preprocessing

For custom datasets, modify the data loading in `utils.py`:

```python
# Example custom sequence class
class CustomSequence(CompiledSequence):
    def __init__(self, data_path, **kwargs):
        # Your custom data loading logic
        pass
    
    def load(self, path):
        # Load your data format
        pass
```

### Transfer Learning

```python
# Load pretrained model
base_model = tf.keras.models.load_model('pretrained_imunet.h5')

# Freeze base layers
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Add custom head
x = base_model.layers[-3].output
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(your_num_classes)(x)

model = tf.keras.Model(base_model.input, outputs)
```

### Hyperparameter Tuning

Use tools like Optuna or Keras Tuner:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Train model with these parameters
    # Return validation loss
    return val_loss

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zeinali2024imunet,
  title={IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning},
  author={Zeinali, Behnam and Zanddizari, Hadi and Chang, Morris J},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024},
  publisher={IEEE}
}
```

## Support

For issues and questions:
1. Check this documentation
2. Review the troubleshooting section
3. Check the original paper for implementation details
4. Create an issue in the repository

---

**Note**: This implementation has been adapted and tested for modern TensorFlow/Keras. Some differences from the original paper implementation may exist.