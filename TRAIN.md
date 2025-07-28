# IMUNet Training Guide for ARCore Data

This document provides a comprehensive guide for training IMUNet on ARCore-collected data for inertial navigation and velocity estimation.

## Overview

This setup trains the IMUNet neural network (from the paper "IMUNet: Efficient Neural Networks for IMU-based Human Motion Estimation") on data collected via ARCore-enabled Android devices. The model learns to predict **3D velocity** from 6-axis IMU data (3-axis gyroscope + 3-axis accelerometer).

**NEW: 3D Velocity Support** - The system now supports full 3D velocity estimation (vx, vy, vz) instead of just 2D (vx, vy). This provides more complete motion tracking including vertical movement.

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
# Velocity targets computed from position differences (now in 3D)
dt = timestamps[i+1] - timestamps[i]
velocity_3d = (position[i+1] - position[i]) / dt
targets = velocity_3d[:3]  # Full X,Y,Z velocity, Shape: (N-1, 3)
```

### 4. Neural Network Architecture (`IMUNet.py`)

#### Model Structure
- **Input**: 6-channel IMU data (3 gyro + 3 accel) × 200 timesteps
- **Architecture**: Deep Separable Convolution network
  - Initial conv block: 6 → 64 channels
  - 7 DSConv blocks with skip connections
  - Channel progression: 64 → 64 → 128 → 256 → 512 → 1024
  - Global feature extraction + fully connected output
- **Output**: 3D velocity estimate [vx, vy, vz] in m/s (automatically adapts to dataset)

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

### 3D Velocity Architecture Updates

The system now automatically detects whether to use 2D or 3D output based on the dataset:

```python
# Dynamic output dimension selection in get_model()
def get_model(arch):
    n_class = 2  # Default to 2D
    if args.dataset == 'px4' or args.dataset == 'proposed':
        n_class = 3  # Use 3D for PX4 and proposed datasets
    
    # Model adapts automatically
    network = IMUNet(num_classes=n_class, ...)
```

**Key improvements:**
- **ProposedSequence**: Now generates full 3D velocity targets from position data
- **IMUNet**: Output layer automatically adapts to 2D/3D based on `num_classes`
- **Evaluation**: Maintains backward compatibility with 2D metrics while supporting 3D
- **Visualization**: Automatically generates 2D or 3D plots based on prediction dimensions

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
- **MSE Loss**: Mean squared error on velocity predictions (vx, vy, vz for 3D)
- **ATE**: Absolute Trajectory Error (position accuracy, computed in 2D for compatibility)
- **RTE**: Relative Trajectory Error (local consistency, computed in 2D for compatibility)

### Output Analysis
Test mode generates:
- **Trajectory plots**: 2D plots for visualization (X-Y plane), with separate velocity component plots for vx, vy, vz
- **Per-sequence error statistics**: Individual velocity component errors (vx, vy, vz)
- **CSV files**: Detailed results with format `seq,vx,vy,vz,avg,ate,rte` (3D) or `seq,vx,vy,avg,ate,rte` (2D)
- **Visualization plots**: Automatic generation of 2D or 3D plots based on prediction dimensions

### 3D Evaluation Features
- **Component-wise analysis**: Separate error metrics for each velocity component
- **3D trajectory reconstruction**: Full 3D path reconstruction from velocity predictions
- **Backward compatibility**: 2D trajectory metrics (ATE/RTE) maintained for comparison with existing results

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

## TensorFlow Lite Conversion

After training your PyTorch model, you can convert it to TensorFlow Lite format for edge device deployment using the provided conversion script.

### Prerequisites

Install required dependencies:
```bash
pip install ai-edge-torch tensorflow
```

### Basic Conversion

Convert the trained model to TensorFlow Lite with optimizations:

```bash
python convert_to_tflite.py --output imunet_model
```

This creates two models:
- `imunet_model.tflite` - Standard TensorFlow Lite model (~14MB)
- `imunet_model_quantized.tflite` - Optimized model with 73% size reduction (~4MB)

### Advanced Options

```bash
python convert_to_tflite.py \
    --checkpoint RONIN_torch/Train_out/IMUNet/proposed/checkpoints/checkpoint_best.pt \
    --output my_model \
    --test_list Datasets/proposed/list_test.txt \
    --root_dir Datasets/proposed \
    --batch_size 512
```

### Conversion Features

The conversion script automatically:
1. **Auto-detects output dimensions** from the trained model (2D or 3D)
2. **Converts** PyTorch model to TensorFlow Lite using ai-edge-torch
3. **Applies optimizations** using TensorFlow Lite's default optimization suite
4. **Validates accuracy** by comparing all three models on the test set
5. **Measures performance** across key metrics (velocity RMSE, speed accuracy, etc.)
6. **Saves detailed results** to JSON file for analysis

**NEW: 3D Model Support** - The conversion script now automatically detects whether your trained model outputs 2D or 3D velocities and creates the appropriate TensorFlow Lite model.

### Performance Results

Typical conversion results:
- **Model Size**: 13.97MB (PyTorch) → 14.13MB (TFLite) → 3.82MB (Optimized) 
- **Size Reduction**: 73% smaller optimized model
- **Accuracy**: <2% difference in velocity RMSE for most metrics
- **Speed**: Maintains excellent short-term prediction accuracy

### Android Deployment

The optimized `.tflite` model is ready for Android deployment with NNAPI acceleration:

```java
// Enable NNAPI delegate for NPU acceleration
NnApiDelegate nnApiDelegate = new NnApiDelegate();
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(nnApiDelegate);
Interpreter interpreter = new Interpreter(modelBuffer, options);
```

### Output Files

After conversion:
- `model_name.tflite` - Full precision TensorFlow Lite model
- `model_name_quantized.tflite` - Optimized model (recommended for deployment)
- `model_name_optimization_results.json` - Detailed accuracy comparison

The quantized model provides the best balance of size and accuracy for mobile deployment.

## Android Sensor API Integration

To run inference with the trained TensorFlow Lite model on Android devices, you need to properly read sensor data and apply the correct transformations to match the training data format.

### Required Android Sensors

The IMUNet model requires 6-channel IMU data from these Android sensors:

```java
// Required sensors
private SensorManager sensorManager;
private Sensor accelerometer;  // TYPE_ACCELEROMETER 
private Sensor gyroscope;      // TYPE_GYROSCOPE
private Sensor gameRotation;   // TYPE_GAME_ROTATION_VECTOR

// Initialize sensors
sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
gameRotation = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);

// Register listeners at 200Hz (SENSOR_DELAY_FASTEST)
sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST);
sensorManager.registerListener(this, gameRotation, SensorManager.SENSOR_DELAY_FASTEST);
```

### Data Collection and Preprocessing

#### 1. Raw Sensor Data Collection
```java
@Override
public void onSensorChanged(SensorEvent event) {
    long timestamp = event.timestamp; // nanoseconds
    
    switch (event.sensor.getType()) {
        case Sensor.TYPE_ACCELEROMETER:
            // Raw accelerometer: [x, y, z] in m/s²
            float[] accel = {event.values[0], event.values[1], event.values[2]};
            storeAccelData(timestamp, accel);
            break;
            
        case Sensor.TYPE_GYROSCOPE:
            // Raw gyroscope: [x, y, z] in rad/s
            float[] gyro = {event.values[0], event.values[1], event.values[2]};
            storeGyroData(timestamp, gyro);
            break;
            
        case Sensor.TYPE_GAME_ROTATION_VECTOR:
            // Game rotation vector: [x, y, z, w] (quaternion)
            float[] gameRV = {event.values[0], event.values[1], event.values[2], 
                             event.values.length > 3 ? event.values[3] : 0.0f};
            storeOrientationData(timestamp, gameRV);
            break;
    }
}
```

#### 2. Data Synchronization and Resampling
```java
// Resample all sensor data to uniform 200Hz timeline
public float[][] resampleTo200Hz(long[] timestamps, float[][] data) {
    // Target: 200Hz = 5ms intervals
    long startTime = timestamps[0];
    long endTime = timestamps[timestamps.length - 1];
    long duration = endTime - startTime;
    int targetSamples = (int) (duration / 5_000_000L); // 5ms in nanoseconds
    
    // Use linear interpolation to resample to 200Hz
    float[][] resampled = new float[targetSamples][data[0].length];
    // ... interpolation logic
    return resampled;
}
```

#### 3. Coordinate Frame Transformation
**Critical: Transform sensor data from device frame to global frame using game rotation vector**

```java
public float[][] transformToGlobalFrame(float[][] gyroData, float[][] accelData, 
                                       float[][] gameRotationData) {
    int numSamples = gyroData.length;
    float[][] globalFeatures = new float[numSamples][6];
    
    for (int i = 0; i < numSamples; i++) {
        // Convert game rotation vector to quaternion [w, x, y, z]
        float[] gameRV = gameRotationData[i];
        float[] quat;
        if (gameRV.length == 4) {
            quat = new float[]{gameRV[3], gameRV[0], gameRV[1], gameRV[2]}; // [w,x,y,z]
        } else {
            // If w component missing, compute it
            float x = gameRV[0], y = gameRV[1], z = gameRV[2];
            float w = (float) Math.sqrt(Math.max(0, 1 - x*x - y*y - z*z));
            quat = new float[]{w, x, y, z};
        }
        
        // Transform gyroscope to global frame: q * gyro_quat * q.conj()
        float[] gyroGlobal = quaternionRotateVector(quat, gyroData[i]);
        
        // Transform accelerometer to global frame: q * accel_quat * q.conj()  
        float[] accelGlobal = quaternionRotateVector(quat, accelData[i]);
        
        // Combine into 6-channel feature vector: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
        System.arraycopy(gyroGlobal, 0, globalFeatures[i], 0, 3);
        System.arraycopy(accelGlobal, 0, globalFeatures[i], 3, 3);
    }
    
    return globalFeatures;
}

private float[] quaternionRotateVector(float[] quat, float[] vector) {
    // Rotate 3D vector by quaternion: q * [0,v] * q.conj()
    float qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    float vx = vector[0], vy = vector[1], vz = vector[2];
    
    // Quaternion multiplication: q * [0,v] * q.conj()
    float[] result = new float[3];
    result[0] = vx*(qw*qw + qx*qx - qy*qy - qz*qz) + 2*vy*(qx*qy - qw*qz) + 2*vz*(qx*qz + qw*qy);
    result[1] = 2*vx*(qx*qy + qw*qz) + vy*(qw*qw - qx*qx + qy*qy - qz*qz) + 2*vz*(qy*qz - qw*qx);
    result[2] = 2*vx*(qx*qz - qw*qy) + 2*vy*(qy*qz + qw*qx) + vz*(qw*qw - qx*qx - qy*qy + qz*qz);
    
    return result;
}
```

### Model Inference Pipeline

#### 1. Sliding Window Data Preparation
```java
public float[][][] prepareInferenceWindows(float[][] globalFeatures) {
    int windowSize = 200;  // Model expects 200 timesteps
    int stepSize = 10;     // 10-sample stride (50ms at 200Hz)
    int numWindows = (globalFeatures.length - windowSize) / stepSize + 1;
    
    float[][][] windows = new float[numWindows][1][windowSize * 6]; // Batch x Channels x Time
    
    for (int w = 0; w < numWindows; w++) {
        int startIdx = w * stepSize;
        for (int t = 0; t < windowSize; t++) {
            for (int c = 0; c < 6; c++) {
                windows[w][0][c * windowSize + t] = globalFeatures[startIdx + t][c];
            }
        }
    }
    return windows;
}
```

#### 2. TensorFlow Lite Inference
```java
public float[][] runInference(float[][][] inputWindows) {
    // Auto-detect output dimensions from model (2D or 3D)
    int outputDim = interpreter.getOutputTensor(0).shape()[1]; // 2 for 2D, 3 for 3D
    float[][] predictions = new float[inputWindows.length][outputDim];
    
    for (int i = 0; i < inputWindows.length; i++) {
        // Set input tensor (1, 6, 200)
        interpreter.getInputTensor(0).copyFrom(inputWindows[i]);
        
        // Run inference
        interpreter.run();
        
        // Get output tensor (1, 2) or (1, 3) depending on model
        float[][] output = new float[1][outputDim];
        interpreter.getOutputTensor(0).copyTo(output);
        
        predictions[i][0] = output[0][0]; // vx (m/s)
        predictions[i][1] = output[0][1]; // vy (m/s)
        if (outputDim == 3) {
            predictions[i][2] = output[0][2]; // vz (m/s) for 3D models
        }
    }
    
    return predictions;
}
```

### Data Format Summary

| Stage | Format | Description |
|-------|--------|-------------|
| **Raw Sensors** | Device Frame | Android sensor coordinates |
| **Game Rotation** | `[x,y,z,w]` | Quaternion for orientation |
| **Global Transform** | `[6 channels]` | `[gyro_global_xyz, accel_global_xyz]` |
| **Model Input** | `[1,6,200]` | Batch × Channels × Time |
| **Model Output 2D** | `[1,2]` | `[velocity_x, velocity_y]` in m/s |
| **Model Output 3D** | `[1,3]` | `[velocity_x, velocity_y, velocity_z]` in m/s |

### Key Implementation Notes

1. **Sampling Rate**: Maintain 200Hz uniform sampling for consistency with training data
2. **Coordinate Frames**: Always transform to global frame using game rotation vector
3. **Quaternion Format**: Convert Android `[x,y,z,w]` to standard `[w,x,y,z]` format
4. **Window Overlap**: Use 10-sample stride (50ms) for smooth velocity estimates
5. **Sensor Fusion**: Game rotation vector provides drift-free orientation without magnetometer
6. **3D Support**: Models automatically adapt to 2D or 3D output based on training dataset
7. **Memory Management**: Process windows in batches to avoid memory issues

### Performance Optimization

```java
// Enable NNAPI for hardware acceleration
NnApiDelegate nnApiDelegate = new NnApiDelegate();
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(nnApiDelegate);
options.setNumThreads(4); // Adjust based on device
Interpreter interpreter = new Interpreter(modelBuffer, options);
```

This preprocessing pipeline ensures that your Android sensor data matches exactly the format expected by the trained IMUNet model for accurate velocity estimation.

## References

- **IMUNet Paper**: [Add paper reference when available]
- **ARCore Documentation**: https://developers.google.com/ar
- **RONIN Dataset**: http://ronin.cs.sfu.ca/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **NNAPI Documentation**: https://developer.android.com/ndk/guides/neuralnetworks
