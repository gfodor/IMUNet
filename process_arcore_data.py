#!/usr/bin/env python3
"""
Streamlined ARCore data processing script for IMUNet training.
Only processes the columns required by ProposedSequence:
- time, gyro_x/y/z, acce_x/y/z, pos_x/y/z, ori_w/x/y/z, rv_w/x/y/z
"""

import os
import sys
import numpy as np
import scipy.interpolate
import quaternion
import quaternion.quaternion_time_series
import pandas as pd
from scipy.interpolate import UnivariateSpline


def interpolate_quaternion_linear(quat_data, input_timestamp, output_timestamp):
    """Interpolate quaternion data to new timestamps."""
    n_input = quat_data.shape[0]
    assert input_timestamp.shape[0] == n_input
    assert quat_data.shape[1] == 4
    n_output = output_timestamp.shape[0]

    quat_inter = np.zeros([n_output, 4])
    ptr1 = 0
    ptr2 = 0
    for i in range(n_output):
        if ptr1 >= n_input - 1 or ptr2 >= n_input:
            raise ValueError("Quaternion interpolation failed")
        
        # Forward to the correct interval
        while input_timestamp[ptr1 + 1] < output_timestamp[i]:
            ptr1 += 1
            if ptr1 == n_input - 1:
                break
        while input_timestamp[ptr2] < output_timestamp[i]:
            ptr2 += 1
            if ptr2 == n_input:
                break
        
        # Ensure ptr2 doesn't go out of bounds
        if ptr2 >= n_input:
            ptr2 = n_input - 1
        
        q1 = quaternion.quaternion(*quat_data[ptr1])
        q2 = quaternion.quaternion(*quat_data[ptr2])
        quat_inter[i] = quaternion.as_float_array(
            quaternion.quaternion_time_series.slerp(
                q1, q2, input_timestamp[ptr1], input_timestamp[ptr2], output_timestamp[i]
            )
        )
    return quat_inter


def interpolate_3dvector_linear(input_data, input_timestamp, output_timestamp):
    """Interpolate 3D vector data to new timestamps."""
    assert input_data.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input_data, axis=0, fill_value="extrapolate")
    interpolated = func(output_timestamp)
    return interpolated


def process_arcore_sequence(data_root, skip_front=200, skip_end=200):
    """Process a single ARCore sequence and create the CSV file."""
    print(f'Processing {data_root}')
    
    nano_to_sec = 1000000000.0
    
    # Check if required files exist
    required_files = ['pose.txt', 'acce.txt', 'gyro.txt', 'orientation.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(data_root, file)):
            raise FileNotFoundError(f"Required file {file} not found in {data_root}")
    
    # Load and process pose data
    pose_data = np.genfromtxt(os.path.join(data_root, 'pose.txt'))[skip_front:-skip_end, :]
    
    # Swap tango's orientation from [x,y,z,w] to [w,x,y,z]
    pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]
    
    # Extract position (swap coordinates)
    position = np.zeros([len(pose_data), 3])
    position[:, 0] = pose_data[:, 1]  # x
    position[:, 1] = -pose_data[:, 3]  # -z -> y
    position[:, 2] = pose_data[:, 2]   # y -> z
    
    # Remove duplicates
    unique_ts, unique_inds = np.unique(pose_data[:, 0], return_index=True)
    print(f'Portion of unique records: {unique_inds.shape[0] / pose_data.shape[0]:.3f}')
    pose_data = pose_data[unique_inds, :]
    position = position[unique_inds, :]
    
    output_timestamp = pose_data[:, 0]
    output_samplerate = output_timestamp.shape[0] * nano_to_sec / (output_timestamp[-1] - output_timestamp[0])
    print(f'Original pose sample rate: {output_samplerate:.2f}Hz')
    
    # Resample to 200Hz
    new_length = int(((200) * (output_timestamp[-1] - output_timestamp[0])) / nano_to_sec)
    
    old_indices = np.arange(0, len(output_timestamp))
    new_indices = np.linspace(0, output_timestamp.shape[0] - 1, new_length)
    spl = UnivariateSpline(old_indices, output_timestamp, k=3, s=0)
    new_output_timestamp = spl(new_indices)
    
    # Ensure output timestamps stay within input range
    new_output_timestamp = np.clip(new_output_timestamp, output_timestamp[0], output_timestamp[-1])
    
    # Interpolate position and orientation
    new_position = interpolate_3dvector_linear(position, output_timestamp, new_output_timestamp)
    new_pose_quat = interpolate_quaternion_linear(pose_data[:, -4:], output_timestamp, new_output_timestamp)
    new_pose_quat[0, :] = pose_data[0, -4:]
    
    new_output_samplerate = new_output_timestamp.shape[0] * nano_to_sec / (
        new_output_timestamp[-1] - new_output_timestamp[0])
    print(f'Resampled pose rate: {new_output_samplerate:.2f}Hz')
    
    # Load sensor data
    acce_data = np.genfromtxt(os.path.join(data_root, 'acce.txt'))
    gyro_data = np.genfromtxt(os.path.join(data_root, 'gyro.txt'))
    orientation_data = np.genfromtxt(os.path.join(data_root, 'orientation.txt'))
    
    print(f'Acceleration sample rate: {(acce_data.shape[0] - 1.0) * nano_to_sec / (acce_data[-1, 0] - acce_data[0, 0]):.2f}Hz')
    print(f'Gyroscope sample rate: {(gyro_data.shape[0] - 1.0) * nano_to_sec / (gyro_data[-1, 0] - gyro_data[0, 0]):.2f}Hz')
    print(f'Orientation sample rate: {(orientation_data.shape[0] - 1.0) * nano_to_sec / (orientation_data[-1, 0] - orientation_data[0, 0]):.2f}Hz')
    
    # Interpolate sensor data to new timestamps
    output_gyro = interpolate_3dvector_linear(gyro_data[:, 1:], gyro_data[:, 0], new_output_timestamp)
    output_acce = interpolate_3dvector_linear(acce_data[:, 1:], acce_data[:, 0], new_output_timestamp)
    
    # Process orientation data (swap from x,y,z,w to w,x,y,z)
    orientation_data[:, [1, 2, 3, 4]] = orientation_data[:, [4, 1, 2, 3]]
    output_orientation = interpolate_quaternion_linear(orientation_data[:, 1:], orientation_data[:, 0], new_output_timestamp)
    
    # Create output folder
    output_folder = os.path.join(data_root, 'processed')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Only keep columns needed for training
    column_list = ['time', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z', 
                   'pos_x', 'pos_y', 'pos_z', 'ori_w', 'ori_x', 'ori_y', 'ori_z',
                   'rv_w', 'rv_x', 'rv_y', 'rv_z']
    
    # Combine all data
    data_mat = np.concatenate([
        new_output_timestamp[:, None],    # time
        output_gyro,                      # gyro_x, gyro_y, gyro_z
        output_acce,                      # acce_x, acce_y, acce_z
        new_position,                     # pos_x, pos_y, pos_z
        new_pose_quat,                    # ori_w, ori_x, ori_y, ori_z
        output_orientation                # rv_w, rv_x, rv_y, rv_z
    ], axis=1)
    
    # Create DataFrame and save
    data_pandas = pd.DataFrame(data_mat, columns=column_list)
    csv_path = os.path.join(output_folder, 'data.csv')
    data_pandas.to_csv(csv_path, index=False)
    
    # Calculate sequence length
    length = (data_pandas['time'].values[-1] - data_pandas['time'].values[0]) / nano_to_sec
    hertz = data_pandas.shape[0] / length
    
    print(f'Dataset written to {csv_path}')
    print(f'Length: {length:.2f}s, Final sample rate: {hertz:.2f}Hz')
    print(f'Data shape: {data_pandas.shape}')
    
    return length, hertz


def main():
    """Process all ARCore sequences in the imu_data directory."""
    # Set up paths
    current_dir = os.getcwd()
    imu_data_root = os.path.join(current_dir, 'imu_data')
    
    if not os.path.exists(imu_data_root):
        raise FileNotFoundError(f"imu_data directory not found at {imu_data_root}")
    
    print(f"Processing ARCore data in: {imu_data_root}")
    
    # Find all sequence directories
    sequence_paths = []
    device_folders = [d for d in os.listdir(imu_data_root) if os.path.isdir(os.path.join(imu_data_root, d))]
    
    for device in device_folders:
        device_path = os.path.join(imu_data_root, device)
        date_folders = [d for d in os.listdir(device_path) if os.path.isdir(os.path.join(device_path, d))]
        
        for date in date_folders:
            sequence_path = os.path.join(device_path, date)
            sequence_paths.append((f"{device}_{date}", sequence_path))
    
    print(f"Found {len(sequence_paths)} sequences:")
    for name, path in sequence_paths:
        print(f"  {name}: {path}")
    
    # Process each sequence
    total_length = 0.0
    processed_sequences = []
    
    for seq_name, seq_path in sequence_paths:
        try:
            length, hertz = process_arcore_sequence(seq_path)
            total_length += length
            processed_sequences.append(seq_name)
            print(f"✅ {seq_name}: {length:.2f}s, {hertz:.2f}Hz")
        except Exception as e:
            print(f"❌ Error processing {seq_name}: {e}")
            continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {len(processed_sequences)}/{len(sequence_paths)} sequences")
    print(f"Total length: {total_length:.2f}s ({total_length/60.0:.2f}min)")
    
    # Create dataset list files
    dataset_root = os.path.join(current_dir, 'Datasets', 'proposed')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    
    # Create list file for training
    train_list_path = os.path.join(dataset_root, 'list_train.txt')
    test_list_path = os.path.join(dataset_root, 'list_test.txt')
    
    # Split sequences into train/test (80/20 split)
    split_idx = int(len(processed_sequences) * 0.8)
    train_sequences = processed_sequences[:split_idx]
    test_sequences = processed_sequences[split_idx:]
    
    with open(train_list_path, 'w') as f:
        for seq_name in train_sequences:
            device, date = seq_name.split('_', 1)
            f.write(f"{device}/{date},arcore\n")
    
    with open(test_list_path, 'w') as f:
        for seq_name in test_sequences:
            device, date = seq_name.split('_', 1)
            f.write(f"{device}/{date},arcore\n")
    
    print(f"\nDataset lists created:")
    print(f"  Train: {train_list_path} ({len(train_sequences)} sequences)")
    print(f"  Test:  {test_list_path} ({len(test_sequences)} sequences)")
    
    return processed_sequences


if __name__ == '__main__':
    processed = main()