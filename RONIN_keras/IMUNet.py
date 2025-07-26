"""
IMUNet implementation for Keras/TensorFlow
Converted from PyTorch implementation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def IMUNet(input_shape, num_classes):
    """
    Create IMUNet model for Keras
    
    Args:
        input_shape: Input shape (6, 200) for IMU data
        num_classes: Number of output classes (typically 2 for vx, vy)
    
    Returns:
        Keras model
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # Input block
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Depthwise separable conv blocks (simplified)
    def depthwise_sep_conv(x, filters, kernel_size=3, strides=1, padding='same'):
        x = layers.DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        x = layers.ELU()(x)
        x = layers.Conv1D(filters, kernel_size=1, use_bias=False)(x)
        x = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        x = layers.ELU()(x)
        return x
    
    def residual_block(x, filters, strides=1, downsample=False):
        residual = x
        
        # Main path
        out = depthwise_sep_conv(x, filters, strides=strides)
        
        # Downsample residual if needed
        if downsample or strides != 1:
            residual = layers.Conv1D(filters, kernel_size=1, strides=strides, use_bias=False)(residual)
            residual = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(residual)
        
        # Add residual
        out = layers.Add()([out, residual])
        out = layers.ELU()(out)
        
        return out
    
    # Conv blocks with residual connections
    x = residual_block(x, 64, strides=1)
    x = residual_block(x, 64, strides=1)
    
    x = residual_block(x, 64, strides=1, downsample=True)
    x = residual_block(x, 64, strides=1)
    
    x = residual_block(x, 128, strides=2, downsample=True)
    x = residual_block(x, 128, strides=1)
    
    x = residual_block(x, 256, strides=2, downsample=True)
    x = residual_block(x, 256, strides=1)
    
    x = residual_block(x, 512, strides=2, downsample=True)
    x = residual_block(x, 512, strides=1)
    
    x = residual_block(x, 1024, strides=2, downsample=True)
    x = residual_block(x, 1024, strides=1)
    
    # Global pooling before final layers
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(400, activation='relu')(x)
    
    # Final dense layer
    outputs = layers.Dense(num_classes)(x)
    
    model = keras.Model(inputs, outputs, name='IMUNet')
    
    return model

# For compatibility with the existing code
def create_imunet_model(input_shape, num_classes):
    """Compatibility function"""
    return IMUNet(input_shape, num_classes)