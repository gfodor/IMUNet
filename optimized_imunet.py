"""
Optimized IMUNet Implementation for Mobile Deployment
Parameterized architecture with multiple optimization variants
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class OptimizedIMUNet:
    """
    Parameterized IMUNet with optimization variants for mobile deployment
    """
    
    def __init__(self, config=None):
        """
        Initialize with configuration
        
        Args:
            config: Dictionary with model parameters or string preset
        """
        if config is None:
            config = 'fast'
            
        if isinstance(config, str):
            self.config = self.get_preset_config(config)
        else:
            self.config = config
            
        self.validate_config()
    
    def get_preset_config(self, preset):
        """Get predefined configuration presets"""
        presets = {
            'ultra_fast': {
                'base_channels': 16,
                'channel_multipliers': [1, 2, 4, 8],  # 16, 32, 64, 128
                'num_blocks': 4,
                'window_size': 64,
                'input_channels': 6,
                'num_classes': 3,
                'dropout_rate': 0.2,
                'activation': 'relu',
                'use_residual': True,
                'use_global_pool': True,
                'dense_units': 128
            },
            'fast': {
                'base_channels': 32,
                'channel_multipliers': [1, 2, 4, 8],  # 32, 64, 128, 256
                'num_blocks': 5,
                'window_size': 100,
                'input_channels': 6,
                'num_classes': 3,
                'dropout_rate': 0.3,
                'activation': 'relu',
                'use_residual': True,
                'use_global_pool': True,
                'dense_units': 256
            },
            'balanced': {
                'base_channels': 32,
                'channel_multipliers': [1, 2, 4, 8, 16],  # 32, 64, 128, 256, 512
                'num_blocks': 5,
                'window_size': 128,
                'input_channels': 6,
                'num_classes': 3,
                'dropout_rate': 0.4,
                'activation': 'relu',
                'use_residual': True,
                'use_global_pool': True,
                'dense_units': 400
            },
            'accurate': {
                'base_channels': 64,
                'channel_multipliers': [1, 2, 4, 8],  # 64, 128, 256, 512
                'num_blocks': 6,
                'window_size': 200,
                'input_channels': 6,
                'num_classes': 3,
                'dropout_rate': 0.5,
                'activation': 'relu',
                'use_residual': True,
                'use_global_pool': True,
                'dense_units': 512
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = [
            'base_channels', 'channel_multipliers', 'num_blocks', 
            'window_size', 'input_channels', 'num_classes'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def depthwise_separable_conv(self, x, filters, kernel_size=3, strides=1, name_prefix="ds_conv"):
        """
        Depthwise separable convolution block optimized for mobile
        """
        # Depthwise convolution
        x = layers.DepthwiseConv1D(
            kernel_size=kernel_size, 
            strides=strides, 
            padding='same', 
            use_bias=False,
            name=f"{name_prefix}_depthwise"
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-5, 
            momentum=0.1,
            name=f"{name_prefix}_bn1"
        )(x)
        x = layers.Activation(self.config.get('activation', 'relu'), name=f"{name_prefix}_act1")(x)
        
        # Pointwise convolution
        x = layers.Conv1D(
            filters, 
            kernel_size=1, 
            use_bias=False,
            name=f"{name_prefix}_pointwise"
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-5, 
            momentum=0.1,
            name=f"{name_prefix}_bn2"
        )(x)
        x = layers.Activation(self.config.get('activation', 'relu'), name=f"{name_prefix}_act2")(x)
        
        return x
    
    def residual_block(self, x, filters, strides=1, block_id=0):
        """
        Residual block with depthwise separable convolutions
        """
        residual = x
        name_prefix = f"block_{block_id}"
        
        # Main path
        out = self.depthwise_separable_conv(
            x, filters, strides=strides, 
            name_prefix=f"{name_prefix}_main"
        )
        
        # Residual path adjustment if needed
        if strides != 1 or x.shape[-1] != filters:
            residual = layers.Conv1D(
                filters, 
                kernel_size=1, 
                strides=strides, 
                use_bias=False,
                name=f"{name_prefix}_residual_conv"
            )(residual)
            residual = layers.BatchNormalization(
                epsilon=1e-5, 
                momentum=0.1,
                name=f"{name_prefix}_residual_bn"
            )(residual)
        
        # Add residual connection if enabled
        if self.config.get('use_residual', True):
            out = layers.Add(name=f"{name_prefix}_add")([out, residual])
        
        out = layers.Activation(
            self.config.get('activation', 'relu'), 
            name=f"{name_prefix}_final_act"
        )(out)
        
        return out
    
    def build_model(self):
        """
        Build the optimized IMUNet model
        """
        input_shape = (self.config['window_size'], self.config['input_channels'])
        inputs = keras.Input(shape=input_shape, name='imu_input')
        
        # Input block - initial convolution
        x = layers.Conv1D(
            self.config['base_channels'], 
            kernel_size=7, 
            strides=2, 
            padding='same', 
            use_bias=False,
            name='input_conv'
        )(inputs)
        x = layers.BatchNormalization(
            epsilon=1e-5, 
            momentum=0.1,
            name='input_bn'
        )(x)
        x = layers.Activation(self.config.get('activation', 'relu'), name='input_act')(x)
        x = layers.MaxPooling1D(
            pool_size=3, 
            strides=2, 
            padding='same',
            name='input_pool'
        )(x)
        
        # Main conv blocks
        current_channels = self.config['base_channels']
        
        for i, multiplier in enumerate(self.config['channel_multipliers'][:self.config['num_blocks']]):
            target_channels = self.config['base_channels'] * multiplier
            
            # First block in each stage may have stride > 1
            stride = 2 if i > 0 and target_channels > current_channels else 1
            
            x = self.residual_block(
                x, 
                target_channels, 
                strides=stride, 
                block_id=i*2
            )
            
            # Second block in each stage
            if i < len(self.config['channel_multipliers']) - 1:
                x = self.residual_block(
                    x, 
                    target_channels, 
                    strides=1, 
                    block_id=i*2+1
                )
            
            current_channels = target_channels
        
        # Global pooling or flatten
        if self.config.get('use_global_pool', True):
            x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        else:
            x = layers.Flatten(name='flatten')(x)
        
        # Dropout for regularization
        if self.config.get('dropout_rate', 0) > 0:
            x = layers.Dropout(
                self.config['dropout_rate'], 
                name='dropout'
            )(x)
        
        # Dense layer(s)
        dense_units = self.config.get('dense_units', 256)
        if dense_units > 0:
            x = layers.Dense(
                dense_units, 
                activation=self.config.get('activation', 'relu'),
                name='dense'
            )(x)
            
            if self.config.get('dropout_rate', 0) > 0:
                x = layers.Dropout(
                    self.config['dropout_rate'] * 0.5, 
                    name='final_dropout'
                )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.config['num_classes'], 
            name='velocity_output'
        )(x)
        
        model = keras.Model(inputs, outputs, name=f"OptimizedIMUNet")
        
        return model
    
    def get_model_info(self):
        """Get model information and statistics"""
        model = self.build_model()
        
        # Calculate parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        # Estimate FLOPs (rough calculation)
        # This is a simplified estimation
        input_size = self.config['window_size'] * self.config['input_channels']
        estimated_flops = total_params * input_size * 2  # Rough estimate
        
        info = {
            'config': self.config,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_flops': estimated_flops,
            'input_shape': (self.config['window_size'], self.config['input_channels']),
            'output_shape': (self.config['num_classes'],)
        }
        
        return info


def create_optimized_model(preset='fast', custom_config=None):
    """
    Convenient function to create optimized IMUNet model
    
    Args:
        preset: Preset configuration ('ultra_fast', 'fast', 'balanced', 'accurate')
        custom_config: Custom configuration dictionary (overrides preset)
    
    Returns:
        Keras model
    """
    config = custom_config if custom_config is not None else preset
    imunet = OptimizedIMUNet(config)
    return imunet.build_model()


def compare_variants():
    """
    Compare different model variants
    """
    variants = ['ultra_fast', 'fast', 'balanced', 'accurate']
    
    print("IMUNet Architecture Comparison")
    print("=" * 60)
    print(f"{'Variant':<12} {'Params':<10} {'Est. FLOPs':<12} {'Window':<8} {'Channels':<10}")
    print("-" * 60)
    
    for variant in variants:
        imunet = OptimizedIMUNet(variant)
        info = imunet.get_model_info()
        
        max_channels = info['config']['base_channels'] * max(info['config']['channel_multipliers'])
        
        print(f"{variant:<12} {info['total_params']:<10,} {info['estimated_flops']:<12,} "
              f"{info['config']['window_size']:<8} {max_channels:<10}")


if __name__ == "__main__":
    # Example usage and comparison
    compare_variants()
    
    # Create a fast model for testing
    model = create_optimized_model('fast')
    model.summary()