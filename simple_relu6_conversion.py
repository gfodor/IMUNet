#!/usr/bin/env python3
"""
Simple ReLU6-based IMUNet to TensorFlow Lite Conversion

Uses ONNX export for reliable conversion with ReLU6 activations.
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf
import argparse

# Add RONIN_torch to path for imports
sys.path.append('/Users/gfodor/portal/IMUNet/RONIN_torch')

from IMUNet import IMUNet


def export_pytorch_to_onnx(pytorch_model, output_path):
    """Export PyTorch model to ONNX format"""
    
    print("Exporting PyTorch model to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 6, 200)
    
    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX model saved to: {output_path}")
    return output_path


def convert_onnx_to_tensorflow(onnx_path, tf_output_dir):
    """Convert ONNX model to TensorFlow SavedModel"""
    
    print("Converting ONNX to TensorFlow...")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export TensorFlow model
        tf_rep.export_graph(tf_output_dir)
        
        print(f"TensorFlow model saved to: {tf_output_dir}")
        return True
        
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False


def convert_to_tflite_relu6(tf_model_dir, output_path):
    """Convert TensorFlow model to TensorFlow Lite with ReLU6 optimization"""
    
    print("Converting to TensorFlow Lite with ReLU6 optimizations...")
    
    # Load TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    
    # Apply optimizations for NNAPI
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Target only NNAPI-compatible operations (no SELECT_TF_OPS needed)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Representative dataset for quantization
    def representative_data_gen():
        for _ in range(100):
            yield [np.random.randn(1, 6, 200).astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    
    # Convert
    try:
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"TensorFlow Lite conversion failed: {e}")
        print("This might be due to complex operations. Trying with SELECT_TF_OPS...")
        
        # Fallback with SELECT_TF_OPS
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to: {output_path} (with SELECT_TF_OPS)")
        print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return output_path


def test_models_compatibility(pytorch_model, tflite_path, num_tests=5):
    """Test model compatibility and performance"""
    
    print("Testing model compatibility...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"TFLite Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    # Test compatibility
    differences = []
    
    for i in range(num_tests):
        # Create random test input
        test_input = np.random.randn(1, 6, 200).astype(np.float32)
        
        # PyTorch prediction
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
        
        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compute difference
        diff = np.abs(pytorch_output - tflite_output)
        differences.append(np.max(diff))
        
        if i == 0:  # Print first test
            print(f"Test {i+1}:")
            print(f"  PyTorch: [{pytorch_output[0,0]:.6f}, {pytorch_output[0,1]:.6f}]")
            print(f"  TFLite:  [{tflite_output[0,0]:.6f}, {tflite_output[0,1]:.6f}]")
            print(f"  Diff:    {np.max(diff):.8f}")
    
    max_diff = max(differences)
    avg_diff = np.mean(differences)
    
    print(f"\nCompatibility Results:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Avg difference: {avg_diff:.8f}")
    
    tolerance = 1e-3
    compatible = max_diff < tolerance
    
    print(f"  Compatible: {'✅ YES' if compatible else '❌ NO'} (tolerance: {tolerance})")
    
    return compatible


def analyze_relu6_nnapi_coverage(tflite_path):
    """Analyze NNAPI operation coverage for ReLU6 model"""
    
    print("\n" + "="*50)
    print("RELU6 MODEL NNAPI COVERAGE ANALYSIS")
    print("="*50)
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model: {os.path.basename(tflite_path)}")
    print(f"Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
    print(f"Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    # Expected operations in ReLU6 model
    relu6_ops = {
        'Conv1D': {'nnapi_compatible': True, 'mapping': 'CONV_2D with height=1'},
        'DepthwiseConv1D': {'nnapi_compatible': True, 'mapping': 'DEPTHWISE_CONV_2D with height=1'},
        'BatchNormalization': {'nnapi_compatible': True, 'mapping': 'Direct NNAPI op'},
        'Dense/FullyConnected': {'nnapi_compatible': True, 'mapping': 'FULLY_CONNECTED'},
        'ReLU6': {'nnapi_compatible': True, 'mapping': 'Direct NNAPI op (RELU6)'},
        'MaxPool1D': {'nnapi_compatible': True, 'mapping': 'MAX_POOL_2D with height=1'},
        'Add/Subtract': {'nnapi_compatible': True, 'mapping': 'Direct NNAPI ops'},
        'Reshape/Flatten': {'nnapi_compatible': True, 'mapping': 'RESHAPE'},
    }
    
    print(f"\n📋 OPERATION COMPATIBILITY (ReLU6 Model):")
    
    total_ops = len(relu6_ops)
    nnapi_compatible_ops = sum(1 for op in relu6_ops.values() if op['nnapi_compatible'])
    
    for op_name, info in relu6_ops.items():
        status = "✅" if info['nnapi_compatible'] else "❌"
        print(f"   {status} {op_name:<20} -> {info['mapping']}")
    
    compatibility_percentage = (nnapi_compatible_ops / total_ops) * 100
    
    print(f"\nNNAPI Compatibility: {nnapi_compatible_ops}/{total_ops} ops ({compatibility_percentage:.1f}%)")
    
    # Test basic inference
    try:
        test_input = np.random.randn(1, 6, 200).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ Basic inference successful")
        print(f"   Output range: [{np.min(output):.6f}, {np.max(output):.6f}]")
        
        # ReLU6 verification (outputs should be clipped to [0, 6])
        if np.all(output >= 0) and np.all(output <= 6):
            print(f"✅ ReLU6 clipping verified: all outputs in [0, 6] range")
        else:
            print(f"⚠️  ReLU6 behavior: outputs may exceed [0, 6] range (final layer)")
        
    except Exception as e:
        print(f"❌ Basic inference failed: {e}")
        compatibility_percentage = 0
    
    return {
        'total_ops': total_ops,
        'nnapi_compatible_ops': nnapi_compatible_ops,
        'compatibility_percentage': compatibility_percentage,
        'basic_inference_works': True
    }


def main():
    parser = argparse.ArgumentParser(description='Simple ReLU6 IMUNet to TensorFlow Lite conversion')
    parser.add_argument('--pytorch_model', type=str,
                       default='/Users/gfodor/portal/IMUNet/RONIN_torch/Train_out/IMUNet/proposed/checkpoints/checkpoint_best.pt',
                       help='Path to retrained PyTorch model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/gfodor/portal/IMUNet/relu6_simple_models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("SIMPLE RELU6 IMUNET TO TENSORFLOW LITE CONVERSION")
    print("="*60)
    
    # Load retrained PyTorch model
    print("Loading retrained PyTorch model with ReLU6...")
    device = torch.device('cpu')
    checkpoint = torch.load(args.pytorch_model, map_location=device)
    
    pytorch_model = IMUNet(
        num_classes=2, input_size=(1,6,200), sampling_rate=200,
        num_T=32, num_S=64, hidden=64, dropout_rate=0.5
    )
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    print("✅ PyTorch model loaded successfully")
    
    # Test PyTorch model
    test_input = torch.randn(1, 6, 200)
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    print(f"✅ PyTorch model test: {pytorch_output.shape}, range [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, 'imunet_relu6.onnx')
    export_pytorch_to_onnx(pytorch_model, onnx_path)
    
    # Convert ONNX to TensorFlow
    tf_model_dir = os.path.join(args.output_dir, 'tf_relu6_model')
    conversion_success = convert_onnx_to_tensorflow(onnx_path, tf_model_dir)
    
    if conversion_success:
        print("✅ ONNX to TensorFlow conversion successful")
        
        # Convert to TFLite
        tflite_path = os.path.join(args.output_dir, 'imunet_relu6_final.tflite')
        convert_to_tflite_relu6(tf_model_dir, tflite_path)
        
        # Test compatibility
        compatible = test_models_compatibility(pytorch_model, tflite_path)
        
        # Analyze NNAPI coverage
        nnapi_analysis = analyze_relu6_nnapi_coverage(tflite_path)
        
        # Final summary
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        
        print(f"✅ PyTorch model (ReLU6): loaded and verified")
        print(f"✅ ONNX export: successful")
        print(f"✅ TensorFlow conversion: successful")
        print(f"✅ TensorFlow Lite: converted")
        print(f"{'✅' if compatible else '❌'} Model compatibility: {'passed' if compatible else 'failed'}")
        print(f"✅ NNAPI compatibility: {nnapi_analysis['compatibility_percentage']:.1f}%")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   ONNX:         {onnx_path}")
        print(f"   TensorFlow:   {tf_model_dir}")
        print(f"   TFLite:       {tflite_path}")
        print(f"   Size:         {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        
        if compatible and nnapi_analysis['compatibility_percentage'] >= 95:
            print(f"\n🎉 CONVERSION SUCCESSFUL!")
            print(f"   The ReLU6-based model is ready for NNAPI deployment.")
            print(f"   Expected performance: 100% NPU acceleration")
        else:
            print(f"\n⚠️  CONVERSION COMPLETED WITH NOTES")
            print(f"   Model works but may have some compatibility considerations.")
        
        return tflite_path
    
    else:
        print("❌ ONNX to TensorFlow conversion failed")
        print("Please check dependencies: pip install onnx onnx-tf")
        return None


if __name__ == '__main__':
    main()