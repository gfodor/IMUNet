#!/usr/bin/env python3
"""
IMUNet PyTorch to TensorFlow Lite Conversion Script

This script converts the trained IMUNet PyTorch model to TensorFlow Lite format
using ai-edge-torch and validates that the conversion maintains model accuracy.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

# Add the RONIN_torch directory to path to import modules
sys.path.insert(0, './RONIN_torch')

from IMUNet import IMUNet
from main import get_dataset_from_list, compute_velocity_metrics
from torch.utils.data import DataLoader
import argparse

try:
    import ai_edge_torch
except ImportError:
    print("ERROR: ai_edge_torch is not installed. Please install it with:")
    print("pip install ai-edge-torch")
    sys.exit(1)


class IMUNetTensorFlowLite:
    """Wrapper class for TensorFlow Lite IMUNet model inference"""
    
    def __init__(self, tflite_model_path):
        """Initialize TFLite interpreter"""
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"TFLite model loaded from {tflite_model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
    def predict(self, x):
        """Run inference on input data"""
        # Ensure input is float32
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = x.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output


def load_pytorch_model(checkpoint_path, device='cpu'):
    """Load the trained PyTorch IMUNet model"""
    print(f"Loading PyTorch model from {checkpoint_path}")
    
    # Create model instance with same parameters used in training
    model = IMUNet(
        num_classes=2, 
        input_size=(1, 6, 200), 
        sampling_rate=200, 
        num_T=32, 
        num_S=64, 
        hidden=64, 
        dropout_rate=0.5
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"PyTorch model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


def convert_to_tflite(pytorch_model, output_path, sample_input_shape=(1, 6, 200)):
    """Convert PyTorch model to TensorFlow Lite using ai-edge-torch"""
    print("Converting PyTorch model to TensorFlow Lite...")
    
    # Create sample input for tracing
    sample_input = torch.randn(sample_input_shape, dtype=torch.float32)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Test PyTorch model with sample input
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input)
        print(f"PyTorch output shape: {pytorch_output.shape}")
    
    try:
        # Convert using ai-edge-torch
        edge_model = ai_edge_torch.convert(pytorch_model, (sample_input,))
        
        # Save the TensorFlow Lite model
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        edge_model.export(os.path.abspath(output_path))
        
        print(f"TensorFlow Lite model saved to {output_path}")
        
        # Verify the conversion by loading and testing
        tflite_model = IMUNetTensorFlowLite(output_path)
        tflite_output = tflite_model.predict(sample_input.numpy())
        
        # Compare outputs
        diff = np.abs(pytorch_output.numpy() - tflite_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Conversion verification:")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-5:
            print("✅ Conversion successful - outputs match within tolerance")
        else:
            print("⚠️  Large difference detected - conversion may have issues")
            
        return tflite_model
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise


def run_pytorch_test(network, data_loader, device, eval_mode=True):
    """Run test evaluation using PyTorch model (adapted from main.py)"""
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    
    batch_count = 0
    total_batches = len(data_loader)
    print(f"Running PyTorch test on {total_batches} batches...")
    
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        if batch_count % 50 == 0:
            print(f"PyTorch batch {batch_count}/{total_batches}")
        batch_count += 1
        
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    
    print("Concatenating PyTorch results...")
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    print(f"PyTorch test completed. Results shape: targets {targets_all.shape}, predictions {preds_all.shape}")
    return targets_all, preds_all


def run_tflite_test(tflite_model, data_loader, device='cpu'):
    """Run test evaluation using TensorFlow Lite model"""
    targets_all = []
    preds_all = []
    
    batch_count = 0
    total_batches = len(data_loader)
    print(f"Running TFLite inference on {total_batches} batches...")
    
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        if batch_count % 50 == 0:
            print(f"TFLite batch {batch_count}/{total_batches}")
        batch_count += 1
        
        # Convert to numpy for TFLite inference
        feat_np = feat.detach().cpu().numpy().astype(np.float32)
        
        # Run inference batch by batch
        batch_preds = []
        for i in range(feat_np.shape[0]):
            sample = feat_np[i:i+1]  # Keep batch dimension
            pred = tflite_model.predict(sample)
            batch_preds.append(pred)
        
        pred = np.concatenate(batch_preds, axis=0)
        
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    
    print("Concatenating TFLite results...")
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    print(f"TFLite test completed. Results shape: targets {targets_all.shape}, predictions {preds_all.shape}")
    
    return targets_all, preds_all


def compare_models(pytorch_model, tflite_model, test_loader, device='cpu'):
    """Compare PyTorch and TensorFlow Lite model performance"""
    print("\n" + "="*60)
    print("COMPARING PYTORCH VS TENSORFLOW LITE MODELS")
    print("="*60)
    
    # Run PyTorch evaluation
    print("\n🔥 Running PyTorch model evaluation...")
    pytorch_model.eval()
    pytorch_targets, pytorch_preds = run_pytorch_test(pytorch_model, test_loader, device, eval_mode=True)
    pytorch_metrics = compute_velocity_metrics(pytorch_preds, pytorch_targets)
    
    # Run TensorFlow Lite evaluation  
    print("\n📱 Running TensorFlow Lite model evaluation...")
    tflite_targets, tflite_preds = run_tflite_test(tflite_model, test_loader, device)
    tflite_metrics = compute_velocity_metrics(tflite_preds, tflite_targets)
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 60)
    
    key_metrics = [
        'velocity_rmse', 'velocity_mae', 'velocity_mse',
        'speed_rmse', 'speed_mae', 'direction_error_deg',
        'short_term_100ms_rmse', 'short_term_100ms_mae'
    ]
    
    print(f"{'Metric':<25} {'PyTorch':<15} {'TFLite':<15} {'Diff':<15} {'Rel %':<10}")
    print("-" * 80)
    
    all_diffs_small = True
    for metric in key_metrics:
        pytorch_val = pytorch_metrics[metric]
        tflite_val = tflite_metrics[metric]
        diff = abs(tflite_val - pytorch_val)
        rel_diff = (diff / pytorch_val * 100) if pytorch_val != 0 else 0
        
        print(f"{metric:<25} {pytorch_val:<15.6f} {tflite_val:<15.6f} {diff:<15.6f} {rel_diff:<10.2f}")
        
        # Check if difference is significant (more than 1% relative difference)
        if rel_diff > 1.0:
            all_diffs_small = False
    
    print("-" * 80)
    
    # Overall assessment
    if all_diffs_small:
        print("✅ CONVERSION SUCCESSFUL: All metrics match within 1% tolerance")
        print("   The TensorFlow Lite model maintains the same accuracy as PyTorch")
    else:
        print("⚠️  SIGNIFICANT DIFFERENCES DETECTED")
        print("   Some metrics show >1% difference - please review conversion")
    
    # Compute model size comparison
    pytorch_size = sum(p.numel() * 4 for p in pytorch_model.parameters()) / (1024*1024)  # MB
    
    print(f"\n💾 MODEL SIZE COMPARISON:")
    print(f"   PyTorch parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    print(f"   PyTorch model size: ~{pytorch_size:.2f} MB")
    
    # Return both metric dictionaries for further analysis if needed
    return pytorch_metrics, tflite_metrics


def main():
    parser = argparse.ArgumentParser(description='Convert IMUNet PyTorch model to TensorFlow Lite')
    parser.add_argument('--checkpoint', type=str, 
                        default='RONIN_torch/Train_out/IMUNet/proposed/checkpoints/checkpoint_best.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='imunet_model.tflite',
                        help='Output TensorFlow Lite model path')
    parser.add_argument('--test_list', type=str, 
                        default='Datasets/proposed/list_test.txt',
                        help='Path to test list file')
    parser.add_argument('--root_dir', type=str, 
                        default='Datasets/proposed',
                        help='Root directory of test data')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use for PyTorch evaluation')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.test_list):
        print(f"❌ Test list file not found: {args.test_list}")
        sys.exit(1)
        
    if not os.path.exists(args.root_dir):
        print(f"❌ Root directory not found: {args.root_dir}")
        sys.exit(1)
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model(args.checkpoint, device)
    pytorch_model = pytorch_model.to(device)
    
    # Convert to TensorFlow Lite
    tflite_model = convert_to_tflite(pytorch_model.cpu(), args.output)
    
    # Load test dataset (using simplified args structure)
    class TestArgs:
        def __init__(self):
            self.dataset = 'proposed'
            self.step_size = 10
            self.window_size = 200
            self.cache_path = None
            self.max_ori_error = 20.0
    
    test_args = TestArgs()
    
    print(f"\nLoading test dataset from {args.test_list}...")
    test_dataset = get_dataset_from_list(args.root_dir, args.test_list, test_args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Compare model performance
    pytorch_metrics, tflite_metrics = compare_models(pytorch_model, tflite_model, test_loader, device)
    
    # Save results
    results = {
        'pytorch_metrics': {k: float(v) for k, v in pytorch_metrics.items()},
        'tflite_metrics': {k: float(v) for k, v in tflite_metrics.items()},
        'conversion_info': {
            'checkpoint_path': args.checkpoint,
            'tflite_output': args.output,
            'test_samples': len(test_dataset),
            'device_used': str(device)
        }
    }
    
    results_file = args.output.replace('.tflite', '_conversion_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Conversion results saved to: {results_file}")
    print(f"🚀 TensorFlow Lite model ready: {args.output}")
    print("\nConversion complete! Your TFLite model is ready for edge deployment.")


if __name__ == '__main__':
    main()