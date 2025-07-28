#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import *
from main import get_dataset, get_dataset_from_list
import argparse

def test_dataset_loading():
    print("Starting dataset loading test...")
    
    # Mock args object
    class Args:
        def __init__(self):
            self.root_dir = "/Users/gfodor/portal/IMUNet/Datasets/ridi/data_publish_v2"
            self.train_list = "/Users/gfodor/portal/IMUNet/Datasets/ridi/data_publish_v2/list_train_small.txt"
            self.cache_path = None
            self.step_size = 10
            self.window_size = 200
            self.max_ori_error = 20.0
            self.dataset = 'ridi'
    
    args = Args()
    
    try:
        print("Testing get_dataset_from_list...")
        dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        print("Testing DataLoader creation...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        print("DataLoader created successfully")
        
        print("Testing single batch...")
        for i, (feat, targ, seq_id, frame_id) in enumerate(train_loader):
            print(f"Batch {i}: feat shape {feat.shape}, targ shape {targ.shape}")
            if i >= 2:  # Only test first few batches
                break
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset_loading()