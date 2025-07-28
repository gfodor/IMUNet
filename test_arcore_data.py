#!/usr/bin/env python3

import sys
import os
sys.path.append('RONIN_torch')

from RONIN_torch.utils import ProposedSequence
import pandas as pd

def test_arcore_loading():
    print("Testing ARCore data loading with ProposedSequence...")
    
    # Test with one of our processed sequences
    path = "/Users/gfodor/portal/IMUNet/Datasets/proposed/fold/20250727061722"
    
    try:
        seq = ProposedSequence(path)
        print(f"✅ Successfully loaded sequence from {path}")
        print(f"Features shape: {seq.get_feature().shape}")
        print(f"Targets shape: {seq.get_target().shape}")
        print(f"Aux shape: {seq.get_aux().shape}")
        print(f"Sequence info: {seq.get_meta()}")
        
        # Check the data looks reasonable
        features = seq.get_feature()
        targets = seq.get_target()
        
        print(f"\nData validation:")
        print(f"Feature range - min: {features.min():.3f}, max: {features.max():.3f}")
        print(f"Target range - min: {targets.min():.3f}, max: {targets.max():.3f}")
        print(f"Features mean: {features.mean(axis=0)}")
        print(f"Targets mean: {targets.mean(axis=0)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arcore_loading()