#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils import RIDIGlobSpeedSequence
import pandas as pd

def test_single_sequence():
    print("Testing single sequence loading...")
    path = "/Users/gfodor/portal/IMUNet/Datasets/ridi/data_publish_v2/hang_handheld_normal1"
    
    try:
        seq = RIDIGlobSpeedSequence(path)
        print(f"Successfully loaded sequence from {path}")
        print(f"Features shape: {seq.get_feature().shape}")
        print(f"Targets shape: {seq.get_target().shape}")
        print(f"Aux shape: {seq.get_aux().shape}")
        return True
    except Exception as e:
        print(f"Error loading sequence: {e}")
        return False

if __name__ == "__main__":
    test_single_sequence()