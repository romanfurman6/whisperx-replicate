#!/usr/bin/env python3
"""
Pre-download WhisperX models during Docker build to reduce cold start time.
This script downloads the Whisper model and caches it in the container.
"""

import os
import torch
import whisperx

# Configuration
MODELS_TO_CACHE = ["large-v3"]
DEVICE = "cpu"  # Use CPU during build to avoid GPU requirement
COMPUTE_TYPE = "int8"  # Use int8 for smaller model size during caching

def download_model(model_name):
    """Download and cache a WhisperX model."""
    print(f"\n{'='*60}")
    print(f"Downloading WhisperX model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load the model (this will download and cache it)
        model = whisperx.load_model(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=None,  # Use default cache location
            language=None
        )
        
        print(f"✓ Successfully cached model: {model_name}")
        
        # Clean up to free memory
        del model
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to cache model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to download all models."""
    print("\n" + "="*60)
    print("WhisperX Model Pre-Download Script")
    print("="*60)
    
    success_count = 0
    total_count = len(MODELS_TO_CACHE)
    
    for model_name in MODELS_TO_CACHE:
        if download_model(model_name):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Model Download Summary: {success_count}/{total_count} successful")
    print("="*60 + "\n")
    
    if success_count < total_count:
        exit(1)  # Exit with error if any download failed


if __name__ == "__main__":
    main()

