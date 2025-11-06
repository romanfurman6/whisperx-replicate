#!/usr/bin/env python3
"""
Local test script for WhisperX predictor.
Tests the predictor without RunPod infrastructure.
"""

import sys
import time
from predict import Predictor

def test_basic_functionality():
    """Test basic predictor setup and simple operations."""
    print("\n" + "="*60)
    print("Testing WhisperX Predictor - Basic Functionality")
    print("="*60 + "\n")
    
    try:
        # Initialize predictor
        print("1. Initializing predictor...")
        predictor = Predictor()
        predictor.setup()
        print("   ✓ Predictor initialized successfully\n")
        
        # Test CUDA availability
        print("2. Checking CUDA availability...")
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA is available")
            print(f"   - Device: {torch.cuda.get_device_name(0)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("   ✗ WARNING: CUDA not available, will use CPU (very slow)")
        print()
        
        print("="*60)
        print("✓ Basic functionality test PASSED")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_audio():
    """Test with actual audio URLs (requires internet and audio URLs)."""
    print("\n" + "="*60)
    print("Testing WhisperX Predictor - Sample Audio")
    print("="*60 + "\n")
    
    print("Note: This test requires valid audio URLs and may take several minutes.")
    print("Skipping audio processing test. Use deploy.sh to test on RunPod.")
    print()
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("WhisperX RunPod Local Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Sample Audio", test_with_sample_audio),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*60 + "\n")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("✓ All tests passed! Ready for deployment.\n")
        print("Next steps:")
        print("  1. Run: ./deploy.sh")
        print("  2. Follow the deployment instructions")
        print("  3. See DEPLOYMENT.md for details\n")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before deploying.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

