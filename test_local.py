#!/usr/bin/env python3
"""
Local test script for WhisperX worker without RunPod infrastructure.
Use this to test the predictor locally before building the Docker image.
"""

import json
import sys
from predict import Predictor

def test_predictor():
    """Test the predictor with sample input."""
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = Predictor()
    predictor.setup()
    
    # Load test input (same format as test_input.json)
    try:
        with open('test_input.json', 'r') as f:
            test_data = json.load(f)
            job_input = test_data.get('input', {})
    except FileNotFoundError:
        print("Error: test_input.json not found. Creating sample test input...")
        job_input = {
            "audio_urls": [
                "https://storage.googleapis.com/meowtxt-bucket/test/chunks/fca16cb3-dbb3-4d2c-8f45-84a75354d125/fca16cb3-dbb3-4d2c-8f45-84a75354d125_chunk_1.flac",
                "https://storage.googleapis.com/meowtxt-bucket/test/chunks/fca16cb3-dbb3-4d2c-8f45-84a75354d125/fca16cb3-dbb3-4d2c-8f45-84a75354d125_chunk_2.flac"
            ],
            "total_duration_seconds": 60.0,
            "chunk_size_seconds": 30.0,
            "language": "en",
            "debug": True,
            "batch_size": 32,
            "temperature": 0.2,
            "align_output": False,
            "diarization": False
        }
        
        # Save sample input for future use
        with open('test_input.json', 'w') as f:
            json.dump({"input": job_input}, f, indent=2)
        print("Created test_input.json with sample data")
    
    print("\nTest input:")
    print(json.dumps(job_input, indent=2))
    print("\n" + "="*60)
    print("Running prediction...")
    print("="*60 + "\n")
    
    # Run prediction
    try:
        result = predictor.predict(**job_input)
        
        print("\n" + "="*60)
        print("Prediction completed successfully!")
        print("="*60)
        print("\nResult:")
        print(json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    result = test_predictor()
    print("\n✅ Test completed successfully!")
