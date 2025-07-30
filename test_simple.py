#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from automl.automl import AutoML
    
    print("Testing AutoML initialization...")
    
    # Test basic initialization
    automl1 = AutoML(seed=42)
    print("✓ Basic initialization successful")
    
    # Test with use_mult_algorithms=True
    automl2 = AutoML(seed=42, use_mult_algorithms=True)
    print("✓ Enhanced initialization successful")
    
    # Test with use_mult_algorithms=False
    automl3 = AutoML(seed=42, use_mult_algorithms=False)
    print("✓ Basic mode initialization successful")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
