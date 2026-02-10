"""
Quick verification script to test the extracted IMS dataset.

This script checks that all test folders are accessible and contain data files.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from section1_load_data import IMSDataLoader


def main():
    print("=" * 70)
    print("IMS DATASET VERIFICATION")
    print("=" * 70)
    print()
    
    # Initialize loader
    dataset_path = Path(__file__).parent / "IMS"
    loader = IMSDataLoader(dataset_path)
    
    # Get available tests
    tests = loader.get_available_tests()
    print(f"✓ Found {len(tests)} test folders:")
    for test in tests:
        print(f"  - {test}")
    print()
    
    # Check each test
    for test_name in tests:
        print(f"Checking {test_name}...")
        test_path = dataset_path / test_name
        
        # Count files
        data_files = [f for f in test_path.iterdir() if f.is_file()]
        print(f"  ✓ Contains {len(data_files)} data files")
        
        # Try loading a small sample
        try:
            signal, files = loader.load_test_data(
                test_name=test_name,
                bearing_column=0,
                max_files=5  # Just load first 5 files for quick test
            )
            print(f"  ✓ Successfully loaded sample data ({len(signal):,} samples)")
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
        
        print()
    
    print("=" * 70)
    print("✓ VERIFICATION COMPLETE")
    print("=" * 70)
    print()
    print("Your dataset is ready to use!")
    print("Next step: Run the main pipeline")
    print("  python main_pipeline.py")


if __name__ == "__main__":
    main()
