"""
SECTION 0 — INSTALLATION & ENVIRONMENT SETUP

This module verifies that all required libraries are installed and accessible.
It checks for GPU availability and ensures the environment is ready for training.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import sys
from typing import Dict, Tuple


def check_library_import(library_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Attempt to import a library and return success status with version info.
    
    Args:
        library_name: Display name of the library
        import_name: Actual import name (if different from library_name)
    
    Returns:
        Tuple of (success: bool, version_info: str)
    """
    if import_name is None:
        import_name = library_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def verify_environment() -> Dict[str, any]:
    """
    Verify all required libraries and system configuration.
    
    Returns:
        Dictionary containing verification results
    """
    print("=" * 70)
    print("MAMBA-RUL ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print()
    
    # Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Required libraries
    libraries = [
        ('PyTorch', 'torch'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('SciPy', 'scipy'),
        ('Matplotlib', 'matplotlib'),
        ('Scikit-learn', 'sklearn'),
        ('tqdm', 'tqdm'),
        ('Seaborn', 'seaborn'),
    ]
    
    results = {}
    all_success = True
    
    print("Checking Required Libraries:")
    print("-" * 70)
    
    for lib_name, import_name in libraries:
        success, info = check_library_import(lib_name, import_name)
        results[lib_name] = {'installed': success, 'version': info}
        
        status = "✓ INSTALLED" if success else "✗ MISSING"
        print(f"{lib_name:20s} {status:15s} {info}")
        
        if not success:
            all_success = False
    
    print()
    
    # Check Mamba SSM separately (may not be installed yet)
    print("Checking Mamba SSM:")
    print("-" * 70)
    success, info = check_library_import('Mamba-SSM', 'mamba_ssm')
    results['Mamba-SSM'] = {'installed': success, 'version': info}
    
    if success:
        print(f"Mamba-SSM:           ✓ INSTALLED     {info}")
    else:
        print(f"Mamba-SSM:           ✗ MISSING")
        print(f"  Note: Install with: pip install mamba-ssm")
        all_success = False
    
    print()
    
    # GPU Check
    print("GPU Availability:")
    print("-" * 70)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        results['cuda_available'] = cuda_available
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"CUDA Available:      ✓ YES")
            print(f"GPU Device:          {gpu_name}")
            print(f"GPU Count:           {gpu_count}")
            results['gpu_name'] = gpu_name
            results['gpu_count'] = gpu_count
        else:
            print(f"CUDA Available:      ✗ NO (will use CPU)")
            print(f"  Note: Training will be slower on CPU")
    except Exception as e:
        print(f"GPU Check Failed:    {e}")
        results['cuda_available'] = False
    
    print()
    print("=" * 70)
    
    if all_success:
        print("✓ ENVIRONMENT READY - All required libraries are installed!")
    else:
        print("✗ ENVIRONMENT INCOMPLETE - Please install missing libraries")
        print("  Run: pip install -r requirements.txt")
    
    print("=" * 70)
    
    return results


def main():
    """Main execution function."""
    results = verify_environment()
    
    # Return exit code based on success
    if all(v.get('installed', False) for k, v in results.items() 
           if isinstance(v, dict) and k != 'cuda_available'):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
