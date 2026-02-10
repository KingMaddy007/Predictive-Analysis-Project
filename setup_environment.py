"""
Setup Script for Mamba-RUL Project

This script helps set up the Python environment and install dependencies.

Usage:
    python setup_environment.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {command}")
    print()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    
    print("=" * 70)
    print(" " * 15 + "MAMBA-RUL ENVIRONMENT SETUP")
    print("=" * 70)
    print()
    
    project_dir = Path(__file__).parent
    venv_dir = project_dir / "venv"
    
    print(f"Project Directory: {project_dir}")
    print(f"Virtual Environment: {venv_dir}")
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ ERROR: Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    
    # Step 1: Create virtual environment
    if not venv_dir.exists():
        success = run_command(
            f'python -m venv "{venv_dir}"',
            "Creating virtual environment"
        )
        if not success:
            return False
    else:
        print(f"\n✓ Virtual environment already exists at {venv_dir}")
    
    # Determine activation script
    if sys.platform == "win32":
        activate_script = venv_dir / "Scripts" / "activate.bat"
        pip_executable = venv_dir / "Scripts" / "pip.exe"
    else:
        activate_script = venv_dir / "bin" / "activate"
        pip_executable = venv_dir / "bin" / "pip"
    
    # Step 2: Upgrade pip
    success = run_command(
        f'"{pip_executable}" install --upgrade pip',
        "Upgrading pip"
    )
    if not success:
        print("⚠ Warning: Could not upgrade pip, continuing anyway...")
    
    # Step 3: Install requirements
    requirements_file = project_dir / "requirements.txt"
    
    if requirements_file.exists():
        success = run_command(
            f'"{pip_executable}" install -r "{requirements_file}"',
            "Installing requirements from requirements.txt"
        )
        if not success:
            print("✗ Failed to install requirements")
            return False
    else:
        print(f"✗ ERROR: requirements.txt not found at {requirements_file}")
        return False
    
    # Step 4: Verify installation
    print("\n" + "=" * 70)
    print("VERIFYING INSTALLATION")
    print("=" * 70)
    
    # Run section0_setup.py to verify
    section0_script = project_dir / "src" / "section0_setup.py"
    
    if section0_script.exists():
        python_executable = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        run_command(
            f'"{python_executable}" "{section0_script}"',
            "Running environment verification"
        )
    
    # Final instructions
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("To activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")
    print()
    print("To run the complete pipeline:")
    print("  python main_pipeline.py")
    print()
    print("To run individual sections:")
    print("  python src/section0_setup.py")
    print("  python src/section1_load_data.py")
    print("  python src/section2_preprocessing.py")
    print("  python src/section3_sequence_gen.py")
    print("  python src/section4_dataset.py")
    print("  python src/section5_mamba_model.py")
    print()
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
