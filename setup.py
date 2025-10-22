#!/usr/bin/env python3
"""
Setup script for the Speech-to-Text with Speaker Diarization project
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist"""
    if not os.path.exists('.venv'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

def install_dependencies():
    """Install required dependencies"""
    pip_path = os.path.join('.venv', 'bin', 'pip') if os.name != 'nt' else os.path.join('.venv', 'Scripts', 'pip.exe')
    
    print("Installing dependencies...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    print("✅ Dependencies installed")

def create_directories():
    """Create necessary directories"""
    directories = ['transcriptions', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/ directory")
        else:
            print(f"✅ {directory}/ directory already exists")

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("\n1. ACTIVATE VIRTUAL ENVIRONMENT:")
    print("   source .venv/bin/activate")
    print("\n2. GET HUGGINGFACE TOKEN:")
    print("   - Visit: https://huggingface.co/settings/tokens")
    print("   - Create a new token with READ permissions")
    print("   - Accept licenses at:")
    print("     • https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("     • https://huggingface.co/pyannote/speaker-diarization-community-1")
    print("\n3. RUN THE SCRIPT:")
    print("   python speech_to_text_with_diarization.py --hf-token YOUR_TOKEN")
    print("\n4. OPTIONAL PARAMETERS:")
    print("   --whisper-model [tiny|base|small|medium|large]")
    print("   --input-dir [directory containing .wav files]")
    print("   --output-dir [directory for results]")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 Setting up Speech-to-Text with Speaker Diarization")
    print("="*60)
    
    try:
        check_python_version()
        setup_virtual_environment()
        install_dependencies()
        create_directories()
        display_next_steps()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()