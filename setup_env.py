import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

def install_requirements():
    # Ensure pip is upgraded
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Install Python packages from requirements.txt
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def install_pytorch():
    """Install PyTorch with CUDA support."""
    torch_url = 'https://download.pytorch.org/whl/cu121'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', torch_url])

def install_ndi_python():
    # Clone the ndi-python repository
    subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/buresu/ndi-python.git', '/app/ndi-python'])

    # Build ndi-python
    subprocess.check_call([sys.executable, '/app/ndi-python/setup.py', 'build'])
    subprocess.check_call([sys.executable, '/app/ndi-python/setup.py', 'bdist_wheel'])

    # Install the built wheel
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '/app/ndi-python/dist/*.whl'])

def install_opencv_cuda():
    # Determine OS
    os_name = platform.system().lower()
    
    if os_name == 'windows':
        # Windows specific OpenCV CUDA wheel download
        opencv_wheel_url = 'https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.7.0.20230527/opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-win_amd64.whl'
        wheel_path = 'opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-win_amd64.whl'
    elif os_name == 'linux':
        # Linux specific OpenCV CUDA wheel download
        opencv_wheel_url = 'https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.7.0.20230527/opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl'
        wheel_path = 'opencv_contrib_python_rolling-4.7.0.20230527-cp36-abi3-linux_x86_64.whl'
    else:
        raise RuntimeError(f'Unsupported OS: {os_name}')
    
    # Download the OpenCV wheel
    urllib.request.urlretrieve(opencv_wheel_url, wheel_path)
    
    # Install the wheel
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel_path])
    
    # Clean up
    os.remove(wheel_path)

def main():
    
    # Install general requirements
    install_requirements()

    install_pytorch()
    
    # Install NDI-Python
    install_ndi_python()
    
    # Install CUDA-optimized OpenCV
    install_opencv_cuda()

    
    print("Setup complete!")

if __name__ == '__main__':
    main()
