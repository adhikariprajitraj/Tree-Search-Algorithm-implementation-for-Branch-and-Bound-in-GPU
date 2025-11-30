import sys
import os

print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
    try:
        print(f"MPS available: {torch.backends.mps.is_available()}")
    except AttributeError:
        print("MPS not available (AttributeError)")

    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX: {ipex.__version__}")
        print(f"XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"XPU device: {torch.xpu.get_device_name(0)}")
    except ImportError:
        print("IPEX not installed")
    except AttributeError:
        print("torch.xpu not found (IPEX might be installed but not loaded correctly)")

except ImportError:
    print("PyTorch not installed")
