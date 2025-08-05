import torch
import platform
from colorama import Fore


def get_device():
    """
    Detects the best available device for inference based on the operating system.
    
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
    os_name = platform.system()
    print(Fore.CYAN + f"OS: {os_name}")
    
    if os_name == "Darwin":  # macOS
        print(Fore.BLUE + "macOS detected - checking for Apple Silicon...")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(Fore.GREEN + "Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(Fore.GREEN + f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print(Fore.YELLOW + "Using CPU on macOS (consider M1/M2/M3 Mac for MPS)")

    elif os_name == "Windows":  # Windows
        print(Fore.BLUE + "Windows detected - checking for NVIDIA GPU...")
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            print(Fore.GREEN + f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            if gpu_count > 1:
                print(Fore.CYAN + f"Note: {gpu_count} GPUs available, using GPU 0")
        else:
            device = "cpu"
            print(Fore.YELLOW + "Using CPU on Windows (consider NVIDIA GPU for CUDA)")
            
    elif os_name == "Linux":  # Linux
        print(Fore.BLUE + "Linux detected - checking for CUDA support...")
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            print(Fore.GREEN + f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            if gpu_count > 1:
                print(Fore.CYAN + f"Note: {gpu_count} GPUs available, using GPU 0")
        else:
            device = "cpu"
            print(Fore.YELLOW + "Using CPU on Linux (install CUDA for GPU acceleration)")
            
    else:  # Other OS
        print(Fore.BLUE + f"Unknown OS ({os_name}) - using generic detection...")
        if torch.cuda.is_available():
            device = "cuda"
            print(Fore.GREEN + f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(Fore.GREEN + "Using Apple Silicon (MPS)")
        else:
            device = "cpu"
            print(Fore.YELLOW + "Using CPU")
    
    return device


def clear_gpu_memory(device):
    """
    Clears GPU memory if using a GPU device.
    
    Args:
        device (str): The device type ('cuda', 'mps', or 'cpu')
    """
    if device in ["cuda", "mps"]:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(Fore.GREEN + "GPU memory cleared")
        except:
            pass
