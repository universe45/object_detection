import os
from colorama import Fore


def check_files(model_path="./model/yolo11s.pt", video_path="./test_material/853889-hd_1920_1080_25fps.mp4"):
    """
    Checks if required model and video files exist.
    
    Args:
        model_path (str): Path to the YOLO model file
        video_path (str): Path to the input video file
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    if not os.path.exists(model_path):
        print(Fore.RED + f"Model file not found: {model_path}")
        return False
    if not os.path.exists(video_path):
        print(Fore.RED + f"Video file not found: {video_path}")
        return False
    
    print(Fore.GREEN + "All files found")
    return True


def validate_model_path(model_path):
    """
    Validates that the model file exists and has the correct extension.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        print(Fore.RED + f"Model file not found: {model_path}")
        return False
    
    if not model_path.endswith('.pt'):
        print(Fore.YELLOW + f"Warning: Model file should have .pt extension: {model_path}")
    
    return True


def validate_video_path(video_path):
    """
    Validates that the video file exists and has a supported extension.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(Fore.RED + f"Video file not found: {video_path}")
        return False
    
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if not any(video_path.lower().endswith(ext) for ext in supported_extensions):
        print(Fore.YELLOW + f"Warning: Video file may not be supported: {video_path}")
    
    return True


def validate_directory_structure(base_dir="results"):
    """
    Validates and creates the results directory structure if needed.
    
    Args:
        base_dir (str): Base directory for results
        
    Returns:
        bool: True if valid/created successfully, False otherwise
    """
    try:
        # Check if directory exists
        if os.path.exists(base_dir):
            print(Fore.CYAN + f"Results directory exists: {base_dir}")
            
            # Check if it's actually a directory
            if not os.path.isdir(base_dir):
                print(Fore.RED + f"Error: {base_dir} exists but is not a directory")
                return False
            
            # Check if it's writable
            if not os.access(base_dir, os.W_OK):
                print(Fore.RED + f"Error: {base_dir} is not writable")
                return False
                
            print(Fore.GREEN + f"Results directory is valid and writable: {base_dir}")
            return True
        else:
            # Create the directory
            os.makedirs(base_dir, exist_ok=True)
            print(Fore.GREEN + f"Created results directory: {base_dir}")
            return True
            
    except PermissionError:
        print(Fore.RED + f"Permission denied: Cannot create or access {base_dir}")
        return False
    except Exception as e:
        print(Fore.RED + f"Error validating directory structure: {e}")
        return False


def get_directory_info(directory_path):
    """
    Get information about a directory.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        dict: Directory information or None if directory doesn't exist
    """
    if not os.path.exists(directory_path):
        return None
    
    try:
        stat_info = os.stat(directory_path)
        return {
            'exists': True,
            'is_directory': os.path.isdir(directory_path),
            'is_writable': os.access(directory_path, os.W_OK),
            'is_readable': os.access(directory_path, os.R_OK),
            'size_bytes': stat_info.st_size,
            'created_time': stat_info.st_ctime,
            'modified_time': stat_info.st_mtime,
            'permissions': oct(stat_info.st_mode)[-3:]
        }
    except Exception as e:
        print(Fore.YELLOW + f"Warning: Could not get directory info for {directory_path}: {e}")
        return {'exists': True, 'error': str(e)}
