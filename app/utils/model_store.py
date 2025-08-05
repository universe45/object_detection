import os
from colorama import Fore
from typing import Dict, List, Optional


class YOLOModelList:
    """Collection of YOLO models for different tasks."""
    
    def __init__(self):
        self.info = "This class is a collection of YOLO models for different tasks"
    
    class YOLOv11:
        """YOLOv11 object detection models."""
        N = "yolo11n.pt"
        S = "yolo11s.pt"
        M = "yolo11m.pt"
        L = "yolo11l.pt"
        X = "yolo11x.pt"
    
    class YOLOv11cls:
        """YOLOv11 classification models."""
        N = "yolo11n-cls.pt"
        S = "yolo11s-cls.pt"
        M = "yolo11m-cls.pt"
        L = "yolo11l-cls.pt"
        X = "yolo11x-cls.pt"
    
    class YOLOv11seg:
        """YOLOv11 segmentation models."""
        N = "yolo11n-seg.pt"
        S = "yolo11s-seg.pt"
        M = "yolo11m-seg.pt"
        L = "yolo11l-seg.pt"
        X = "yolo11x-seg.pt"
    
    class YOLOv11pose:
        """YOLOv11 pose estimation models."""
        N = "yolo11n-pose.pt"
        S = "yolo11s-pose.pt"
        M = "yolo11m-pose.pt"
        L = "yolo11l-pose.pt"
        X = "yolo11x-pose.pt"
    
    class YOLOv10:
        """YOLOv10 object detection models."""
        N = "yolo10n.pt"
        S = "yolo10s.pt"
        M = "yolo10m.pt"
        L = "yolo10l.pt"
        X = "yolo10x.pt"
    
    class YOLOv8:
        """YOLOv8 object detection models."""
        N = "yolo8n.pt"
        S = "yolo8s.pt"
        M = "yolo8m.pt"
        L = "yolo8l.pt"
        X = "yolo8x.pt"


class ModelInfo:
    """Information about YOLO models including size, speed, and accuracy metrics."""
    
    MODEL_SPECS = {
        # YOLOv11 Detection
        "yolo11n.pt": {
            "name": "YOLOv11 Nano",
            "task": "detection",
            "size_mb": 5.1,
            "speed_cpu_ms": 2.4,
            "speed_gpu_ms": 0.5,
            "map50_95": 39.5,
            "description": "Fastest, smallest model for basic detection tasks"
        },
        "yolo11s.pt": {
            "name": "YOLOv11 Small",
            "task": "detection", 
            "size_mb": 19.8,
            "speed_cpu_ms": 8.1,
            "speed_gpu_ms": 0.7,
            "map50_95": 47.0,
            "description": "Good balance of speed and accuracy"
        },
        "yolo11m.pt": {
            "name": "YOLOv11 Medium",
            "task": "detection",
            "size_mb": 50.5,
            "speed_cpu_ms": 18.4,
            "speed_gpu_ms": 1.2,
            "map50_95": 51.5,
            "description": "Better accuracy, moderate speed"
        },
        "yolo11l.pt": {
            "name": "YOLOv11 Large",
            "task": "detection",
            "size_mb": 85.8,
            "speed_cpu_ms": 27.6,
            "speed_gpu_ms": 1.8,
            "map50_95": 53.4,
            "description": "High accuracy, slower inference"
        },
        "yolo11x.pt": {
            "name": "YOLOv11 Extra Large",
            "task": "detection",
            "size_mb": 140.4,
            "speed_cpu_ms": 49.2,
            "speed_gpu_ms": 2.8,
            "map50_95": 54.7,
            "description": "Highest accuracy, slowest inference"
        }
    }
    
    @classmethod
    def get_model_info(cls, model_filename: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return cls.MODEL_SPECS.get(model_filename)
    
    @classmethod
    def list_all_models(cls) -> List[str]:
        """Get list of all available model filenames."""
        return list(cls.MODEL_SPECS.keys())


class ModelSelector:
    """Interactive model selector and downloader."""
    
    def __init__(self, model_directory: str = "./model"):
        self.model_directory = model_directory
        self.models = YOLOModelList()
        self.ensure_model_directory()
    
    def ensure_model_directory(self):
        """Ensure the model directory exists."""
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory, exist_ok=True)
            print(Fore.GREEN + f"Created model directory: {self.model_directory}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models organized by category."""
        return {
            "YOLOv11 Detection": [
                self.models.YOLOv11.N, self.models.YOLOv11.S, 
                self.models.YOLOv11.M, self.models.YOLOv11.L, self.models.YOLOv11.X
            ],
            "YOLOv11 Classification": [
                self.models.YOLOv11cls.N, self.models.YOLOv11cls.S,
                self.models.YOLOv11cls.M, self.models.YOLOv11cls.L, self.models.YOLOv11cls.X
            ],
            "YOLOv11 Segmentation": [
                self.models.YOLOv11seg.N, self.models.YOLOv11seg.S,
                self.models.YOLOv11seg.M, self.models.YOLOv11seg.L, self.models.YOLOv11seg.X
            ],
            "YOLOv11 Pose": [
                self.models.YOLOv11pose.N, self.models.YOLOv11pose.S,
                self.models.YOLOv11pose.M, self.models.YOLOv11pose.L, self.models.YOLOv11pose.X
            ],
            "YOLOv10 Detection": [
                self.models.YOLOv10.N, self.models.YOLOv10.S,
                self.models.YOLOv10.M, self.models.YOLOv10.L, self.models.YOLOv10.X
            ],
            "YOLOv8 Detection": [
                self.models.YOLOv8.N, self.models.YOLOv8.S,
                self.models.YOLOv8.M, self.models.YOLOv8.L, self.models.YOLOv8.X
            ]
        }
    
    def check_model_exists(self, model_filename: str) -> bool:
        """Check if a model file exists in the model directory."""
        model_path = os.path.join(self.model_directory, model_filename)
        return os.path.exists(model_path)
    
    def get_model_path(self, model_filename: str) -> str:
        """Get the full path to a model file."""
        return os.path.join(self.model_directory, model_filename)
    
    def list_local_models(self) -> List[str]:
        """List all model files present in the model directory."""
        if not os.path.exists(self.model_directory):
            return []
        
        model_files = []
        for file in os.listdir(self.model_directory):
            if file.endswith('.pt'):
                model_files.append(file)
        return sorted(model_files)
    
    def display_model_selection_menu(self):
        """Display an interactive model selection menu."""
        print(Fore.MAGENTA + "\n=== YOLO Model Selection Menu ===\n")
        
        available_models = self.get_available_models()
        local_models = self.list_local_models()
        
        if local_models:
            print(Fore.GREEN + "Local Models Available:")
            for i, model in enumerate(local_models, 1):
                info = ModelInfo.get_model_info(model)
                if info:
                    print(f"  {i}. {model} - {info['name']} ({info['size_mb']}MB)")
                else:
                    print(f"  {i}. {model}")
            print()
        
        print(Fore.CYAN + "All Available Models:")
        model_index = 1
        model_list = []
        
        for category, models in available_models.items():
            print(Fore.YELLOW + f"\n{category}:")
            for model in models:
                status = "Local" if self.check_model_exists(model) else "Download"
                info = ModelInfo.get_model_info(model)
                if info:
                    print(f"  {model_index}. {model} - {info['name']} ({info['size_mb']}MB) [{status}]")
                    print(f"      mAP: {info['map50_95']}, GPU: {info['speed_gpu_ms']}ms")
                else:
                    print(f"  {model_index}. {model} [{status}]")
                
                model_list.append(model)
                model_index += 1
        
        return model_list
    
    def select_model_interactive(self) -> Optional[str]:
        """Interactive model selection."""
        model_list = self.display_model_selection_menu()
        
        try:
            print(Fore.WHITE + f"\nEnter model number (1-{len(model_list)}) or 'q' to quit: ", end="")
            choice = input().strip()
            
            if choice.lower() == 'q':
                return None
            
            model_index = int(choice) - 1
            if 0 <= model_index < len(model_list):
                selected_model = model_list[model_index]
                model_path = self.get_model_path(selected_model)
                
                if self.check_model_exists(selected_model):
                    print(Fore.GREEN + f"Selected local model: {selected_model}")
                    return model_path
                else:
                    print(Fore.YELLOW + f"Model not found locally: {selected_model}")
                    print(Fore.CYAN + "This model will be automatically downloaded by YOLO when first used.")
                    return selected_model  # Return just filename for auto-download
            else:
                print(Fore.RED + "Invalid selection!")
                return None
                
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid input!")
            return None
    
    def recommend_model(self, use_case: str = "general") -> str:
        """Recommend a model based on use case."""
        recommendations = {
            "general": self.models.YOLOv11.S,
            "speed": self.models.YOLOv11.N,
            "accuracy": self.models.YOLOv11.L,
            "balanced": self.models.YOLOv11.M,
            "best": self.models.YOLOv11.X
        }
        
        recommended = recommendations.get(use_case, self.models.YOLOv11.S)
        print(Fore.GREEN + f"Recommended model for '{use_case}': {recommended}")
        
        info = ModelInfo.get_model_info(recommended)
        if info:
            print(Fore.CYAN + f"   {info['description']}")
            print(Fore.CYAN + f"   Size: {info['size_mb']}MB, mAP: {info['map50_95']}")
        
        return self.get_model_path(recommended)