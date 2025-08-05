"""
Model Management Utility

A standalone utility for managing YOLO models.
"""

import sys
import os
from colorama import Fore, init

# Handle imports based on how the script is run
try:
    # Try relative import first (when imported as module)
    from .model_store import ModelSelector, ModelInfo, YOLOModelList
except ImportError:
    # If relative import fails, add parent directory to path and use absolute import
    # This allows the script to run standalone
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from utils.model_store import ModelSelector, ModelInfo, YOLOModelList

# Initialize colorama
init()


def main():
    """Main model management interface."""
    print(Fore.MAGENTA + "\nYOLO Model Management Utility\n")
    
    selector = ModelSelector()
    
    while True:
        print(Fore.CYAN + "Choose an option:")
        print("1. List all available models")
        print("2. Show local models")
        print("3. Interactive model selection")
        print("4. Get model recommendations")
        print("5. Show model information")
        print("6. Show model categories")
        print("7. Exit")
        
        try:
            choice = input(Fore.WHITE + "\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                model_list = selector.display_model_selection_menu()
                
            elif choice == "2":
                local_models = selector.list_local_models()
                if local_models:
                    print(Fore.GREEN + "\nLocal Models:")
                    for model in local_models:
                        info = ModelInfo.get_model_info(model)
                        if info:
                            print(f"  {model} - {info['name']} ({info['size_mb']}MB)")
                        else:
                            print(f"  {model}")
                else:
                    print(Fore.YELLOW + "\nNo local models found in ./model directory")
                    
            elif choice == "3":
                selected = selector.select_model_interactive()
                if selected:
                    print(Fore.GREEN + f"\nSelected: {selected}")
                    
            elif choice == "4":
                print(Fore.CYAN + "\nModel Recommendations:")
                use_cases = ["speed", "accuracy", "balanced", "general", "best"]
                for use_case in use_cases:
                    recommended = selector.recommend_model(use_case)
                    
            elif choice == "5":
                print(Fore.CYAN + "\nEnter model filename (e.g., yolo11s.pt): ", end="")
                model_name = input().strip()
                info = ModelInfo.get_model_info(model_name)
                if info:
                    print(Fore.GREEN + f"\nModel Information for {model_name}:")
                    print(f"  Name: {info['name']}")
                    print(f"  Task: {info['task']}")
                    print(f"  Size: {info['size_mb']}MB")
                    print(f"  mAP50-95: {info['map50_95']}")
                    print(f"  CPU Speed: {info['speed_cpu_ms']}ms")
                    print(f"  GPU Speed: {info['speed_gpu_ms']}ms")
                    print(f"  Description: {info['description']}")
                else:
                    print(Fore.YELLOW + f"\nNo information available for {model_name}")
                    
            elif choice == "6":
                print(Fore.CYAN + "\nModel Categories:")
                models = YOLOModelList()
                categories = {
                    "YOLOv11 Detection": [models.YOLOv11.N, models.YOLOv11.S, models.YOLOv11.M, models.YOLOv11.L, models.YOLOv11.X],
                    "YOLOv11 Classification": [models.YOLOv11cls.N, models.YOLOv11cls.S, models.YOLOv11cls.M, models.YOLOv11cls.L, models.YOLOv11cls.X],
                    "YOLOv11 Segmentation": [models.YOLOv11seg.N, models.YOLOv11seg.S, models.YOLOv11seg.M, models.YOLOv11seg.L, models.YOLOv11seg.X],
                    "YOLOv11 Pose": [models.YOLOv11pose.N, models.YOLOv11pose.S, models.YOLOv11pose.M, models.YOLOv11pose.L, models.YOLOv11pose.X],
                    "YOLOv10 Detection": [models.YOLOv10.N, models.YOLOv10.S, models.YOLOv10.M, models.YOLOv10.L, models.YOLOv10.X],
                    "YOLOv8 Detection": [models.YOLOv8.N, models.YOLOv8.S, models.YOLOv8.M, models.YOLOv8.L, models.YOLOv8.X]
                }
                
                for category, model_list in categories.items():
                    print(Fore.YELLOW + f"\n{category}:")
                    for model in model_list:
                        status = "LOCAL" if selector.check_model_exists(model) else "REMOTE"
                        print(f"  [{status}] {model}")
                        
            elif choice == "7":
                print(Fore.GREEN + "\nGoodbye!")
                break
                
            else:
                print(Fore.RED + "\nInvalid choice! Please enter 1-7.")
                
            print(Fore.WHITE + "\n" + "="*50)
            
        except (KeyboardInterrupt, EOFError):
            print(Fore.YELLOW + "\nGoodbye!")
            break
        except Exception as e:
            print(Fore.RED + f"\nError: {e}")


if __name__ == "__main__":
    main()
