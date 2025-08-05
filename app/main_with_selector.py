#!/usr/bin/env python3
"""
Object Detection System - Main Application with Model Selection

Enhanced version with interactive model selection capability.
"""

import time
import sys
from colorama import Fore, Style

from config import Config
from utils.device_manager import get_device, clear_gpu_memory
from utils.file_validator import check_files, validate_directory_structure
from utils.video_processor import VideoProcessor, calculate_progress_info
from utils.object_detector import ObjectDetector
from utils.model_store import ModelSelector, ModelInfo


def select_model_interactive() -> str:
    """Interactive model selection interface."""
    print(Fore.MAGENTA + "\nModel Selection Interface\n")
    
    selector = ModelSelector(model_directory="./model")
    
    print(Fore.CYAN + "Choose how to select your model:")
    print("1. Interactive selection from all available models")
    print("2. Get recommendation based on use case")
    print("3. Use default model from config")
    print("4. Quit")
    
    try:
        choice = input(Fore.WHITE + "\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            model_path = selector.select_model_interactive()
            if model_path:
                return model_path
            else:
                print(Fore.YELLOW + "No model selected, using default from config.")
                return Config.MODEL_PATH
                
        elif choice == "2":
            print(Fore.CYAN + "\nSelect use case:")
            print("1. Speed priority (fastest inference)")
            print("2. Accuracy priority (best detection)")
            print("3. Balanced (good speed + accuracy)")
            print("4. General purpose (recommended)")
            print("5. Best quality (highest accuracy)")
            
            use_case_choice = input(Fore.WHITE + "Enter choice (1-5): ").strip()
            use_cases = {
                "1": "speed",
                "2": "accuracy", 
                "3": "balanced",
                "4": "general",
                "5": "best"
            }
            
            use_case = use_cases.get(use_case_choice, "general")
            return selector.recommend_model(use_case)
            
        elif choice == "3":
            print(Fore.GREEN + f"Using default model: {Config.MODEL_PATH}")
            return Config.MODEL_PATH
            
        elif choice == "4":
            print(Fore.YELLOW + "Goodbye!")
            sys.exit(0)
            
        else:
            print(Fore.RED + "Invalid choice, using default model.")
            return Config.MODEL_PATH
            
    except (KeyboardInterrupt, EOFError):
        print(Fore.YELLOW + "\nModel selection interrupted, using default model.")
        return Config.MODEL_PATH


def main():
    """Main application entry point."""
    print(Fore.MAGENTA + "\nObject Detection System Starting...\n")
    
    try:
        # Initialize components
        device = None
        detector = None
        processor = None
        
        # Interactive model selection
        selected_model_path = select_model_interactive()
        print(Fore.GREEN + f"\nSelected model: {selected_model_path}")
        
        # Display model information if available
        model_filename = selected_model_path.split('/')[-1] if '/' in selected_model_path else selected_model_path.split('\\')[-1]
        model_info = ModelInfo.get_model_info(model_filename)
        if model_info:
            print(Fore.CYAN + f"Model Info: {model_info['name']}")
            print(Fore.CYAN + f"   Task: {model_info['task']}, Size: {model_info['size_mb']}MB")
            print(Fore.CYAN + f"   mAP: {model_info['map50_95']}, GPU Speed: {model_info['speed_gpu_ms']}ms")
            print(Fore.CYAN + f"   Description: {model_info['description']}")
        
        # Check if required files exist (video file)
        if not check_files(selected_model_path, Config.VIDEO_PATH):
            return 1
        
        # Validate and create results directory structure
        if not validate_directory_structure(Config.RESULTS_BASE_DIR):
            print(Fore.RED + "Failed to validate or create results directory")
            return 1
        
        # Get optimal device for inference
        device = get_device()
        
        # Initialize object detector with selected model
        detector = ObjectDetector(selected_model_path)
        detector.load_model(device)
        
        # Initialize video processor
        processor = VideoProcessor(Config.OUTPUT_FILENAME, Config.RESULTS_BASE_DIR)
        
        # Validate the output structure
        if not processor.validate_output_structure():
            print(Fore.RED + "Failed to validate output directory structure")
            return 1
        
        # Setup video capture and writer
        frame_width, frame_height, fps, total_frames = processor.setup_video_capture(Config.VIDEO_PATH)
        processor.setup_video_writer(frame_width, frame_height, fps)
        
        print(Fore.CYAN + f"Output will be saved to: {processor.get_output_path()}")
        
        # Process video frames
        frame_count = 0
        last_logged_progress = 0
        overall_start_time = time.time()
        print(Fore.CYAN + "Starting detection...")
        
        while True:
            start_time = time.time()
            
            # Read frame
            success, frame = processor.read_frame()
            if not success:
                break
                
            frame_count += 1
            
            try:
                # Perform object detection and tracking
                results = detector.detect_objects(
                    frame, 
                    conf_threshold=Config.CONFIDENCE_THRESHOLD,
                    tracker=Config.TRACKER_CONFIG
                )
                
                # Process frame with tracking visualization
                annotated_frame = processor.process_frame_with_tracking(frame, results)
                
            except Exception as e:
                print(Fore.YELLOW + f"Frame {frame_count} processing error: {e}")
                annotated_frame = frame
            
            # Write processed frame to output
            processor.write_frame(annotated_frame)
            
            # Calculate and log progress
            end_time = time.time()
            processing_time = end_time - start_time
            progress, fps_actual = calculate_progress_info(frame_count, total_frames, processing_time)
            
            if int(progress) > last_logged_progress:
                last_logged_progress = int(progress)
                print(Fore.YELLOW + f"Processing - ({progress:.0f}%) | FPS: {fps_actual:.1f}")
        
        overall_end_time = time.time()
        print(Fore.GREEN + f"\nProcessing complete!")
        print(Fore.CYAN + f"Output saved to: {processor.get_output_path()}")
        
        # Save processing information
        processor.save_processing_info(frame_count, total_frames, overall_start_time, overall_end_time)
        
        return 0
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        print(Fore.RED + f"Error occurred: {e}")
        
        # Try CPU fallback if GPU fails
        if device and device != "cpu":
            print(Fore.YELLOW + "Trying CPU fallback...")
            try:
                detector.load_model("cpu")
                # Could restart processing here if needed
            except Exception as fallback_error:
                print(Fore.RED + f"CPU fallback also failed: {fallback_error}")
        
        return 1
        
    finally:
        # Cleanup resources
        if processor:
            processor.cleanup()
        
        if device:
            clear_gpu_memory(device)
        
        print(Fore.GREEN + "Cleanup complete!\n")


if __name__ == "__main__":
    exit(main())
