#!/usr/bin/env python3
"""
Detection System - Main Application

A modular object detection system using YOLO for real-time tracking and detection.
"""

import time
from colorama import Fore, Style

from config import Config
from utils.device_manager import get_device, clear_gpu_memory
from utils.file_validator import check_files, validate_directory_structure
from utils.video_processor import VideoProcessor, calculate_progress_info
from utils.object_detector import ObjectDetector


def main():
    """Main application entry point."""
    print(Fore.MAGENTA + "\nObject Detection System Starting...\n")
    
    try:
        # Initialize components
        device = None
        detector = None
        processor = None
        
        # Check if required files exist
        if not check_files(Config.MODEL_PATH, Config.VIDEO_PATH):
            return 1
        
        # Validate and create results directory structure
        if not validate_directory_structure(Config.RESULTS_BASE_DIR):
            print(Fore.RED + "Failed to validate or create results directory")
            return 1
        
        # Get optimal device for inference
        device = get_device()
        
        # Initialize object detector
        detector = ObjectDetector(Config.MODEL_PATH)
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
