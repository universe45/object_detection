from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import torch
import platform
import os
from colorama import Fore, Back, Style

print(Fore.MAGENTA + "\nDetection System Starting...\n")

# Suppress YOLO logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# OS-aware device detection
def get_device():
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

# Check if files exist
def check_files():
    model_path = "./model/yolo11s.pt"
    video_path = "./test_material/853889-hd_1920_1080_25fps.mp4"
    
    if not os.path.exists(model_path):
        print(Fore.RED + f"Model file not found: {model_path}")
        return False
    if not os.path.exists(video_path):
        print(Fore.RED + f"Video file not found: {video_path}")
        return False
    
    print(Fore.GREEN + "All files found")
    return True

# Get device and check files
if not check_files():
    exit(1)
device = get_device()

try:
    # Load the YOLO11 model
    print(Fore.CYAN + "Loading YOLO model...")
    model = YOLO("./model/yolo11s.pt")
    print(Fore.GREEN + "Model loaded successfully")

    # Open the video file
    video_path = "./test_material/853889-hd_1920_1080_25fps.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(Fore.GREEN + f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    # Output video writer with fallback codecs
    output_path = "result.mp4"
    codecs = ["avc1", "mp4v", "XVID"]
    out = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                print(Fore.GREEN + f"Video writer ready with {codec} codec")
                break
            out.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        raise IOError("Failed to initialize video writer")

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    frame_count = 0
    last_logged_progress = 0
    print(Fore.CYAN + "Starting detection...")
    
    while cap.isOpened():
        start_time = time.time()
        
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            frame_count += 1
            
            try:
                # Run YOLO11 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                    conf=0.5, device=device, verbose=False)

                # Check if any detections were made
                if results[0].boxes is not None:
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id
                    if track_ids is not None:
                        track_ids = track_ids.int().cpu().tolist()
                    else:
                        track_ids = []

                    # Visualize the results on the frame with smaller labels
                    annotated_frame = results[0].plot(line_width=2, font_size=0.5)

                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, 
                                    color=(230, 230, 230), thickness=2)
                else:
                    # If no boxes are detected, just use the current frame as is
                    annotated_frame = frame
                    
            except Exception as e:
                print(Fore.YELLOW + f"Frame {frame_count} processing error: {e}")
                annotated_frame = frame

            # Write the frame to the output video
            out.write(annotated_frame)

            # Calculate and print processing time and progress
            end_time = time.time()
            processing_time = end_time - start_time
            progress = (frame_count / total_frames) * 100
            if int(progress) > last_logged_progress:
                last_logged_progress = int(progress)
                fps_actual = 1.0 / processing_time if processing_time > 0 else 0
                print(Fore.YELLOW + f"Processing - ({progress:.0f}%) | FPS: {fps_actual:.1f}")

        else:
            # Break the loop if the end of the video is reached
            break

except KeyboardInterrupt:
    print(Fore.YELLOW + "\nProcessing interrupted by user")
except Exception as e:
    print(Fore.RED + f"Error occurred: {e}")
    # Try to fall back to CPU if GPU fails
    if device != "cpu":
        print(Fore.YELLOW + "🔄 Trying CPU fallback...")
        device = "cpu"

finally:
    # Release the video capture object, writer, and close display window
    print(Fore.CYAN + "\nCleaning up...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    if 'out' in locals() and out.isOpened():
        out.release()
    
    # Clear GPU memory if using GPU
    if device in ["cuda", "mps"]:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(Fore.GREEN + "GPU memory cleared")
        except:
            pass
    
    cv2.destroyAllWindows()
    print(Fore.GREEN + "\nProcessing complete!\n")