import cv2
import numpy as np
from collections import defaultdict
import time
import os
from datetime import datetime
from colorama import Fore


class VideoProcessor:
    """Handles video processing and output writing."""
    
    def __init__(self, output_filename="result.mp4", results_base_dir="results"):
        self.output_filename = output_filename
        self.results_base_dir = results_base_dir
        self.track_history = defaultdict(lambda: [])
        self.cap = None
        self.out = None
        self.output_path = self._create_timestamped_output_path()
        
    def _create_timestamped_output_path(self):
        """
        Create a timestamped folder and return the full output path.
        Format: results/DDMMYYYY-HHMMSS/output_filename
        
        Returns:
            str: Full path to the output video file
        """
        # Check if results base directory exists, create if not
        if not os.path.exists(self.results_base_dir):
            os.makedirs(self.results_base_dir, exist_ok=True)
            print(Fore.GREEN + f"Created base results directory: {self.results_base_dir}")
        else:
            print(Fore.CYAN + f"Results directory exists: {self.results_base_dir}")
        
        # Create timestamp in DDMMYYYY-HHMMSS format
        now = datetime.now()
        timestamp = now.strftime("%d%m%Y-%H%M%S")
        
        # Create timestamped subdirectory
        timestamped_dir = os.path.join(self.results_base_dir, timestamp)
        
        # Create the timestamped directory
        os.makedirs(timestamped_dir, exist_ok=True)
        print(Fore.GREEN + f"Created timestamped directory: {timestamped_dir}")
        
        # Return full path to output file
        return os.path.join(timestamped_dir, self.output_filename)
    
    def validate_output_structure(self):
        """
        Validate that the output directory structure is properly set up.
        
        Returns:
            bool: True if structure is valid, False otherwise
        """
        try:
            # Check if base results directory exists
            if not os.path.exists(self.results_base_dir):
                print(Fore.YELLOW + f"Base results directory does not exist: {self.results_base_dir}")
                return False
            
            # Check if timestamped directory exists
            output_dir = self.get_output_directory()
            if not os.path.exists(output_dir):
                print(Fore.YELLOW + f"Timestamped directory does not exist: {output_dir}")
                return False
            
            # Check if directory is writable
            if not os.access(output_dir, os.W_OK):
                print(Fore.RED + f"Output directory is not writable: {output_dir}")
                return False
            
            print(Fore.GREEN + f"Output structure validated: {output_dir}")
            return True
            
        except Exception as e:
            print(Fore.RED + f"Error validating output structure: {e}")
            return False
    
    def get_output_directory(self):
        """
        Get the output directory path.
        
        Returns:
            str: Path to the timestamped output directory
        """
        return os.path.dirname(self.output_path)
    
    def get_output_path(self):
        """
        Get the full output file path.
        
        Returns:
            str: Full path to the output video file
        """
        return self.output_path
        
    def setup_video_capture(self, video_path):
        """
        Initialize video capture and get video properties.
        
        Args:
            video_path (str): Path to input video
            
        Returns:
            tuple: (frame_width, frame_height, fps, total_frames)
        """
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")
        
        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(Fore.GREEN + f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        
        return frame_width, frame_height, fps, total_frames
    
    def setup_video_writer(self, frame_width, frame_height, fps):
        """
        Initialize video writer with fallback codecs.
        
        Args:
            frame_width (int): Width of output video
            frame_height (int): Height of output video
            fps (int): Frames per second
        """
        codecs = ["avc1", "mp4v", "XVID"]
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
                if self.out.isOpened():
                    print(Fore.GREEN + f"Video writer ready with {codec} codec")
                    return
                self.out.release()
            except:
                continue
        
        if self.out is None or not self.out.isOpened():
            raise IOError("Failed to initialize video writer")
    
    def process_frame_with_tracking(self, frame, results):
        """
        Process a single frame with object tracking.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            annotated_frame: Frame with annotations and tracking
        """
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
                track = self.track_history[track_id]
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
            
        return annotated_frame
    
    def write_frame(self, frame):
        """Write frame to output video."""
        if self.out and self.out.isOpened():
            self.out.write(frame)
    
    def read_frame(self):
        """Read next frame from video."""
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
    
    def cleanup(self):
        """Release video capture and writer resources."""
        print(Fore.CYAN + "\nCleaning up video resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.out and self.out.isOpened():
            self.out.release()
        cv2.destroyAllWindows()
    
    def save_processing_info(self, frame_count, total_frames, start_time, end_time):
        """
        Save processing information to a text file in the output directory.
        
        Args:
            frame_count (int): Number of frames processed
            total_frames (int): Total frames in video
            start_time (float): Processing start time
            end_time (float): Processing end time
        """
        try:
            processing_time = end_time - start_time
            avg_fps = frame_count / processing_time if processing_time > 0 else 0
            
            info_path = os.path.join(self.get_output_directory(), "processing_info.txt")
            
            with open(info_path, 'w') as f:
                f.write("Object Detection - Processing Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Frames Processed: {frame_count}/{total_frames}\n")
                f.write(f"Processing Time: {processing_time:.2f} seconds\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n")
                f.write(f"Output Video: {os.path.basename(self.output_path)}\n")
                f.write(f"Output Directory: {self.get_output_directory()}\n")
            
            print(Fore.GREEN + f"Processing info saved to: {info_path}")
            
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not save processing info: {e}")


def calculate_progress_info(frame_count, total_frames, processing_time):
    """
    Calculate processing progress and FPS.
    
    Args:
        frame_count (int): Current frame number
        total_frames (int): Total number of frames
        processing_time (float): Time taken to process current frame
        
    Returns:
        tuple: (progress_percentage, fps_actual)
    """
    progress = (frame_count / total_frames) * 100
    fps_actual = 1.0 / processing_time if processing_time > 0 else 0
    return progress, fps_actual
