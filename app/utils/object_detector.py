import logging
from ultralytics import YOLO
from colorama import Fore


class ObjectDetector:
    """Handles YOLO model loading and object detection."""
    
    def __init__(self, model_path="./model/yolo11s.pt"):
        self.model_path = model_path
        self.model = None
        self.device = None
        
        # Suppress YOLO logs
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
    
    def load_model(self, device="cpu"):
        """
        Load the YOLO model.
        
        Args:
            device (str): Device to run inference on ('cuda', 'mps', 'cpu')
        """
        print(Fore.CYAN + "Loading YOLO model...")
        self.model = YOLO(self.model_path)
        self.device = device
        print(Fore.GREEN + "Model loaded successfully")
    
    def detect_objects(self, frame, conf_threshold=0.5, persist=True, tracker="bytetrack.yaml"):
        """
        Perform object detection and tracking on a frame.
        
        Args:
            frame: Input frame for detection
            conf_threshold (float): Confidence threshold for detections
            persist (bool): Whether to persist tracks between frames
            tracker (str): Tracker configuration file
            
        Returns:
            Detection results from YOLO
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = self.model.track(
            frame, 
            persist=persist, 
            tracker=tracker,
            conf=conf_threshold, 
            device=self.device, 
            verbose=False
        )
        
        return results
