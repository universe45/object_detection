from utils.model_store import YOLOModelList


class Config:
    """Configuration settings for the object detection system."""
    
    # Initialize model list for easy access
    models = YOLOModelList()
    
    # File paths
    MODEL_PATH = f"model/{models.YOLOv11.S}"  # Default to YOLOv11 Small
    VIDEO_PATH = "media/853889-hd_1920_1080_25fps.mp4"
    OUTPUT_FILENAME = "result.mp4"  # Just the filename, directory will be auto-generated
    RESULTS_BASE_DIR = "results"  # Base directory for all results
    
    # Quick model selection - uncomment the one you want to use
    # MODEL_PATH = f"./model/{models.YOLOv11.N}"    # Fastest
    # MODEL_PATH = f"./model/{models.YOLOv11.S}"    # Balanced (default)
    # MODEL_PATH = f"./model/{models.YOLOv11.M}"    # Better accuracy
    # MODEL_PATH = f"./model/{models.YOLOv11.L}"    # High accuracy
    # MODEL_PATH = f"./model/{models.YOLOv11.X}"    # Highest accuracy
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    TRACKER_CONFIG = "bytetrack.yaml"
    
    # Tracking settings
    MAX_TRACK_HISTORY = 30  # Number of frames to keep in track history
    
    # Video processing settings
    LINE_WIDTH = 2
    FONT_SIZE = 0.5
    TRACK_LINE_COLOR = (230, 230, 230)
    TRACK_LINE_THICKNESS = 2
    
    # Performance settings
    PROGRESS_LOG_INTERVAL = 1  # Log progress every N percent
    
    # Model selection preferences
    DEFAULT_MODEL_SIZE = "S"  # N, S, M, L, X
    AUTO_DOWNLOAD_MODELS = True  # Allow automatic model downloading
    
    @classmethod
    def get_model_by_size(cls, size: str) -> str:
        """Get model path by size (N, S, M, L, X)."""
        size_map = {
            "N": cls.models.YOLOv11.N,
            "S": cls.models.YOLOv11.S,
            "M": cls.models.YOLOv11.M,
            "L": cls.models.YOLOv11.L,
            "X": cls.models.YOLOv11.X
        }
        model_file = size_map.get(size.upper(), cls.models.YOLOv11.S)
        return f"./model/{model_file}"
