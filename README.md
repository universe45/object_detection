# Object Detection System

A modular object detection system using YOLO for real-time tracking and detection with advanced model management.

## Project Structure

```
object_detection/
├── app/
│   ├── main.py                    # Main application entry point
│   ├── main_with_selector.py      # Enhanced app with interactive model selection
│   ├── config.py                  # Configuration settings
│   ├── object_detction.py         # Original monolithic file (kept for reference)
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py            # Package initialization
│   │   ├── device_manager.py      # GPU/CPU device detection and management
│   │   ├── file_validator.py      # File existence and validation utilities
│   │   ├── video_processor.py     # Video processing and tracking logic
│   │   ├── object_detector.py     # YOLO model loading and detection
│   │   ├── model_store.py         # Model management and selection system
│   │   └── model_manager.py       # Standalone model management utility
│   ├── model/                     # YOLO model files
│   └── media/                     # Media files (videos, images)
├── results/                       # Output results (auto-generated)
│   └── DDMMYYYY-HHMMSS/          # Timestamped folders
│       ├── output_result.mp4      # Processed video
│       └── processing_info.txt    # Processing metadata
├── requirements.txt               # Python dependencies
└── LICENSE
```

## Features

- **Modular Design**: Separated concerns into dedicated modules
- **Cross-Platform**: Automatic device detection (CUDA, MPS, CPU)
- **Object Tracking**: Real-time object tracking with ByteTrack
- **Progress Monitoring**: Real-time FPS and progress reporting
- **Timestamped Results**: Automatic creation of timestamped result folders
- **Directory Validation**: Automatic validation and creation of required directories
- **Interactive Model Selection**: Choose from multiple YOLO models with detailed specs
- **Model Management**: Comprehensive model store with performance metrics
- **Smart Recommendations**: Get model suggestions based on your use case
- **Processing Metadata**: Saves processing information alongside video output
- **Error Handling**: Robust error handling with fallback options
- **Configurable**: Easy configuration through config.py

## Output Structure

Each run creates a timestamped folder in the format `DDMMYYYY-HHMMSS`:

```
results/
├── 05082025-143022/           # Example: Aug 5, 2025 at 14:30:22
│   ├── output_result.mp4      # Processed video with detections
│   └── processing_info.txt    # Processing metadata and statistics
└── 05082025-151445/           # Another run
    ├── output_result.mp4
    └── processing_info.txt
```

## Model Management

### Available Models

The system supports multiple YOLO model variants:

#### YOLOv11 Detection Models
- **Nano (N)**: `yolo11n.pt` - 5.1MB, fastest inference
- **Small (S)**: `yolo11s.pt` - 19.8MB, balanced performance (default)
- **Medium (M)**: `yolo11m.pt` - 50.5MB, better accuracy
- **Large (L)**: `yolo11l.pt` - 85.8MB, high accuracy  
- **Extra Large (X)**: `yolo11x.pt` - 140.4MB, highest accuracy

#### Other Model Types
- **Classification**: YOLOv11 models for image classification
- **Segmentation**: YOLOv11 models for instance segmentation
- **Pose Estimation**: YOLOv11 models for human pose detection
- **Legacy**: YOLOv10 and YOLOv8 models

### Model Selection Strategies

1. **Speed Priority**: Use Nano (N) for real-time applications
2. **Balanced**: Use Small (S) for general purposes (recommended)
3. **Accuracy Priority**: Use Large (L) or Extra Large (X) for best results
4. **Resource Constrained**: Use Nano (N) or Small (S)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Models will be automatically downloaded when first used
3. Place your test video in the appropriate path (configurable in `config.py`)

## Usage

### Interactive Model Selection (Recommended):
```bash
cd app
python main_with_selector.py
```

### Model Management Utility:
```bash
cd app
python utils/model_manager.py
```

### Standard Application:
```bash
cd app
python main.py
```

### Test directory validation:
```bash
cd app
python test_directory.py
```

### Use individual modules:
```python
from utils import ObjectDetector, VideoProcessor, get_device, ModelSelector
from config import Config

# Model selection
selector = ModelSelector()
model_path = selector.select_model_interactive()

# Initialize components
device = get_device()
detector = ObjectDetector(model_path)
detector.load_model(device)

processor = VideoProcessor(Config.OUTPUT_FILENAME, Config.RESULTS_BASE_DIR)
# ... process video
```

### Quick Model Change in Config:
```python
# In config.py, simply change the model size:
Config.DEFAULT_MODEL_SIZE = "L"  # N, S, M, L, X
# Or use the helper method:
model_path = Config.get_model_by_size("M")
```

## Model Performance Comparison

| Model | Size | mAP50-95 | GPU Speed | CPU Speed | Best For |
|-------|------|----------|-----------|-----------|----------|
| Nano  | 5.1MB | 39.5 | 0.5ms | 2.4ms | Real-time, mobile |
| Small | 19.8MB | 47.0 | 0.7ms | 8.1ms | General purpose |
| Medium | 50.5MB | 51.5 | 1.2ms | 18.4ms | Better accuracy |
| Large | 85.8MB | 53.4 | 1.8ms | 27.6ms | High accuracy |
| X-Large | 140.4MB | 54.7 | 2.8ms | 49.2ms | Maximum accuracy |

## Configuration

Edit `app/config.py` to modify:
- File paths
- Detection thresholds
- Tracking parameters
- Video processing settings

## Modules

### `device_manager.py`
- Automatic device detection (CUDA/MPS/CPU)
- GPU memory management
- Cross-platform compatibility

### `file_validator.py`
- File existence validation
- Format validation for models and videos
- Error reporting

### `video_processor.py`
- Video capture and writing
- Object tracking visualization
- Frame processing pipeline

### `object_detector.py`
- YOLO model loading and management
- Object detection and tracking
- Configurable detection parameters

### `config.py`
- Centralized configuration
- Easy parameter tuning
- Default settings

## Benefits of Modular Structure

1. **Maintainability**: Each module has a single responsibility
2. **Reusability**: Components can be used independently
3. **Testing**: Easier to unit test individual components
4. **Scalability**: Easy to add new features or modify existing ones
5. **Readability**: Cleaner, more organized code structure
