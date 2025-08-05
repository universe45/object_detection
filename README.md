# YOLO Object Detection System

A highly optimized, modular system for object detection and tracking using YOLO models. The system features cross-platform GPU/CPU acceleration, real-time tracking, and a comprehensive model management system.

![YOLO Object Detection](https://img.shields.io/badge/YOLO-v11-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Cross--Platform-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This system provides a complete end-to-end solution for object detection and tracking in videos using state-of-the-art YOLO models. It features automatic device optimization, interactive model selection, and real-time performance monitoring.

## Project Structure

```
object_detection/
├── app/
│   ├── main.py                    # Standard application entry point
│   ├── main_with_selector.py      # Enhanced app with model selection UI
│   ├── config.py                  # Centralized configuration settings
│   ├── utils/                     # Modular utility components
│   │   ├── device_manager.py      # Cross-platform GPU/CPU optimization
│   │   ├── file_validator.py      # File and directory validation
│   │   ├── video_processor.py     # Video handling and visualization
│   │   ├── object_detector.py     # YOLO model integration
│   │   ├── model_store.py         # Model catalog and selection system
│   │   └── model_manager.py       # Standalone model management utility
│   ├── model/                     # YOLO model storage
│   └── media/                     # Input video files
├── results/                       # Processed outputs (auto-generated)
│   └── DDMMYYYY-HHMMSS/          # Timestamped result folders
│       ├── result.mp4             # Processed video with annotations
│       └── processing_info.txt    # Performance metadata and stats
├── requirements.txt               # Python dependencies
└── LICENSE                        # Project license
```

## Key Features

### Performance & Compatibility
- **Cross-Platform Acceleration**: Automatic detection of CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- **Real-time Processing**: Optimized for speed with performance metrics display
- **Smart Device Management**: Memory optimization and GPU resource handling

### Detection & Tracking
- **Advanced Object Detection**: High-accuracy YOLOv11 model integration
- **Real-time Tracking**: Object tracking with ByteTrack algorithm
- **Motion Path Visualization**: Trajectory lines showing object movement paths

### User Experience
- **Interactive Model Selection**: Visual interface for choosing optimal models
- **Use-Case Recommendations**: AI model recommendations based on specific needs
- **Progress Monitoring**: Live FPS and percentage completion updates
- **Timestamped Results**: Automatically organized output with metadata

### Architecture
- **Modular Design**: Clean separation of concerns for maintainability
- **Robust Error Handling**: Automatic fallbacks and recovery mechanisms
- **Centralized Configuration**: Easy customization through config.py

## Output System

Each detection run creates a timestamped output directory for organization and reproducibility:

```
results/
├── 05082025-143022/           # Example: Aug 5, 2025 at 14:30:22
│   ├── result.mp4             # Processed video with detection annotations
│   └── processing_info.txt    # Performance data and processing statistics
└── 05082025-151445/           # Subsequent run with different timestamp
    ├── result.mp4
    └── processing_info.txt
```

The processing info file contains valuable metadata:
- Processing timestamp
- Total frames processed
- Processing duration
- Average FPS achieved
- Output paths and configurations

## Model Ecosystem

The system provides access to a diverse range of YOLO models to suit different requirements and hardware constraints.

### Detection Models (Primary)

| Model | Size | Performance | GPU Speed | CPU Speed | Best For |
|-------|------|------------|-----------|-----------|----------|
| **YOLOv11n** | 5.1MB | mAP 39.5 | 0.5ms | 2.4ms | Mobile devices, real-time applications |
| **YOLOv11s** | 19.8MB | mAP 47.0 | 0.7ms | 8.1ms | General use, balanced (default) |
| **YOLOv11m** | 50.5MB | mAP 51.5 | 1.2ms | 18.4ms | Better accuracy needs |
| **YOLOv11l** | 85.8MB | mAP 53.4 | 1.8ms | 27.6ms | High accuracy requirements |
| **YOLOv11x** | 140.4MB | mAP 54.7 | 2.8ms | 49.2ms | Maximum precision needs |

### Additional Model Types

The system also supports specialized YOLO models for various computer vision tasks:

- **Classification Models**: Image classification (`yolo11*-cls.pt`)
- **Segmentation Models**: Instance segmentation (`yolo11*-seg.pt`)
- **Pose Estimation**: Human pose detection (`yolo11*-pose.pt`)
- **Legacy Models**: Support for YOLOv10 and YOLOv8

### Selection Strategy Guide

- **Speed Priority**: Choose Nano (N) for fastest processing
- **Balanced Performance**: Use Small (S) for general applications 
- **Accuracy Focus**: Select Large (L) or Extra Large (X)
- **Resource Constrained**: Opt for Nano (N) or Small (S)
- **Special Tasks**: Use task-specific models for classification, segmentation or pose

## Installation

### Prerequisites

- Python 3.8+ required
- CUDA-compatible NVIDIA GPU recommended for optimal performance
- Apple Silicon Mac for MPS acceleration support
- For CPU-only, no special hardware required

### Method 1: Using pip

```bash
# Clone the repository
git clone https://github.com/universe45/object_detection.git
cd object_detection

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/universe45/object_detection.git
cd object_detection

# Create and activate Conda environment
conda create -n object_detection python=3.11
conda activate object_detection

# Install core dependencies
conda install -c pytorch pytorch torchvision
conda install -c conda-forge opencv numpy
conda install -c conda-forge colorama

# Install ultralytics (not available in conda)
pip install ultralytics
```

### GPU Acceleration Setup

#### For NVIDIA GPUs:
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### For Apple Silicon (M1/M2/M3):
```bash
# MPS acceleration is included with PyTorch 2.0+
# No additional setup required
```

### Post-Installation
- Models will be automatically downloaded when first used
- Place input videos in `app/media/` directory
- Configuration can be customized in `app/config.py`

## Usage Guide

### Running the Application

#### Interactive Mode (Recommended)

Launch the application with an interactive model selection interface:

```bash
cd app
python main_with_selector.py
```

This mode provides:
- Visual selection from all available models
- Model recommendations based on use case
- Detailed model specifications display
- Performance metrics for informed decision making

#### Standard Mode

Run with the default model specified in config:

```bash
cd app
python main.py
```

#### Model Management Utility

Access the standalone model management tool:

```bash
cd app
python utils/model_manager.py
```

### Customization

#### 1. Quick Model Switching

Edit `app/config.py` to quickly change the default model:

```python
# Change this line to switch default model size
Config.DEFAULT_MODEL_SIZE = "L"  # N, S, M, L, X

# Or use the helper method
model_path = Config.get_model_by_size("M")
```

#### 2. API Usage

Integrate into your own Python applications:

```python
from utils import ObjectDetector, VideoProcessor, get_device, ModelSelector
from config import Config

# Select model interactively or programmatically
selector = ModelSelector()
model_path = selector.select_model_interactive()
# Or: model_path = selector.recommend_model("accuracy")

# Initialize components with hardware optimization
device = get_device()  # Automatically selects CUDA, MPS, or CPU
detector = ObjectDetector(model_path)
detector.load_model(device)

# Set up video processing pipeline
processor = VideoProcessor(Config.OUTPUT_FILENAME, Config.RESULTS_BASE_DIR)

# Process video frames (simplified)
while True:
    success, frame = processor.read_frame()
    if not success:
        break
    
    # Detect objects and visualize tracks
    results = detector.detect_objects(frame, conf_threshold=0.5)
    annotated_frame = processor.process_frame_with_tracking(frame, results)
    processor.write_frame(annotated_frame)
```

## Configuration

The system is designed to be highly configurable through the `app/config.py` file:

```python
class Config:
    # File paths
    MODEL_PATH = f"model/yolo11s.pt"      # Default model
    VIDEO_PATH = "media/video_file.mp4"    # Input video
    OUTPUT_FILENAME = "result.mp4"         # Output filename
    RESULTS_BASE_DIR = "results"           # Results directory

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5             # Detection confidence threshold
    TRACKER_CONFIG = "bytetrack.yaml"      # Tracking algorithm config
    MAX_TRACK_HISTORY = 30                 # Tracking history length

    # Visual settings
    LINE_WIDTH = 2                         # Bounding box line width
    FONT_SIZE = 0.5                        # Text font size
    TRACK_LINE_COLOR = (230, 230, 230)     # Tracking line color
    
    # Performance settings
    PROGRESS_LOG_INTERVAL = 1              # Progress logging frequency
```

## System Architecture

### Core Components

#### 1. Device Manager (`device_manager.py`)
- Auto-detects optimal hardware (NVIDIA CUDA, Apple MPS, CPU)
- Platform-specific optimizations for Windows, macOS, Linux
- GPU memory management for efficient resource utilization

#### 2. Object Detector (`object_detector.py`)
- YOLO model integration with ultralytics framework
- Configurable detection parameters and thresholds
- Object tracking with ByteTrack algorithm

#### 3. Video Processor (`video_processor.py`)
- Video capture and frame extraction
- Real-time tracking visualization
- Results storage with performance metadata

#### 4. Model Store (`model_store.py`)
- Comprehensive model catalog with performance metrics
- Selection interface with recommendations
- Auto-download capability for missing models

#### 5. File Validator (`file_validator.py`)
- Validates input files and formats
- Creates directory structures
- Ensures proper permissions and access

## Development and Extension

### Modular Architecture Benefits

The system is built with a modular architecture providing several advantages:

1. **Independent Components**: Each module can be used separately
2. **Simplified Testing**: Unit testing is straightforward with isolated components
3. **Easy Extension**: Add new features without affecting existing functionality
4. **Clear Responsibility**: Each module has a single well-defined purpose
5. **Maintainable Code**: Changes in one area don't affect others

### Extension Points

The system can be extended in various ways:

- **Custom Models**: Add new YOLO or custom models to the model store
- **New Tracking Algorithms**: Integrate different tracking approaches
- **Additional Visualizations**: Enhance the visual output with new features
- **Alternative Backends**: Integrate with other ML frameworks beyond ultralytics
- **Cloud Integration**: Add cloud storage or processing capabilities

## Performance Tips

- Use GPU acceleration when available for optimal performance
- Select model size based on your specific performance/accuracy needs
- For real-time applications, use Nano or Small models
- For maximum accuracy without time constraints, use Large or X-Large models
- Adjust confidence threshold based on specific detection scenarios
