# JAWJI Drone Detection

Real-time drone detection using YOLOv8 with optional tracking.

## Quick Start

```bash
# Install dependencies
pip install ultralytics opencv-python torch

# Run live detection
python jawji.py --mode live

# Run on video
python jawji.py --mode video --input path/to/video.mp4

# Run on images
python jawji.py --mode image --input path/to/image.jpg

# Enable tracking (optional)
python jawji.py --mode video --input video.mp4 --track
```

## Features
- Live camera detection
- Video file processing
- Single image detection
- Optional SORT-based tracking
- Automatic result saving

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8

## Project Structure
```
├── jawji.py          # Main detection script
├── data/             # Input data
│   ├── images/       
│   └── videos/       
├── models/          
│   └── jawji-finetune.pt  # Trained model
└── results/          # Output detections
```