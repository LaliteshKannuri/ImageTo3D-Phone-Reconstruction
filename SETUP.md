# Setup & Installation Guide

## Prerequisites

- Python 3.8+
- CUDA (optional, for GPU)
- Git

## Quick Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download models:
- Depth Anything V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
- Place in `models/depth_anything_v2_vitl.pth`
- Copy your YOLO model to `models/yolov8_screws.pt`

3. Test:
```bash
python scripts/run_reconstruction.py -i data/input/test.jpg
```

See full documentation in README.md
