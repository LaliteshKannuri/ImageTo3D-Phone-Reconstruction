# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Setup Models

```bash
# Create models directory
mkdir -p models

# Download Depth Anything V2 (ViT-Large)
# Visit: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
# Download depth_anything_v2_vitl.pth to models/

# Copy your trained YOLO model
cp /path/to/your/yolov8_screws.pt models/
```

## 3. Configure

Edit `config/config.yaml`:
```yaml
calibration:
  screw_diameter_mm: 3.0  # YOUR actual screw size!
```

## 4. Run!

```bash
# Single image
python scripts/run_reconstruction.py --image data/input/phone.jpg

# Check output
ls data/output/phone/
# Output: phone.obj, phone.urdf, etc.
```

## 5. Visualize

```bash
# View 3D mesh
python scripts/visualize.py view-mesh -m data/output/phone/phone.obj

# Or use PyBullet
python examples/example_usage.py
```

## What You Get

- `phone.obj` - 3D mesh with screws
- `phone.urdf` - For PyBullet simulation
- `phone_detections.png` - YOLO detections
- `phone_depth.png` - Depth map

## Troubleshooting

**No screws detected?**
- Lower confidence in config: `yolo.confidence_threshold: 0.3`
- Check image quality

**Wrong scale?**
- Measure real screw diameter
- Update `calibration.screw_diameter_mm` in config

**Slow?**
- Use GPU: `processing.device: "cuda"` in config
- Or reduce `mesh.poisson.depth: 7`

Done! 🎉
