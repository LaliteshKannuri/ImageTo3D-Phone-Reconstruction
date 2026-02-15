"""
Depth Estimator Module
Wraps Depth Anything V2 for monocular depth estimation.
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from loguru import logger

# Add Depth Anything V2 to path
# Assumes Depth-Anything-V2 is in parent directory or installed
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    logger.warning("Depth Anything V2 not found. Please add to path or install.")
    DepthAnythingV2 = None


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        }
    }
    
    def __init__(
        self,
        checkpoint_path: str,
        encoder: str = 'vitl',
        device: str = 'auto'
    ):
        """
        Initialize depth estimator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            encoder: Model size (vits, vitb, vitl)
            device: Device to use (auto, cuda, cpu)
        """
        if DepthAnythingV2 is None:
            raise ImportError(
                "Depth Anything V2 not available. "
                "Please install or add to PYTHONPATH."
            )
        
        self.encoder = encoder
        self.checkpoint_path = Path(checkpoint_path)
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing DepthEstimator on {self.device}")
        logger.info(f"Model: {encoder}, Checkpoint: {self.checkpoint_path.name}")
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load Depth Anything V2 model."""
        if self.encoder not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid encoder: {self.encoder}. "
                f"Choose from {list(self.MODEL_CONFIGS.keys())}"
            )
        
        config = self.MODEL_CONFIGS[self.encoder]
        
        # Create model
        self.model = DepthAnythingV2(**config)
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        state_dict = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        logger.success(f"Model loaded successfully")
    
    def estimate_depth(
        self,
        image: np.ndarray,
        normalize: bool = True,
        bilateral_filter: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            image: RGB image (H, W, 3) as numpy array
            normalize: Whether to normalize depth to [0, 1]
            bilateral_filter: Bilateral filter params (diameter, sigma_color, sigma_space)
        
        Returns:
            Depth map (H, W) as numpy array
        """
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Run inference
        with torch.no_grad():
            depth = self.model.infer_image(image)
        
        # Normalize if requested
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                logger.warning("Depth map has no variation, setting to zeros")
                depth = np.zeros_like(depth)
        
        # Apply bilateral filter for smoothing
        if bilateral_filter is not None:
            depth = self._apply_bilateral_filter(depth, bilateral_filter)
        
        return depth
    
    def _apply_bilateral_filter(
        self,
        depth: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Apply bilateral filter to depth map.
        
        Args:
            depth: Input depth map
            params: Filter parameters
        
        Returns:
            Filtered depth map
        """
        diameter = params.get('diameter', 9)
        sigma_color = params.get('sigma_color', 0.1)
        sigma_space = params.get('sigma_space', 5)
        
        # Convert to float32 for filtering
        depth_float = depth.astype(np.float32)
        
        # Apply filter
        filtered = cv2.bilateralFilter(
            depth_float,
            diameter,
            sigma_color,
            sigma_space
        )
        
        return filtered
    
    def estimate_depth_from_file(
        self,
        image_path: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from image file.
        
        Args:
            image_path: Path to image file
            **kwargs: Additional arguments for estimate_depth
        
        Returns:
            (depth_map, original_image)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Estimate depth
        depth = self.estimate_depth(image, **kwargs)
        
        return depth, image
    
    def save_depth_visualization(
        self,
        depth: np.ndarray,
        output_path: str,
        colormap: int = cv2.COLORMAP_INFERNO
    ):
        """
        Save depth map as colored visualization.
        
        Args:
            depth: Depth map
            output_path: Output file path
            colormap: OpenCV colormap
        """
        # Normalize to [0, 255]
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)
        
        # Save
        cv2.imwrite(str(output_path), depth_colored)
        logger.info(f"Depth visualization saved: {output_path}")
    
    def get_depth_statistics(self, depth: np.ndarray) -> Dict:
        """
        Compute statistics of depth map.
        
        Args:
            depth: Depth map
        
        Returns:
            Dictionary of statistics
        """
        return {
            'min': float(depth.min()),
            'max': float(depth.max()),
            'mean': float(depth.mean()),
            'std': float(depth.std()),
            'median': float(np.median(depth)),
            'shape': depth.shape
        }
    
    def __repr__(self):
        return (
            f"DepthEstimator("
            f"encoder={self.encoder}, "
            f"device={self.device})"
        )


def test_depth_estimator():
    """Test function for DepthEstimator."""
    import matplotlib.pyplot as plt
    
    # Initialize
    estimator = DepthEstimator(
        checkpoint_path='models/depth_anything_v2_vitl.pth',
        encoder='vitl'
    )
    
    # Test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Estimate depth
    depth = estimator.estimate_depth(test_image, bilateral_filter={
        'diameter': 9,
        'sigma_color': 0.1,
        'sigma_space': 5
    })
    
    # Print statistics
    stats = estimator.get_depth_statistics(depth)
    print("Depth statistics:", stats)
    
    # Visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(test_image)
    plt.title('Input Image')
    plt.subplot(122)
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.title('Depth Map')
    plt.tight_layout()
    plt.savefig('depth_test.png')
    print("Test visualization saved to depth_test.png")


if __name__ == '__main__':
    test_depth_estimator()
