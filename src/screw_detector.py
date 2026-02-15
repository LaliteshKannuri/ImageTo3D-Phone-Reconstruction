"""
Screw Detector Module
Wraps YOLOv8 for automatic screw detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("Ultralytics YOLO not found. Please install: pip install ultralytics")
    YOLO = None


@dataclass
class ScrewDetection:
    """Container for screw detection results."""
    center: Tuple[float, float]  # (x, y) in pixels
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    confidence: float
    class_id: int = 0
    class_name: str = "screw"


class ScrewDetector:
    """
    Automatic screw detection using YOLOv8.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        device: str = 'auto'
    ):
        """
        Initialize screw detector.
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: Device to use (auto, cuda, cpu, or device ID)
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO not available. "
                "Install with: pip install ultralytics"
            )
        
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if self._cuda_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing ScrewDetector on {self.device}")
        logger.info(f"Model: {self.model_path.name}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Load model
        self._load_model()
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Set device
        self.model.to(self.device)
        
        logger.success(f"Model loaded successfully")
        
        # Get class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        logger.info(f"Classes: {self.class_names}")
    
    def detect(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> List[ScrewDetection]:
        """
        Detect screws in image.
        
        Args:
            image: RGB image (H, W, 3)
            visualize: Whether to return visualization
        
        Returns:
            List of ScrewDetection objects
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = self._parse_results(results[0])
        
        logger.info(f"Detected {len(detections)} screws")
        
        return detections
    
    def detect_from_file(
        self,
        image_path: str,
        visualize: bool = False
    ) -> Tuple[List[ScrewDetection], np.ndarray]:
        """
        Detect screws from image file.
        
        Args:
            image_path: Path to image
            visualize: Whether to return visualization
        
        Returns:
            (detections, image)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect
        detections = self.detect(image, visualize=visualize)
        
        return detections, image
    
    def _parse_results(self, result) -> List[ScrewDetection]:
        """
        Parse YOLO results into ScrewDetection objects.
        
        Args:
            result: YOLO result object
        
        Returns:
            List of detections
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract boxes
        boxes = result.boxes.xywh.cpu().numpy()  # (x_center, y_center, width, height)
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            conf = float(confidences[i])
            class_id = int(class_ids[i])
            
            detection = ScrewDetection(
                center=(float(x), float(y)),
                bbox=(float(x), float(y), float(w), float(h)),
                confidence=conf,
                class_id=class_id,
                class_name=self.class_names.get(class_id, f"class_{class_id}")
            )
            
            detections.append(detection)
        
        return detections
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[ScrewDetection],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: RGB image
            detections: List of detections
            save_path: Optional path to save visualization
        
        Returns:
            Annotated image
        """
        # Copy image
        vis_image = image.copy()
        
        # Draw each detection
        for det in detections:
            x, y, w, h = det.bbox
            
            # Convert to corner coordinates
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(vis_image, (cx, cy), 3, (255, 0, 0), -1)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Save if requested
        if save_path is not None:
            # Convert RGB to BGR for saving
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), vis_bgr)
            logger.info(f"Visualization saved: {save_path}")
        
        return vis_image
    
    def filter_detections(
        self,
        detections: List[ScrewDetection],
        min_confidence: Optional[float] = None,
        max_detections: Optional[int] = None,
        class_filter: Optional[List[str]] = None
    ) -> List[ScrewDetection]:
        """
        Filter detections based on criteria.
        
        Args:
            detections: Input detections
            min_confidence: Minimum confidence threshold
            max_detections: Maximum number of detections to keep
            class_filter: List of allowed class names
        
        Returns:
            Filtered detections
        """
        filtered = detections.copy()
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        # Filter by class
        if class_filter is not None:
            filtered = [d for d in filtered if d.class_name in class_filter]
        
        # Sort by confidence
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        
        # Limit number
        if max_detections is not None:
            filtered = filtered[:max_detections]
        
        return filtered
    
    def get_detection_statistics(
        self,
        detections: List[ScrewDetection]
    ) -> Dict:
        """
        Compute statistics of detections.
        
        Args:
            detections: List of detections
        
        Returns:
            Dictionary of statistics
        """
        if len(detections) == 0:
            return {
                'count': 0,
                'mean_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        confidences = [d.confidence for d in detections]
        
        return {
            'count': len(detections),
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'mean_bbox_area': float(np.mean([d.bbox[2] * d.bbox[3] for d in detections]))
        }
    
    def __repr__(self):
        return (
            f"ScrewDetector("
            f"model={self.model_path.name}, "
            f"conf={self.confidence_threshold}, "
            f"device={self.device})"
        )


def test_screw_detector():
    """Test function for ScrewDetector."""
    import matplotlib.pyplot as plt
    
    # Initialize
    detector = ScrewDetector(
        model_path='models/yolov8_screws.pt',
        confidence_threshold=0.5
    )
    
    # Test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect
    detections = detector.detect(test_image)
    
    # Print statistics
    stats = detector.get_detection_statistics(detections)
    print("Detection statistics:", stats)
    
    # Visualize
    vis_image = detector.visualize_detections(test_image, detections)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(vis_image)
    plt.title(f'Detected {len(detections)} screws')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('detection_test.png')
    print("Test visualization saved to detection_test.png")


if __name__ == '__main__':
    test_screw_detector()
