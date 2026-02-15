"""
Main Reconstruction Pipeline
Orchestrates the complete mechanical reconstruction process.
"""

import yaml
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger
import time

from .depth_estimator import DepthEstimator
from .screw_detector import ScrewDetector
from .mesh_generator import MeshGenerator, ScrewPose
from .urdf_exporter import URDFExporter


class MechanicalReconstructor:
    """
    Complete pipeline for hybrid mechanical reconstruction.
    
    Workflow:
    1. Detect screws (YOLO)
    2. Estimate depth (Depth Anything V2)
    3. Calibrate scale
    4. Generate point cloud & panel mesh
    5. Estimate screw 3D poses
    6. Generate parametric screws
    7. Fuse meshes
    8. Export OBJ & URDF
    """
    
    def __init__(
        self,
        depth_model_path: str,
        yolo_model_path: str,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict] = None
    ):
        """
        Initialize reconstruction pipeline.
        
        Args:
            depth_model_path: Path to Depth Anything V2 checkpoint
            yolo_model_path: Path to YOLOv8 model
            config_path: Path to YAML config file
            config_dict: Config dictionary (overrides config_path)
        """
        # Load configuration
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        logger.info("="*60)
        logger.info("MECHANICAL RECONSTRUCTION PIPELINE INITIALIZATION")
        logger.info("="*60)
        
        # Initialize components
        self._initialize_components(depth_model_path, yolo_model_path)
        
        # Calibration state
        self.scale_factor = None
        self.pixels_per_mm = None
        
        logger.success("Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from: {config_path}")
        return config
    
    def _initialize_components(self, depth_model_path: str, yolo_model_path: str):
        """Initialize all pipeline components."""
        # Depth estimator
        depth_config = self.config.get('models', {}).get('depth_anything', {})
        self.depth_estimator = DepthEstimator(
            checkpoint_path=depth_model_path,
            encoder=depth_config.get('encoder', 'vitl'),
            device=self.config.get('processing', {}).get('device', 'auto')
        )
        
        # Screw detector
        yolo_config = self.config.get('models', {}).get('yolo', {})
        self.screw_detector = ScrewDetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_config.get('confidence_threshold', 0.5),
            iou_threshold=yolo_config.get('iou_threshold', 0.4),
            device=self.config.get('processing', {}).get('device', 'auto')
        )
        
        # Mesh generator
        self.mesh_generator = MeshGenerator(self.config)
        
        # URDF exporter
        self.urdf_exporter = URDFExporter(self.config)
    
    def reconstruct(
        self,
        image_path: str,
        output_dir: str,
        save_intermediate: bool = None
    ) -> Dict:
        """
        Run complete reconstruction pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            save_intermediate: Whether to save intermediate results
        
        Returns:
            Dictionary with results and paths
        """
        start_time = time.time()
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_intermediate is None:
            save_intermediate = self.config.get('output', {}).get('save_intermediate', {}).get('enabled', True)
        
        logger.info("")
        logger.info("="*60)
        logger.info(f"STARTING RECONSTRUCTION: {image_path.name}")
        logger.info("="*60)
        
        results = {
            'image_path': str(image_path),
            'output_dir': str(output_dir),
            'intermediate': {}
        }
        
        # Step 1: Detect screws
        logger.info("\n[1/8] Detecting screws...")
        detections, image = self._detect_screws(image_path)
        results['num_screws'] = len(detections)
        results['detections'] = detections
        
        if len(detections) == 0:
            raise ValueError("No screws detected! Check YOLO model or image quality.")
        
        if save_intermediate:
            det_vis_path = output_dir / f"{image_path.stem}_detections.png"
            self.screw_detector.visualize_detections(image, detections, save_path=det_vis_path)
            results['intermediate']['detections'] = str(det_vis_path)
        
        # Step 2: Estimate depth
        logger.info("\n[2/8] Estimating depth...")
        depth_map = self._estimate_depth(image)
        results['depth_shape'] = depth_map.shape
        
        if save_intermediate:
            depth_vis_path = output_dir / f"{image_path.stem}_depth.png"
            self.depth_estimator.save_depth_visualization(depth_map, depth_vis_path)
            results['intermediate']['depth'] = str(depth_vis_path)
        
        # Step 3: Calibrate scale
        logger.info("\n[3/8] Calibrating scale...")
        self._calibrate_scale(detections, image.shape[:2])
        results['scale_factor'] = float(self.scale_factor)
        results['pixels_per_mm'] = float(self.pixels_per_mm)
        
        # Step 4: Generate point cloud
        logger.info("\n[4/8] Generating point cloud...")
        point_cloud = self.mesh_generator.generate_point_cloud(
            depth_map,
            image.shape[:2],
            depth_scale=self.scale_factor
        )
        results['point_cloud_size'] = len(point_cloud.points)
        
        if save_intermediate:
            pcd_path = output_dir / f"{image_path.stem}_pointcloud.ply"
            o3d.io.write_point_cloud(str(pcd_path), point_cloud)
            results['intermediate']['point_cloud'] = str(pcd_path)
        
        # Step 5: Reconstruct panel mesh
        logger.info("\n[5/8] Reconstructing panel mesh...")
        panel_mesh = self.mesh_generator.reconstruct_mesh(point_cloud)
        results['panel_vertices'] = len(panel_mesh.vertices)
        results['panel_triangles'] = len(panel_mesh.triangles)
        
        # Step 6: Estimate screw poses
        logger.info("\n[6/8] Estimating screw 3D poses...")
        screw_diameter_mm = self.config.get('calibration', {}).get('screw_diameter_mm', 3.0)
        screw_poses = self.mesh_generator.estimate_screw_poses(
            detections,
            depth_map,
            image.shape[:2],
            self.scale_factor,
            screw_diameter_mm
        )
        results['screw_poses'] = len(screw_poses)
        
        # Step 7: Generate and fuse screw meshes
        logger.info("\n[7/8] Generating parametric screws and fusing meshes...")
        screw_meshes = []
        segments = self.config.get('screws', {}).get('types', {}).get('default', {}).get('thread_segments', 32)
        
        for pose in screw_poses:
            screw_mesh = self.mesh_generator.generate_screw_mesh(pose, segments=segments)
            screw_meshes.append(screw_mesh)
        
        final_mesh = self.mesh_generator.fuse_meshes(panel_mesh, screw_meshes)
        results['final_vertices'] = len(final_mesh.vertices)
        results['final_triangles'] = len(final_mesh.triangles)
        
        # Step 8: Export
        logger.info("\n[8/8] Exporting results...")
        
        # Export OBJ
        obj_path = output_dir / f"{image_path.stem}.obj"
        o3d.io.write_triangle_mesh(str(obj_path), final_mesh)
        results['mesh_path'] = str(obj_path)
        logger.info(f"  ✓ OBJ mesh: {obj_path}")
        
        # Export URDF
        if self.config.get('export', {}).get('urdf', {}).get('enabled', True):
            urdf_results = self.urdf_exporter.export(
                final_mesh,
                output_dir,
                robot_name=image_path.stem
            )
            results['urdf_path'] = str(urdf_results['urdf'])
            results['visual_mesh_path'] = str(urdf_results['visual_mesh'])
            results['collision_mesh_path'] = str(urdf_results['collision_mesh'])
            logger.info(f"  ✓ URDF: {urdf_results['urdf']}")
        
        # Export STL (optional)
        if self.config.get('export', {}).get('stl', {}).get('enabled', False):
            stl_path = output_dir / f"{image_path.stem}.stl"
            ascii_stl = self.config.get('export', {}).get('stl', {}).get('ascii', False)
            self.urdf_exporter.export_stl(final_mesh, stl_path, ascii=ascii_stl)
            results['stl_path'] = str(stl_path)
            logger.info(f"  ✓ STL: {stl_path}")
        
        # Timing
        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed
        
        logger.info("")
        logger.info("="*60)
        logger.success(f"RECONSTRUCTION COMPLETE ({elapsed:.2f}s)")
        logger.info("="*60)
        logger.info(f"  Screws detected: {results['num_screws']}")
        logger.info(f"  Final vertices: {results['final_vertices']}")
        logger.info(f"  Final triangles: {results['final_triangles']}")
        logger.info(f"  Output: {output_dir}")
        logger.info("="*60)
        
        return results
    
    def _detect_screws(self, image_path: Path):
        """Detect screws in image."""
        detections, image = self.screw_detector.detect_from_file(str(image_path))
        
        # Filter detections
        min_conf = self.config.get('screws', {}).get('min_confidence', 0.5)
        detections = self.screw_detector.filter_detections(
            detections,
            min_confidence=min_conf
        )
        
        stats = self.screw_detector.get_detection_statistics(detections)
        logger.info(f"  Detected: {stats['count']} screws")
        logger.info(f"  Confidence: {stats['mean_confidence']:.3f} ± {stats['min_confidence']:.3f}-{stats['max_confidence']:.3f}")
        
        return detections, image
    
    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from image."""
        bilateral_config = self.config.get('depth', {}).get('bilateral_filter', {})
        
        bilateral = bilateral_config if bilateral_config.get('enabled', True) else None
        
        depth_map = self.depth_estimator.estimate_depth(
            image,
            normalize=True,
            bilateral_filter=bilateral
        )
        
        stats = self.depth_estimator.get_depth_statistics(depth_map)
        logger.info(f"  Depth range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Depth mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return depth_map
    
    def _calibrate_scale(self, detections: List, image_shape: tuple):
        """Calibrate scale from known screw dimensions."""
        calib_config = self.config.get('calibration', {})
        
        screw_diameter_mm = calib_config.get('screw_diameter_mm', 3.0)
        depth_range_mm = calib_config.get('depth_range_mm', 100.0)
        
        # Compute median screw bbox size
        bbox_widths = [d.bbox[2] for d in detections]
        bbox_heights = [d.bbox[3] for d in detections]
        bbox_sizes = [(w + h) / 2 for w, h in zip(bbox_widths, bbox_heights)]
        median_bbox_size = np.median(bbox_sizes)
        
        # Pixels per millimeter
        self.pixels_per_mm = median_bbox_size / screw_diameter_mm
        
        # Scale factor for depth (maps [0,1] to millimeters, then to meters)
        self.scale_factor = depth_range_mm / 1000.0  # Convert mm to meters
        
        # Estimate scene dimensions
        h, w = image_shape
        scene_width_mm = w / self.pixels_per_mm
        scene_height_mm = h / self.pixels_per_mm
        
        logger.info(f"  Pixels per mm: {self.pixels_per_mm:.2f}")
        logger.info(f"  Depth scale: {self.scale_factor*1000:.1f}mm")
        logger.info(f"  Scene size: {scene_width_mm:.1f}mm × {scene_height_mm:.1f}mm")
    
    def batch_reconstruct(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> List[Dict]:
        """
        Reconstruct multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Base output directory
        
        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            logger.info(f"{'='*60}\n")
            
            try:
                # Create subdirectory for this image
                image_name = Path(image_path).stem
                image_output_dir = output_dir / image_name
                
                # Reconstruct
                results = self.reconstruct(image_path, str(image_output_dir))
                results['status'] = 'success'
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                all_results.append({
                    'image_path': image_path,
                    'status': 'failed',
                    'error': str(e)
                })
                
                if not self.config.get('advanced', {}).get('error_handling', {}).get('continue_on_error', False):
                    raise
        
        # Summary
        successful = sum(1 for r in all_results if r['status'] == 'success')
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH COMPLETE: {successful}/{len(image_paths)} successful")
        logger.info(f"{'='*60}\n")
        
        return all_results


def test_pipeline():
    """Test function for pipeline."""
    config = {
        'models': {
            'depth_anything': {'encoder': 'vitl'},
            'yolo': {'confidence_threshold': 0.5}
        },
        'calibration': {
            'screw_diameter_mm': 3.0,
            'depth_range_mm': 100.0
        },
        'depth': {
            'bilateral_filter': {'enabled': True, 'diameter': 9}
        },
        'mesh': {
            'poisson': {'depth': 9}
        },
        'export': {
            'urdf': {'enabled': True}
        },
        'processing': {
            'device': 'auto'
        }
    }
    
    reconstructor = MechanicalReconstructor(
        depth_model_path='models/depth_anything_v2_vitl.pth',
        yolo_model_path='models/yolov8_screws.pt',
        config_dict=config
    )
    
    print("Pipeline initialized successfully")


if __name__ == '__main__':
    test_pipeline()
