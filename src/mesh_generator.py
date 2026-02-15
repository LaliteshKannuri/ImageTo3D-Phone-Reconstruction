"""
Mesh Generator Module
Handles point cloud generation, surface reconstruction, parametric primitives, and mesh fusion.
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .geometry_utils import (
    backproject_pixel,
    estimate_surface_normal,
    rotation_from_normal,
    compute_camera_intrinsics
)
from .screw_detector import ScrewDetection


@dataclass
class ScrewPose:
    """3D pose of a screw."""
    position: np.ndarray  # (x, y, z) in meters
    normal: np.ndarray    # Surface normal
    radius: float         # Radius in meters
    height: float         # Height in meters
    confidence: float
    detection: ScrewDetection  # Original detection


class MeshGenerator:
    """
    Generates 3D meshes from depth maps and screw detections.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize mesh generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Initialized MeshGenerator")
    
    def generate_point_cloud(
        self,
        depth_map: np.ndarray,
        image_shape: Tuple[int, int],
        depth_scale: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        Generate point cloud from depth map.
        
        Args:
            depth_map: Depth map (normalized or in meters)
            image_shape: (height, width) of image
            depth_scale: Scale factor to convert depth to meters
        
        Returns:
            Open3D point cloud
        """
        h, w = depth_map.shape
        
        # Compute camera intrinsics
        fx, fy, cx, cy = compute_camera_intrinsics(w, h)
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Scale depth to meters
        depth_meters = depth_map * depth_scale
        
        # Backproject to 3D
        X = (u - cx) * depth_meters / fx
        Y = (v - cy) * depth_meters / fy
        Z = depth_meters
        
        # Stack into point cloud
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        
        # Remove invalid points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Downsampling
        voxel_size = self.config.get('depth', {}).get('point_cloud', {}).get('downsample_voxel_size', 0.001)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
            logger.info(f"Point cloud downsampled to {len(pcd.points)} points")
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        
        # Remove outliers
        if self.config.get('depth', {}).get('point_cloud', {}).get('remove_outliers', True):
            pcd = self._remove_outliers(pcd)
        
        logger.info(f"Generated point cloud with {len(pcd.points)} points")
        
        return pcd
    
    def _remove_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Remove statistical outliers from point cloud."""
        nb_neighbors = self.config.get('depth', {}).get('point_cloud', {}).get('outlier_neighbors', 20)
        std_ratio = self.config.get('depth', {}).get('point_cloud', {}).get('outlier_std_ratio', 2.0)
        
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        
        return pcd_filtered
    
    def reconstruct_mesh(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct mesh with AGGRESSIVE artifact removal.
        """
        # Poisson reconstruction
        poisson_config = self.config.get('mesh', {}).get('poisson', {})
        
        logger.info("Running Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=poisson_config.get('depth', 8),
            width=poisson_config.get('width', 0),
            scale=poisson_config.get('scale', 1.05),
            linear_fit=poisson_config.get('linear_fit', False)
        )
        
        # 🔥 CRITICAL FIX 1: Aggressive density filtering
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.10)  # Bottom 10%
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        logger.info(f"Removed {vertices_to_remove.sum()} low-density vertices")
        
        # 🔥 CRITICAL FIX 2: Crop to point cloud bounding box
        pcd_bbox = point_cloud.get_axis_aligned_bounding_box()
        bbox_min = pcd_bbox.min_bound * 0.95
        bbox_max = pcd_bbox.max_bound * 1.05
        
        vertices = np.asarray(mesh.vertices)
        inside_mask = np.all(
            (vertices >= bbox_min) & (vertices <= bbox_max),
            axis=1
        )
        mesh.remove_vertices_by_mask(~inside_mask)
        logger.info(f"Cropped to bounding box")
        
        # 🔥 CRITICAL FIX 3: Keep only largest connected component
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 1:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            logger.info(f"Kept largest component: {cluster_n_triangles[largest_cluster_idx]} triangles")
        
        # 🔥 CRITICAL FIX 4: Laplacian smoothing
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
        
        # Standard cleanup
        mesh = self._clean_mesh(mesh)
        
        # 🔥 CRITICAL FIX 5: Optional simplification
        postproc_config = self.config.get('mesh_postprocessing', {})
        if postproc_config.get('simplification', {}).get('enabled', True):
            target_reduction = postproc_config['simplification'].get('target_reduction', 0.3)
            target_triangles = int(len(mesh.triangles) * (1 - target_reduction))
            if target_triangles > 1000:
                mesh = mesh.simplify_quadric_decimation(target_triangles)
                logger.info(f"Simplified to {len(mesh.triangles)} triangles")
        
        logger.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        return mesh
    
    def _clean_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Clean up mesh by removing artifacts."""
        cleanup_config = self.config.get('mesh', {}).get('cleanup', {})
        
        if cleanup_config.get('remove_degenerate_triangles', True):
            mesh.remove_degenerate_triangles()
        
        if cleanup_config.get('remove_duplicated_vertices', True):
            mesh.remove_duplicated_vertices()
        
        if cleanup_config.get('remove_duplicated_triangles', True):
            mesh.remove_duplicated_triangles()
        
        if cleanup_config.get('remove_non_manifold_edges', False):
            mesh.remove_non_manifold_edges()
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh
    
    def estimate_screw_poses(
        self,
        detections: List[ScrewDetection],
        depth_map: np.ndarray,
        image_shape: Tuple[int, int],
        depth_scale: float,
        screw_diameter_mm: float
    ) -> List[ScrewPose]:
        """
        Estimate 3D poses of screws from detections and depth.
        
        Args:
            detections: List of screw detections
            depth_map: Depth map
            image_shape: Image shape (h, w)
            depth_scale: Scale factor for depth
            screw_diameter_mm: Known screw diameter in mm
        
        Returns:
            List of screw poses
        """
        h, w = depth_map.shape
        fx, fy, cx, cy = compute_camera_intrinsics(w, h)
        
        screw_poses = []
        
        # Get screw configuration
        screw_config = self.config.get('screws', {}).get('types', {}).get('default', {})
        screw_height_mm = screw_config.get('height_mm', 2.5)
        normal_window = self.config.get('screws', {}).get('normal_estimation', {}).get('window_size', 5)
        
        for det in detections:
            # Get center position
            u, v = det.center
            u, v = int(u), int(v)
            
            # Validate coordinates
            if not (0 <= u < w and 0 <= v < h):
                logger.warning(f"Screw center out of bounds: ({u}, {v})")
                continue
            
            # Get depth with local averaging for robustness
            u_min = max(0, u - 2)
            u_max = min(w, u + 3)
            v_min = max(0, v - 2)
            v_max = min(h, v + 3)
            
            local_depth = depth_map[v_min:v_max, u_min:u_max]
            depth_value = np.median(local_depth)
            
            # Convert to meters
            depth_meters = depth_value * depth_scale
            
            # Backproject to 3D
            position = backproject_pixel(u, v, depth_meters, cx, cy, fx, fy)
            
            # Estimate surface normal
            normal = estimate_surface_normal(
                depth_map,
                u, v,
                window_size=normal_window,
                fx=fx,
                fy=fy
            )
            
            # Estimate screw radius from bounding box
            bbox_width = det.bbox[2]
            bbox_height = det.bbox[3]
            bbox_size = (bbox_width + bbox_height) / 2
            
            # Pixels per millimeter (using known screw diameter)
            pixels_per_mm = bbox_size / screw_diameter_mm
            
            # Screw radius in meters
            radius_meters = (screw_diameter_mm / 2) / 1000.0
            height_meters = screw_height_mm / 1000.0
            
            # Create pose
            pose = ScrewPose(
                position=position,
                normal=normal,
                radius=radius_meters,
                height=height_meters,
                confidence=det.confidence,
                detection=det
            )
            
            screw_poses.append(pose)
        
        logger.info(f"Estimated poses for {len(screw_poses)} screws")
        
        return screw_poses
    
    def generate_screw_mesh(
        self,
        pose: ScrewPose,
        segments: int = 32
    ) -> o3d.geometry.TriangleMesh:
        """
        Generate parametric mesh for a screw.
        
        Args:
            pose: Screw pose
            segments: Number of segments for cylinder
        
        Returns:
            Screw mesh
        """
        # Create cylinder
        screw_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=pose.radius,
            height=pose.height,
            resolution=segments,
            split=4
        )
        
        # Rotate to align with surface normal
        # Cylinder is created along Z-axis, need to align with normal
        R = rotation_from_normal(pose.normal)
        screw_mesh.rotate(R, center=(0, 0, 0))
        
        # Translate to position
        screw_mesh.translate(pose.position)
        
        # Compute normals
        screw_mesh.compute_vertex_normals()
        
        return screw_mesh
    
    def fuse_meshes(
        self,
        panel_mesh: o3d.geometry.TriangleMesh,
        screw_meshes: List[o3d.geometry.TriangleMesh]
    ) -> o3d.geometry.TriangleMesh:
        """
        Fuse panel and screw meshes.
        
        Args:
            panel_mesh: Base panel mesh
            screw_meshes: List of screw meshes
        
        Returns:
            Fused mesh
        """
        fusion_config = self.config.get('fusion', {})
        use_boolean = fusion_config.get('boolean_ops', {}).get('enabled', True)
        
        if use_boolean and len(screw_meshes) > 0:
            try:
                final_mesh = self._boolean_fusion(panel_mesh, screw_meshes)
            except Exception as e:
                logger.warning(f"Boolean fusion failed: {e}")
                if fusion_config.get('fallback_to_concatenation', True):
                    logger.info("Falling back to simple concatenation")
                    final_mesh = self._simple_fusion(panel_mesh, screw_meshes)
                else:
                    raise
        else:
            final_mesh = self._simple_fusion(panel_mesh, screw_meshes)
        
        # Final cleanup
        final_mesh = self._clean_mesh(final_mesh)
        
        logger.info(f"Fused mesh: {len(final_mesh.vertices)} vertices, {len(final_mesh.triangles)} triangles")
        
        return final_mesh
    
    def _boolean_fusion(
        self,
        panel_mesh: o3d.geometry.TriangleMesh,
        screw_meshes: List[o3d.geometry.TriangleMesh]
    ) -> o3d.geometry.TriangleMesh:
        """Fuse meshes using boolean operations (requires trimesh)."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("Trimesh required for boolean operations: pip install trimesh")
        
        # Convert to trimesh
        panel_tm = trimesh.Trimesh(
            vertices=np.asarray(panel_mesh.vertices),
            faces=np.asarray(panel_mesh.triangles)
        )
        
        # Union with each screw
        for screw_mesh in screw_meshes:
            screw_tm = trimesh.Trimesh(
                vertices=np.asarray(screw_mesh.vertices),
                faces=np.asarray(screw_mesh.triangles)
            )
            
            panel_tm = panel_tm.union(screw_tm, engine='blender')
        
        # Convert back to Open3D
        final_mesh = o3d.geometry.TriangleMesh()
        final_mesh.vertices = o3d.utility.Vector3dVector(panel_tm.vertices)
        final_mesh.triangles = o3d.utility.Vector3iVector(panel_tm.faces)
        
        return final_mesh
    
    def _simple_fusion(
        self,
        panel_mesh: o3d.geometry.TriangleMesh,
        screw_meshes: List[o3d.geometry.TriangleMesh]
    ) -> o3d.geometry.TriangleMesh:
        """Simple mesh fusion by concatenation."""
        final_mesh = o3d.geometry.TriangleMesh(panel_mesh)
        
        for screw_mesh in screw_meshes:
            final_mesh += screw_mesh
        
        return final_mesh
    
    def simplify_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_reduction: float = 0.5
    ) -> o3d.geometry.TriangleMesh:
        """
        Simplify mesh by reducing triangle count.
        
        Args:
            mesh: Input mesh
            target_reduction: Fraction of triangles to remove (0-1)
        
        Returns:
            Simplified mesh
        """
        target_triangles = int(len(mesh.triangles) * (1 - target_reduction))
        
        simplified = mesh.simplify_quadric_decimation(target_triangles)
        simplified.compute_vertex_normals()
        
        logger.info(f"Simplified mesh: {len(simplified.triangles)} triangles (from {len(mesh.triangles)})")
        
        return simplified


def test_mesh_generator():
    """Test function for MeshGenerator."""
    # Test configuration
    config = {
        'depth': {
            'point_cloud': {
                'downsample_voxel_size': 0.001,
                'remove_outliers': True
            }
        },
        'mesh': {
            'poisson': {'depth': 8},
            'cleanup': {'remove_degenerate_triangles': True}
        },
        'screws': {
            'types': {'default': {'height_mm': 2.5}}
        }
    }
    
    generator = MeshGenerator(config)
    
    # Test point cloud generation
    depth_map = np.random.rand(480, 640)
    pcd = generator.generate_point_cloud(depth_map, (480, 640), depth_scale=0.1)
    
    print(f"Generated point cloud with {len(pcd.points)} points")
    
    # Test mesh reconstruction
    mesh = generator.reconstruct_mesh(pcd)
    
    print(f"Reconstructed mesh with {len(mesh.vertices)} vertices")


if __name__ == '__main__':
    test_mesh_generator()
