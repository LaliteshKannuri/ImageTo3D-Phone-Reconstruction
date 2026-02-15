"""
Geometry Utilities for 3D Reconstruction
Handles coordinate transformations, projections, and geometric operations.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


def backproject_pixel(
    u: float,
    v: float,
    depth: float,
    cx: float,
    cy: float,
    fx: float,
    fy: float
) -> np.ndarray:
    """
    Backproject a 2D pixel to 3D using pinhole camera model.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value at pixel
        cx, cy: Principal point
        fx, fy: Focal lengths
    
    Returns:
        3D point [X, Y, Z]
    """
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    return np.array([X, Y, Z])


def project_point(
    point_3d: np.ndarray,
    cx: float,
    cy: float,
    fx: float,
    fy: float
) -> Tuple[float, float]:
    """
    Project a 3D point to 2D image plane.
    
    Args:
        point_3d: 3D point [X, Y, Z]
        cx, cy: Principal point
        fx, fy: Focal lengths
    
    Returns:
        Pixel coordinates (u, v)
    """
    X, Y, Z = point_3d
    
    if Z < 1e-6:
        raise ValueError("Point is behind camera (Z <= 0)")
    
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    
    return u, v


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.
    
    Args:
        v: Input vector
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    
    if norm < 1e-8:
        return v
    
    return v / norm


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Find rotation matrix that aligns vec1 to vec2.
    
    Args:
        vec1: Source vector
        vec2: Target vector
    
    Returns:
        3x3 rotation matrix
    """
    vec1 = normalize_vector(vec1)
    vec2 = normalize_vector(vec2)
    
    # Compute rotation axis and angle
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(v)
    
    # Handle special cases
    if s < 1e-6:
        # Vectors are parallel
        if c > 0:
            return np.eye(3)
        else:
            # Vectors are anti-parallel - rotate 180 degrees
            # Find any perpendicular vector
            perp = np.cross(vec1, np.array([1, 0, 0]))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(vec1, np.array([0, 1, 0]))
            perp = normalize_vector(perp)
            
            return Rotation.from_rotvec(np.pi * perp).as_matrix()
    
    # Rodrigues' rotation formula
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    return R


def rotation_from_normal(normal: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align Z-axis with given normal vector.
    
    Args:
        normal: Surface normal vector
    
    Returns:
        3x3 rotation matrix
    """
    z_axis = np.array([0, 0, 1])
    return rotation_matrix_from_vectors(z_axis, normal)


def estimate_surface_normal(
    depth_map: np.ndarray,
    u: int,
    v: int,
    window_size: int = 5,
    fx: float = 1.0,
    fy: float = 1.0
) -> np.ndarray:
    """
    Estimate surface normal from local depth gradient.
    
    Args:
        depth_map: Depth image
        u, v: Pixel coordinates
        window_size: Size of local neighborhood
        fx, fy: Focal lengths (for proper scaling)
    
    Returns:
        Surface normal vector
    """
    h, w = depth_map.shape
    
    # Define local window
    u_min = max(0, u - window_size)
    u_max = min(w, u + window_size + 1)
    v_min = max(0, v - window_size)
    v_max = min(h, v + window_size + 1)
    
    local_depth = depth_map[v_min:v_max, u_min:u_max]
    
    # Compute gradients
    gy, gx = np.gradient(local_depth)
    
    # Average gradients in window
    gx_mean = np.mean(gx)
    gy_mean = np.mean(gy)
    
    # Normal is perpendicular to gradient
    # Account for camera model: depth gradient needs to be scaled by focal length
    normal = np.array([-gx_mean / fx, -gy_mean / fy, 1.0])
    normal = normalize_vector(normal)
    
    return normal


def estimate_plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a plane to a set of 3D points using PCA.
    
    Args:
        points: Nx3 array of points
    
    Returns:
        (normal, offset): Plane parameters (normal . p = offset)
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(centered)
    
    # Normal is the last right singular vector
    normal = Vt[-1]
    
    # Ensure normal points in positive Z direction
    if normal[2] < 0:
        normal = -normal
    
    # Compute offset
    offset = np.dot(normal, centroid)
    
    return normal, offset


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box of points.
    
    Args:
        points: Nx3 array of points
    
    Returns:
        (min_point, max_point): Corners of bounding box
    """
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    
    return min_point, max_point


def compute_oriented_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute oriented bounding box using PCA.
    
    Args:
        points: Nx3 array of points
    
    Returns:
        (center, extents, rotation): OBB parameters
    """
    # Center points
    center = np.mean(points, axis=0)
    centered = points - center
    
    # Compute principal axes via PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Rotation matrix (principal axes)
    rotation = eigenvectors
    
    # Project points onto principal axes
    projected = centered @ rotation
    
    # Compute extents along each axis
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    extents = max_proj - min_proj
    
    return center, extents, rotation


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray
) -> np.ndarray:
    """
    Apply rigid transformation to points.
    
    Args:
        points: Nx3 array of points
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    
    Returns:
        Transformed points
    """
    return (rotation @ points.T).T + translation


def compute_camera_intrinsics(
    image_width: int,
    image_height: int,
    fov_degrees: float = 60.0
) -> Tuple[float, float, float, float]:
    """
    Compute camera intrinsics from image size and FOV.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees
    
    Returns:
        (fx, fy, cx, cy): Camera intrinsic parameters
    """
    # Principal point at image center
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Compute focal length from FOV
    fov_radians = np.deg2rad(fov_degrees)
    fx = image_width / (2.0 * np.tan(fov_radians / 2.0))
    
    # Assume square pixels
    fy = fx
    
    return fx, fy, cx, cy


def smooth_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3
) -> np.ndarray:
    """
    Smooth vertex normals using iterative averaging.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        iterations: Number of smoothing iterations
    
    Returns:
        Smoothed normals
    """
    # Compute initial face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    
    # Initialize vertex normals
    vertex_normals = np.zeros_like(vertices)
    
    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    
    # Normalize
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    
    # Iterative smoothing
    for _ in range(iterations):
        smoothed = vertex_normals.copy()
        
        # Average with neighbors
        for face in faces:
            avg_normal = np.mean(vertex_normals[face], axis=0)
            for idx in face:
                smoothed[idx] = 0.7 * smoothed[idx] + 0.3 * avg_normal
        
        # Normalize
        vertex_normals = smoothed / np.linalg.norm(smoothed, axis=1, keepdims=True)
    
    return vertex_normals
