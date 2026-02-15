"""
Example: Basic Usage of Mechanical Reconstructor

This script demonstrates how to use the phone reconstruction pipeline programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MechanicalReconstructor


def example_single_reconstruction():
    """Example 1: Reconstruct a single image"""
    
    print("="*60)
    print("Example 1: Single Image Reconstruction")
    print("="*60)
    
    # Initialize reconstructor
    reconstructor = MechanicalReconstructor(
        depth_model_path='models/depth_anything_v2_vitl.pth',
        yolo_model_path='models/yolov8_screws.pt',
        config_path='config/config.yaml'
    )
    
    # Run reconstruction
    results = reconstructor.reconstruct(
        image_path='data/input/phone.jpg',
        output_dir='data/output/phone'
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Screws detected: {results['num_screws']}")
    print(f"  Processing time: {results['elapsed_time']:.2f}s")
    print(f"  Output mesh: {results['mesh_path']}")
    print(f"  Output URDF: {results['urdf_path']}")


def example_batch_reconstruction():
    """Example 2: Batch process multiple images"""
    
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    # Initialize reconstructor
    reconstructor = MechanicalReconstructor(
        depth_model_path='models/depth_anything_v2_vitl.pth',
        yolo_model_path='models/yolov8_screws.pt',
        config_path='config/config.yaml'
    )
    
    # List of images
    image_paths = [
        'data/input/phone1.jpg',
        'data/input/phone2.jpg',
        'data/input/phone3.jpg',
    ]
    
    # Process all images
    all_results = reconstructor.batch_reconstruct(
        image_paths=image_paths,
        output_dir='data/output/batch'
    )
    
    # Print summary
    successful = sum(1 for r in all_results if r['status'] == 'success')
    print(f"\nBatch complete: {successful}/{len(image_paths)} successful")


def example_custom_config():
    """Example 3: Use custom configuration"""
    
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    # Define custom config
    custom_config = {
        'models': {
            'depth_anything': {'encoder': 'vitl'},
            'yolo': {'confidence_threshold': 0.4}  # Lower threshold
        },
        'calibration': {
            'screw_diameter_mm': 2.5,  # Smaller screws
            'depth_range_mm': 80.0
        },
        'depth': {
            'bilateral_filter': {
                'enabled': True,
                'diameter': 9,
                'sigma_color': 0.2,
                'sigma_space': 5
            }
        },
        'mesh': {
            'poisson': {'depth': 8},  # Faster reconstruction
            'cleanup': {
                'remove_degenerate_triangles': True,
                'remove_duplicated_vertices': True
            }
        },
        'export': {
            'urdf': {'enabled': True},
            'stl': {'enabled': False}
        },
        'processing': {
            'device': 'cuda'  # Force GPU
        },
        'output': {
            'save_intermediate': {
                'enabled': True,
                'depth_map': True,
                'detections': True
            }
        }
    }
    
    # Initialize with custom config
    reconstructor = MechanicalReconstructor(
        depth_model_path='models/depth_anything_v2_vitl.pth',
        yolo_model_path='models/yolov8_screws.pt',
        config_dict=custom_config
    )
    
    # Run reconstruction
    results = reconstructor.reconstruct(
        image_path='data/input/phone.jpg',
        output_dir='data/output/custom'
    )
    
    print(f"\nCustom reconstruction complete")
    print(f"  Output: {results['mesh_path']}")


def example_inspect_results():
    """Example 4: Inspect reconstruction results"""
    
    print("\n" + "="*60)
    print("Example 4: Inspect Results")
    print("="*60)
    
    import open3d as o3d
    
    # Load reconstructed mesh
    mesh = o3d.io.read_triangle_mesh('data/output/phone/phone.obj')
    
    # Get mesh statistics
    print(f"\nMesh Statistics:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    print(f"  Is watertight: {mesh.is_watertight()}")
    print(f"  Is vertex manifold: {mesh.is_vertex_manifold()}")
    print(f"  Is edge manifold: {mesh.is_edge_manifold()}")
    
    # Get bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    dimensions = bbox.max_bound - bbox.min_bound
    print(f"\nDimensions (meters):")
    print(f"  Width:  {dimensions[0]:.4f}")
    print(f"  Height: {dimensions[1]:.4f}")
    print(f"  Depth:  {dimensions[2]:.4f}")
    
    # Surface area and volume
    print(f"\nGeometry:")
    print(f"  Surface area: {mesh.get_surface_area():.6f} m²")
    
    # Visualize
    print(f"\nOpening 3D viewer...")
    o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Phone")


def example_pybullet_simulation():
    """Example 5: Load in PyBullet for simulation"""
    
    print("\n" + "="*60)
    print("Example 5: PyBullet Simulation")
    print("="*60)
    
    import pybullet as p
    import time
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load reconstructed phone
    phone_id = p.loadURDF(
        "data/output/phone/phone.urdf",
        [0, 0, 0.5],  # Start position
        [0, 0, 0, 1]  # Orientation (quaternion)
    )
    
    print(f"\nPhone loaded in PyBullet (ID: {phone_id})")
    print("Running simulation for 5 seconds...")
    
    # Run simulation
    for i in range(500):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Get final position
    pos, orn = p.getBasePositionAndOrientation(phone_id)
    print(f"Final position: {pos}")
    
    p.disconnect()


if __name__ == '__main__':
    # Run examples
    print("Phone Mechanical Reconstruction - Usage Examples\n")
    
    # Uncomment the example you want to run:
    
    example_single_reconstruction()
    # example_batch_reconstruction()
    # example_custom_config()
    # example_inspect_results()
    # example_pybullet_simulation()
    
    print("\nDone!")
