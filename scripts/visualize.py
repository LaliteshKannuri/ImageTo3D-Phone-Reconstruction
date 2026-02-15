#!/usr/bin/env python3
"""
Visualization Utilities
Visualize reconstruction results and intermediate outputs.
"""

import sys
import click
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.group()
def cli():
    """Visualization tools for reconstruction results."""
    pass


@cli.command()
@click.option('--mesh', '-m', required=True, type=click.Path(exists=True),
              help='Mesh file to visualize (.obj, .ply, .stl)')
@click.option('--point-cloud', '-p', type=click.Path(exists=True),
              help='Also show point cloud if available')
@click.option('--screenshot', '-s', type=click.Path(),
              help='Save screenshot to file')
def view_mesh(mesh, point_cloud, screenshot):
    """
    View 3D mesh interactively.
    
    Example:
        python scripts/visualize.py view-mesh -m data/output/phone.obj
    """
    click.echo(f"Loading mesh: {mesh}")
    
    # Load mesh
    mesh_obj = o3d.io.read_triangle_mesh(mesh)
    mesh_obj.compute_vertex_normals()
    
    geometries = [mesh_obj]
    
    # Load point cloud if provided
    if point_cloud:
        click.echo(f"Loading point cloud: {point_cloud}")
        pcd = o3d.io.read_point_cloud(point_cloud)
        geometries.append(pcd)
    
    # Visualize
    click.echo("Opening visualization window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = False
    opt.point_size = 2.0
    
    # Run
    vis.run()
    
    # Screenshot if requested
    if screenshot:
        vis.capture_screen_image(screenshot)
        click.echo(f"Screenshot saved: {screenshot}")
    
    vis.destroy_window()


@cli.command()
@click.option('--results-dir', '-d', required=True, type=click.Path(exists=True),
              help='Results directory containing intermediate files')
@click.option('--output', '-o', type=click.Path(),
              help='Save composite figure')
def view_pipeline(results_dir, output):
    """
    View all pipeline stages in one figure.
    
    Example:
        python scripts/visualize.py view-pipeline -d data/output/
    """
    results_dir = Path(results_dir)
    
    # Find intermediate files
    depth_file = list(results_dir.glob('*_depth.png'))
    detection_file = list(results_dir.glob('*_detections.png'))
    
    if len(depth_file) == 0 or len(detection_file) == 0:
        click.echo("Intermediate files not found. Run with --save-intermediate", err=True)
        return 1
    
    # Load images
    import cv2
    depth_img = cv2.imread(str(depth_file[0]))
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
    
    det_img = cv2.imread(str(detection_file[0]))
    det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(det_img)
    axes[0].set_title('Screw Detections (YOLO)', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(depth_img)
    axes[1].set_title('Depth Map (Depth Anything V2)', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        click.echo(f"Figure saved: {output}")
    else:
        plt.show()


@cli.command()
@click.option('--mesh1', '-m1', required=True, type=click.Path(exists=True),
              help='First mesh')
@click.option('--mesh2', '-m2', required=True, type=click.Path(exists=True),
              help='Second mesh')
@click.option('--labels', '-l', default='Mesh 1,Mesh 2',
              help='Labels (comma-separated)')
def compare_meshes(mesh1, mesh2, labels):
    """
    Compare two meshes side by side.
    
    Example:
        python scripts/visualize.py compare-meshes -m1 mesh1.obj -m2 mesh2.obj
    """
    labels = labels.split(',')
    
    # Load meshes
    mesh1_obj = o3d.io.read_triangle_mesh(mesh1)
    mesh2_obj = o3d.io.read_triangle_mesh(mesh2)
    
    mesh1_obj.compute_vertex_normals()
    mesh2_obj.compute_vertex_normals()
    
    # Print stats
    click.echo("="*60)
    click.echo("MESH COMPARISON")
    click.echo("="*60)
    click.echo(f"{labels[0]}:")
    click.echo(f"  Vertices: {len(mesh1_obj.vertices)}")
    click.echo(f"  Triangles: {len(mesh1_obj.triangles)}")
    click.echo("")
    click.echo(f"{labels[1]}:")
    click.echo(f"  Vertices: {len(mesh2_obj.vertices)}")
    click.echo(f"  Triangles: {len(mesh2_obj.triangles)}")
    click.echo("="*60)
    
    # Visualize side by side
    # Offset second mesh
    bbox = mesh1_obj.get_axis_aligned_bounding_box()
    width = bbox.max_bound[0] - bbox.min_bound[0]
    mesh2_obj.translate([width * 1.5, 0, 0])
    
    o3d.visualization.draw_geometries(
        [mesh1_obj, mesh2_obj],
        window_name="Mesh Comparison"
    )


@cli.command()
@click.option('--report', '-r', required=True, type=click.Path(exists=True),
              help='Batch processing report JSON')
@click.option('--output', '-o', type=click.Path(),
              help='Save plot to file')
def plot_batch_stats(report, output):
    """
    Plot statistics from batch processing report.
    
    Example:
        python scripts/visualize.py plot-batch-stats -r data/output_batch/processing_report.json
    """
    import json
    
    # Load report
    with open(report, 'r') as f:
        results = json.load(f)
    
    # Filter successful results
    successful = [r for r in results if r.get('status') == 'success']
    
    if len(successful) == 0:
        click.echo("No successful reconstructions in report", err=True)
        return 1
    
    # Extract statistics
    num_screws = [r['num_screws'] for r in successful]
    processing_times = [r['elapsed_time'] for r in successful]
    num_vertices = [r['final_vertices'] for r in successful]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Screws detected
    axes[0, 0].hist(num_screws, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Screws')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Screw Detection Distribution')
    axes[0, 0].axvline(np.mean(num_screws), color='red', linestyle='--',
                       label=f'Mean: {np.mean(num_screws):.1f}')
    axes[0, 0].legend()
    
    # Processing time
    axes[0, 1].hist(processing_times, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Processing Time (s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Processing Time Distribution')
    axes[0, 1].axvline(np.mean(processing_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(processing_times):.2f}s')
    axes[0, 1].legend()
    
    # Mesh complexity
    axes[1, 0].scatter(num_screws, num_vertices, alpha=0.6)
    axes[1, 0].set_xlabel('Number of Screws')
    axes[1, 0].set_ylabel('Mesh Vertices')
    axes[1, 0].set_title('Mesh Complexity vs Screws')
    
    # Success rate
    success_rate = len(successful) / len(results) * 100
    axes[1, 1].bar(['Successful', 'Failed'],
                   [len(successful), len(results) - len(successful)],
                   color=['green', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Success Rate: {success_rate:.1f}%')
    axes[1, 1].text(0, len(successful), str(len(successful)),
                    ha='center', va='bottom', fontweight='bold')
    axes[1, 1].text(1, len(results) - len(successful),
                    str(len(results) - len(successful)),
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        click.echo(f"Plot saved: {output}")
    else:
        plt.show()


if __name__ == '__main__':
    cli()
