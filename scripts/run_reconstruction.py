#!/usr/bin/env python3
"""
Single Image Reconstruction Script
Run mechanical reconstruction on a single image.
"""

import sys
import click
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MechanicalReconstructor


@click.command()
@click.option('--image', '-i', required=True, type=click.Path(exists=True),
              help='Input image path')
@click.option('--output', '-o', default='data/output', type=click.Path(),
              help='Output directory (default: data/output)')
@click.option('--config', '-c', default='config/config.yaml', type=click.Path(exists=True),
              help='Config file path (default: config/config.yaml)')
@click.option('--depth-model', '-d', default='models/depth_anything_v2_vitl.pth',
              type=click.Path(exists=True),
              help='Depth model checkpoint')
@click.option('--yolo-model', '-y', default='models/yolov8_screws.pt',
              type=click.Path(exists=True),
              help='YOLO model checkpoint')
@click.option('--save-intermediate/--no-save-intermediate', default=True,
              help='Save intermediate results')
@click.option('--visualize/--no-visualize', default=True,
              help='Generate visualizations')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose logging')
def main(image, output, config, depth_model, yolo_model, save_intermediate, visualize, verbose):
    """
    Run mechanical reconstruction on a single image.
    
    Example:
        python scripts/run_reconstruction.py -i data/input/phone.jpg
    """
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Print header
    click.echo("="*60)
    click.echo("Phone Mechanical Reconstruction System")
    click.echo("="*60)
    click.echo(f"Input image: {image}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Config: {config}")
    click.echo("="*60)
    
    try:
        # Initialize pipeline
        click.echo("\nInitializing pipeline...")
        reconstructor = MechanicalReconstructor(
            depth_model_path=depth_model,
            yolo_model_path=yolo_model,
            config_path=config
        )
        
        # Run reconstruction
        click.echo("\nRunning reconstruction...")
        results = reconstructor.reconstruct(
            image_path=image,
            output_dir=output,
            save_intermediate=save_intermediate
        )
        
        # Print results
        click.echo("\n" + "="*60)
        click.echo("RESULTS")
        click.echo("="*60)
        click.echo(f"✓ Screws detected: {results['num_screws']}")
        click.echo(f"✓ Final mesh vertices: {results['final_vertices']}")
        click.echo(f"✓ Final mesh triangles: {results['final_triangles']}")
        click.echo(f"✓ Processing time: {results['elapsed_time']:.2f}s")
        click.echo("")
        click.echo(f"✓ OBJ mesh: {results['mesh_path']}")
        if 'urdf_path' in results:
            click.echo(f"✓ URDF: {results['urdf_path']}")
        click.echo("="*60)
        
        # Visualize if requested
        if visualize and 'intermediate' in results:
            click.echo("\nGenerated visualizations:")
            for key, path in results['intermediate'].items():
                click.echo(f"  - {key}: {path}")
        
        click.echo("\n✓ Reconstruction complete!")
        return 0
        
    except Exception as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
