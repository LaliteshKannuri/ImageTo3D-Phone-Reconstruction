#!/usr/bin/env python3
"""
Batch Processing Script
Process multiple images in batch.
"""

import sys
import click
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MechanicalReconstructor


@click.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Input directory containing images')
@click.option('--output-dir', '-o', default='data/output_batch', type=click.Path(),
              help='Output directory')
@click.option('--config', '-c', default='config/config.yaml', type=click.Path(exists=True),
              help='Config file path')
@click.option('--depth-model', '-d', default='models/depth_anything_v2_vitl.pth',
              type=click.Path(exists=True),
              help='Depth model checkpoint')
@click.option('--yolo-model', '-y', default='models/yolov8_screws.pt',
              type=click.Path(exists=True),
              help='YOLO model checkpoint')
@click.option('--pattern', '-p', default='*.jpg,*.jpeg,*.png',
              help='File patterns to process (comma-separated)')
@click.option('--continue-on-error/--stop-on-error', default=True,
              help='Continue processing on error')
@click.option('--save-report', is_flag=True,
              help='Save processing report as JSON')
def main(input_dir, output_dir, config, depth_model, yolo_model, pattern, continue_on_error, save_report):
    """
    Process multiple images in batch.
    
    Example:
        python scripts/batch_process.py -i data/input/ -o data/output_batch/
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    patterns = pattern.split(',')
    image_paths = []
    for pat in patterns:
        image_paths.extend(input_dir.glob(pat.strip()))
    
    if len(image_paths) == 0:
        click.echo(f"No images found in {input_dir} matching patterns: {pattern}", err=True)
        return 1
    
    # Print header
    click.echo("="*60)
    click.echo("Batch Processing")
    click.echo("="*60)
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Images found: {len(image_paths)}")
    click.echo(f"Continue on error: {continue_on_error}")
    click.echo("="*60)
    
    try:
        # Initialize pipeline once
        click.echo("\nInitializing pipeline...")
        reconstructor = MechanicalReconstructor(
            depth_model_path=depth_model,
            yolo_model_path=yolo_model,
            config_path=config
        )
        
        # Process each image
        all_results = []
        successful = 0
        failed = 0
        
        with tqdm(total=len(image_paths), desc="Processing") as pbar:
            for image_path in image_paths:
                try:
                    # Create output subdirectory
                    image_output_dir = output_dir / image_path.stem
                    
                    # Reconstruct
                    results = reconstructor.reconstruct(
                        image_path=str(image_path),
                        output_dir=str(image_output_dir),
                        save_intermediate=True
                    )
                    
                    results['status'] = 'success'
                    all_results.append(results)
                    successful += 1
                    
                    pbar.set_postfix({'success': successful, 'failed': failed})
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    
                    all_results.append({
                        'image_path': str(image_path),
                        'status': 'failed',
                        'error': str(e)
                    })
                    failed += 1
                    
                    pbar.set_postfix({'success': successful, 'failed': failed})
                    
                    if not continue_on_error:
                        raise
                
                pbar.update(1)
        
        # Print summary
        click.echo("\n" + "="*60)
        click.echo("BATCH PROCESSING COMPLETE")
        click.echo("="*60)
        click.echo(f"Total images: {len(image_paths)}")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {failed}")
        click.echo(f"Success rate: {successful/len(image_paths)*100:.1f}%")
        click.echo("="*60)
        
        # Save report
        if save_report:
            report_path = output_dir / 'processing_report.json'
            with open(report_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            click.echo(f"\n✓ Report saved: {report_path}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        click.echo(f"\n✗ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
