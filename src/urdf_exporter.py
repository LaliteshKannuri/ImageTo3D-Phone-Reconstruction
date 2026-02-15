"""
URDF Exporter Module
Exports 3D meshes to URDF format for PyBullet simulation.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom
from loguru import logger


class URDFExporter:
    """
    Export meshes to URDF format for robot simulation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize URDF exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('export', {}).get('urdf', {})
        logger.info("Initialized URDFExporter")
    
    def export(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_dir: Path,
        robot_name: str = "phone"
    ) -> Dict[str, Path]:
        """
        Export mesh to URDF.
        
        Args:
            mesh: 3D mesh to export
            output_dir: Output directory
            robot_name: Name for robot
        
        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export visual mesh (OBJ)
        visual_mesh_path = output_dir / f"{robot_name}_visual.obj"
        o3d.io.write_triangle_mesh(str(visual_mesh_path), mesh)
        
        # Export collision mesh (simplified or convex hull)
        collision_mesh_path = self._export_collision_mesh(mesh, output_dir, robot_name)
        
        # Generate URDF
        urdf_path = output_dir / f"{robot_name}.urdf"
        self._generate_urdf(
            visual_mesh_path,
            collision_mesh_path,
            urdf_path,
            robot_name
        )
        
        logger.success(f"URDF exported: {urdf_path}")
        
        return {
            'urdf': urdf_path,
            'visual_mesh': visual_mesh_path,
            'collision_mesh': collision_mesh_path
        }
    
    def _export_collision_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_dir: Path,
        robot_name: str
    ) -> Path:
        """Export collision mesh (simplified or convex hull)."""
        use_simplified = self.config.get('collision', {}).get('use_simplified_mesh', False)
        use_convex_hull = self.config.get('collision', {}).get('convex_hull', False)
        
        collision_mesh = mesh
        
        if use_convex_hull:
            # Create convex hull for collision
            collision_mesh, _ = mesh.compute_convex_hull()
            logger.info("Created convex hull for collision mesh")
        elif use_simplified:
            # Simplify mesh for collision (faster physics)
            target_triangles = max(100, len(mesh.triangles) // 10)
            collision_mesh = mesh.simplify_quadric_decimation(target_triangles)
            logger.info(f"Simplified collision mesh to {len(collision_mesh.triangles)} triangles")
        
        # Export
        collision_path = output_dir / f"{robot_name}_collision.obj"
        o3d.io.write_triangle_mesh(str(collision_path), collision_mesh)
        
        return collision_path
    
    def _generate_urdf(
        self,
        visual_mesh_path: Path,
        collision_mesh_path: Path,
        output_path: Path,
        robot_name: str
    ):
        """Generate URDF XML file."""
        # Get physics parameters
        physics = self.config.get('physics', {})
        mass = physics.get('mass', 0.17)
        inertia_scale = physics.get('inertia_scale', 1.0)
        friction = physics.get('friction', 0.5)
        restitution = physics.get('restitution', 0.3)
        
        # Estimate inertia from mesh (simple approximation)
        inertia = self._estimate_inertia(mass, inertia_scale)
        
        # Create XML structure
        robot = ET.Element('robot', name=robot_name)
        
        # Base link
        base_link_name = self.config.get('base_link_name', 'phone_base')
        link = ET.SubElement(robot, 'link', name=base_link_name)
        
        # Inertial properties
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia',
                     ixx=str(inertia[0]), ixy='0', ixz='0',
                     iyy=str(inertia[1]), iyz='0',
                     izz=str(inertia[2]))
        
        # Visual
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
        geometry_visual = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry_visual, 'mesh',
                     filename=str(visual_mesh_path.name),
                     scale='1 1 1')
        
        # Material (optional)
        material = ET.SubElement(visual, 'material', name='phone_material')
        ET.SubElement(material, 'color', rgba='0.5 0.5 0.5 1.0')
        
        # Collision
        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
        geometry_collision = ET.SubElement(collision, 'geometry')
        ET.SubElement(geometry_collision, 'mesh',
                     filename=str(collision_mesh_path.name),
                     scale='1 1 1')
        
        # Contact properties
        contact = ET.SubElement(collision, 'contact')
        ET.SubElement(contact, 'lateral_friction', value=str(friction))
        ET.SubElement(contact, 'restitution', value=str(restitution))
        
        # Prettify XML
        xml_string = ET.tostring(robot, encoding='unicode')
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(pretty_xml)
        
        logger.info(f"Generated URDF: {output_path}")
    
    def _estimate_inertia(
        self,
        mass: float,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Estimate inertia tensor (simplified).
        
        For a phone-like rectangular object:
        Ixx = (1/12) * m * (h^2 + d^2)
        Iyy = (1/12) * m * (w^2 + d^2)
        Izz = (1/12) * m * (w^2 + h^2)
        
        Args:
            mass: Object mass in kg
            scale: Scaling factor
        
        Returns:
            Inertia tensor diagonal [Ixx, Iyy, Izz]
        """
        # Assume typical phone dimensions (meters)
        width = 0.07  # 70mm
        height = 0.15  # 150mm
        depth = 0.008  # 8mm
        
        ixx = (1/12) * mass * (height**2 + depth**2) * scale
        iyy = (1/12) * mass * (width**2 + depth**2) * scale
        izz = (1/12) * mass * (width**2 + height**2) * scale
        
        return np.array([ixx, iyy, izz])
    
    def export_stl(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_path: Path,
        ascii: bool = False
    ):
        """
        Export mesh to STL format.
        
        Args:
            mesh: Mesh to export
            output_path: Output file path
            ascii: Whether to use ASCII format (vs binary)
        """
        output_path = Path(output_path)
        o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=ascii
        )
        logger.info(f"STL exported: {output_path}")


def test_urdf_exporter():
    """Test function for URDFExporter."""
    # Test configuration
    config = {
        'export': {
            'urdf': {
                'robot_name': 'test_phone',
                'base_link_name': 'phone_base',
                'physics': {
                    'mass': 0.17,
                    'friction': 0.5,
                    'restitution': 0.3
                },
                'collision': {
                    'use_simplified_mesh': True,
                    'convex_hull': False
                }
            }
        }
    }
    
    exporter = URDFExporter(config)
    
    # Create test mesh (simple cube)
    mesh = o3d.geometry.TriangleMesh.create_box(0.15, 0.07, 0.008)
    mesh.compute_vertex_normals()
    
    # Export
    from pathlib import Path
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    result = exporter.export(mesh, output_dir, robot_name='test_phone')
    
    print(f"URDF exported to: {result['urdf']}")
    print(f"Visual mesh: {result['visual_mesh']}")
    print(f"Collision mesh: {result['collision_mesh']}")


if __name__ == '__main__':
    test_urdf_exporter()
