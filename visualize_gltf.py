
import argparse

import pyvista as pv
from vtk import vtkAssembly

def print_actor_tree(actor, indent=0):
    """Recursively print actor hierarchy and types."""
    prefix = "  " * indent
    print(f"{prefix}- (Type: {type(actor)})")
    
    if isinstance(actor, vtkAssembly):
        parts = actor.GetParts()
        parts.InitTraversal()
        for _ in range(parts.GetNumberOfItems()):
            part = parts.GetNextProp()
            print_actor_tree(part, indent + 1)

def rotate_actor(actor, angle_x=0, angle_y=0, angle_z=0):
    """Recursively rotate an actor and its sub-components."""
    if isinstance(actor, vtkAssembly):
        # Get all sub-parts of the assembly
        parts = actor.GetParts()
        parts.InitTraversal()
        for _ in range(parts.GetNumberOfItems()):
            part = parts.GetNextProp()
            rotate_actor(part, angle_x, angle_y, angle_z)  # Recursion
    else:
        # Rotate individual non-assembly actors
        actor.RotateX(angle_x)
        actor.RotateY(angle_y)
        actor.RotateZ(angle_z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize GLTF/GLB 3D models with PyVista')
    parser.add_argument('gltf_file', type=str, help='Path to the GLTF/GLB file')
    parser.add_argument('--translate-x', type=float, default=0.0, help='X-axis translation/displacement for axes origin')
    parser.add_argument('--translate-y', type=float, default=0.0, help='Y-axis translation/displacement for axes origin')
    parser.add_argument('--translate-z', type=float, default=0.0, help='Z-axis translation/displacement for axes origin')
    args = parser.parse_args()
    
    gltf_file = args.gltf_file

    pl = pv.Plotter()

    # Keep a reference to the imported GLTF to prevent it
    # from being garbage-collected prematurely.
    tmp = pl.import_gltf(gltf_file)

    for key, actor in pl.actors.items():
        print(f"Root actor key: {key}")
        print_actor_tree(actor)

    # Create colored axes to visualize the coordinate frame at a
    # user-specified origin.
    axis_length = 200
    x_offset = args.translate_x
    y_offset = args.translate_y
    z_offset = args.translate_z
    
    print(f"\nAxes origin positioned at: ({x_offset}, {y_offset}, {z_offset})")
    
    x_axis = pv.Line(pointa=(x_offset, y_offset, z_offset), pointb=(x_offset+axis_length, y_offset, z_offset))
    # Y-axis from (0, -1, 0) to (0, 1, 0)
    y_axis = pv.Line(pointa=(x_offset, y_offset, z_offset), pointb=(x_offset, y_offset+axis_length, z_offset))
    # Z-axis from (0, 0, -1) to (0, 0, 1)
    z_axis = pv.Line(pointa=(x_offset, y_offset, z_offset), pointb=(x_offset, y_offset, z_offset+axis_length))

    # Add the axis lines to the plotter with distinct colors
    pl.add_mesh(x_axis, color='red', line_width=4, label='X Axis')
    pl.add_mesh(y_axis, color='green', line_width=4, label='Y Axis')
    pl.add_mesh(z_axis, color='blue', line_width=4, label='Z Axis')

    headlight = pl.renderer.lights[0]
    headlight.intensity = 8.0  # Default is 1.0
    pl.renderer.ambient_light = (4.0, 4.0, 4.0)  # RGB values (white light at 50% intensity)

    # pl.set_environment_texture(env_map)       # HDR lighting is still useful
    # pl.enable_ray_tracing()                   # optional: nicer PBR shading
    pl.camera.zoom(1.4)
    # pl.set_background('#7b8bc4')
    pl.show()
