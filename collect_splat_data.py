import pyvista as pv
import os 
import csv 
from scipy.spatial.transform import Rotation 
import numpy as np
import argparse 

script_dir = os.path.dirname(os.path.realpath(__file__))

def capture_transformed_view(
    plotter,
    camera_position,
    target=(0, 0, 0),
    output_file="screenshot.png",
):
    """Render a single view from the given camera position looking at target."""
    plotter.camera.position = camera_position
    plotter.camera.focal_point = target
    plotter.camera.up = (0, 1, 0)

    plotter.show(interactive=False, auto_close=False)
    plotter.render()
    plotter.screenshot(filename=output_file)

def camera_extrinsics_from_vtk(camera, z_is_forward=True):
    """
    Given a vtk.vtkCamera (e.g. from PyVista's plotter.camera),
    return a rotation matrix R (3x3) and translation vector t (3,).
    This defines the transform from world coords to camera coords.
    
    By default, we define +Z as 'forward' from the camera to the focal point.
    If you want +Z to be 'backward' (like typical OpenGL -Z forward),
    set z_is_forward=False.
    """
    # Get position (camera center), focal point, and view-up from the VTK camera
    C = np.array(camera.GetPosition())    # shape (3,)
    F = np.array(camera.GetFocalPoint())  # shape (3,)
    up = np.array(camera.GetViewUp())     # shape (3,)

    # 1) Forward vector: from camera to the focal point
    f = F - C
    f /= np.linalg.norm(f)  # normalize

    # If you want +Z = forward, keep it as-is
    # If you want +Z = backward (like typical OpenGL), invert it:
    if not z_is_forward:
        f = -f

    # 2) Right vector
    # If you want +Z = backward (like typical OpenGL), invert it:
    if not z_is_forward:
        f = -f

    # 2) Right vector
    r = np.cross(f, up)
    r /= np.linalg.norm(r)

    # 3) True up vector
    u = np.cross(r, f)
    u /= np.linalg.norm(u)

    # Construct rotation matrix R so that:
    # R[0,:] = r, R[1,:] = u, R[2,:] = f
    # That means each row is one of the basis vectors.
    R = np.array([r, u, f])  # shape (3,3)

    # 4) Compute translation so that X_cam = R * (X_world - C).
    # Typically, extrinsic is [R | -R*C], but we can store t = - R*C:
    t = -R @ C

    return R, t

def set_camera_intrinsics(camera, K, width, height):
    """
    Approximate a pinhole camera model in VTK from a 3x3 intrinsics matrix K.
      K = [[f_x,   0,   c_x],
           [  0, f_y,   c_y],
           [  0,   0,     1]]
    width, height: the desired rendering window size (pixels).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 1) Set the vertical field-of-view (view angle) in degrees:
    #    theta_v = 2 * atan( (height / 2) / fy )
    theta_v = 2.0 * np.degrees(np.arctan2(height/2.0, fy))
    camera.SetViewAngle(theta_v)
    
    # 2) Shift the principal point from the image center:
    #    Window center is 0.0 at center, +1.0 at right edge, -1.0 at left edge, etc.
    shift_x = (cx - (width / 2.0)) / (width / 2.0)
    shift_y = ((height / 2.0) - cy)  / (height / 2.0)
    camera.SetWindowCenter(shift_x, shift_y)
    
    # # 3) Set aspect ratio (width/height) if needed:
    # camera.SetAspectRatio(width / float(height))
    
    # If you want perspective projection, ensure:
    camera.SetParallelProjection(False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Collect training data from GLTF/GLB models for Gaussian Splatting')
    parser.add_argument('gltf_file', type=str, help='Path to the GLTF/GLB file')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory name (default: auto-generated from filename)')
    parser.add_argument('--trajectory', type=str, choices=['circular', 'elliptical'], default='circular', 
                        help='Camera trajectory type (default: circular)')
    parser.add_argument('--radius', type=float, default=30, help='Radius for circular trajectory (default: 30)')
    parser.add_argument('--radius-x', type=float, default=30, help='X-axis radius for elliptical trajectory (default: 30)')
    parser.add_argument('--radius-y', type=float, default=30, help='Y-axis radius for elliptical trajectory (default: 30)')
    parser.add_argument('--radius-z', type=float, default=30, help='Z-axis radius for elliptical trajectory (default: 30)')
    parser.add_argument('--target-x', type=float, default=0.0, help='X coordinate of camera look-at target (default: 0.0)')
    parser.add_argument('--target-y', type=float, default=0.0, help='Y coordinate of camera look-at target (default: 0.0)')
    parser.add_argument('--target-z', type=float, default=0.0, help='Z coordinate of camera look-at target (default: 0.0)')
    parser.add_argument('--num-subtargets', type=int, default=1,
                        help='Number of sub-targets along +Z/-Z to look at from each camera position (default: 1 = only main target)')
    parser.add_argument('--z-length', type=float, default=0.0,
                        help='Approximate object length along +Z; used to place sub-targets from -z_length/2 to +z_length/2 (default: 0.0)')
    args = parser.parse_args()
    
    gltf_file = args.gltf_file
    
    # Generate unique output folder name from GLB filename if not provided
    if args.output_dir:
        base_name = args.output_dir
    else:
        # Extract filename without path and extension: "fuel_truck.glb" -> "fuel_truck"
        base_name = os.path.splitext(os.path.basename(gltf_file))[0]
    
    # Initialize data files and folders with unique names
    output_dir_name = os.path.join(script_dir, f'{base_name}_sampled')
    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)
    image_folder = os.path.join(output_dir_name, 'images')
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    transform_fn = os.path.join(output_dir_name, 'pose.csv')
    
    print(f"Output directory: {output_dir_name}")
    print(f"Processing: {gltf_file}")
    print(f"Trajectory: {args.trajectory}")
    if args.trajectory == 'circular':
        print(f"Radius: {args.radius}")
    else:
        print(f"Radii: X={args.radius_x}, Y={args.radius_y}, Z={args.radius_z}")
    
    with open(transform_fn, mode='w') as csv_file:
        fieldnames = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # Rendering resolution
    width = 640
    height = 400

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    # Keep a reference to the imported GLTF to prevent it from
    # being garbage-collected.
    tmp = plotter.import_gltf(gltf_file)      # don’t drop this variable

    headlight = plotter.renderer.lights[0]
    headlight.intensity = 4.0  # Default is 1.0
    plotter.renderer.ambient_light = (4.0, 4.0, 4.0)  # RGB values (white light at 50% intensity)

    plotter.set_background('black')
    
    # Camera intrinsics (must match generate_json.py / transforms.json)
    fx, fy = 705, 705
    cx, cy = width/2, height/2
    K = np.array([
        [fx,   0,  cx],
        [0,   fy,  cy],
        [0,    0,   1]
    ], dtype=np.float32)

    camera = plotter.camera

    # Ensure VTK camera intrinsics match the K we will later
    # encode in transforms.json (fx, fy, cx, cy, width, height).
    set_camera_intrinsics(camera, K, width, height)

    steps1 = 12
    steps2 = 5
    pitches = np.linspace(-np.pi/2, np.pi/2, steps1)
    # yaws = np.linspace(-np.pi, np.pi, 36)

    i = 0

    for j, pitch in enumerate(pitches):
        if j<=steps1//2:
            yaws = np.linspace(-np.pi, np.pi, steps2*j+1)
        else:
            yaws = np.linspace(-np.pi, np.pi, steps2*(steps1-1-j)+1)
        for yaw in yaws:
            # Central target (typically object center)
            center_target = np.array([args.target_x, args.target_y, args.target_z])

            # Calculate camera position based on trajectory type (orbit around center_target)
            if args.trajectory == 'circular':
                r = args.radius
                eye = center_target + np.array([
                    r * np.cos(pitch) * np.cos(yaw), 
                    r * np.cos(pitch) * np.sin(yaw), 
                    r * np.sin(pitch)
                ])
                reference_radius = r
            else:  # elliptical
                eye = center_target + np.array([
                    args.radius_x * np.cos(pitch) * np.cos(yaw), 
                    args.radius_y * np.cos(pitch) * np.sin(yaw), 
                    args.radius_z * np.sin(pitch)
                ])
                # Use average radius as reference for normalization
                reference_radius = (args.radius_x + args.radius_y + args.radius_z) / 3.0

            # Determine sub-targets along the +Z/-Z direction through center_target
            num_sub = max(1, args.num_subtargets)
            z_len = max(0.0, args.z_length)
            if num_sub == 1 or z_len == 0.0:
                sub_targets = [center_target]
            else:
                z_start = args.target_z - z_len / 2.0
                z_end = args.target_z + z_len / 2.0
                z_samples = np.linspace(z_start, z_end, num_sub)
                sub_targets = [
                    np.array([args.target_x, args.target_y, z_val], dtype=float)
                    for z_val in z_samples
                ]
            
            # For each camera position, look at all sub-targets along +Z/-Z
            for sub_target in sub_targets:
                img_fn = os.path.join(image_folder, f'img_{i:04d}.png')

                capture_transformed_view(
                    plotter,
                    eye,
                    sub_target,
                    img_fn,
                )

                print(f"Saved screenshot {i+1} as {img_fn}")

                R, t = camera_extrinsics_from_vtk(camera)

                quats = Rotation.from_matrix(R).as_quat()
                qx, qy, qz, qw = quats

                # Consistent scaling of both camera position and target
                scale_factor = 4.0 / reference_radius
                eye_scaled = eye * scale_factor
                target_scaled = sub_target * scale_factor

                with open(transform_fn, mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({
                        'x': eye_scaled[0], 'y': eye_scaled[1], 'z': eye_scaled[2],
                        'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                        'tx': target_scaled[0], 'ty': target_scaled[1], 'tz': target_scaled[2]
                    })

                i += 1
