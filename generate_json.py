import json
import os
import sys
import argparse

import numpy as np
from PIL import Image

def scale_down_image(img, output_path, factor=2):
    """Scale down image by given factor."""
    width, height = img.size
    new_width, new_height = width // factor, height // factor
    img_resized = img.resize((new_width, new_height))
    img_resized.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pose CSV to NeRFstudio JSON format')
    parser.add_argument('input_dir', type=str, help='Input directory containing pose.csv and images/')
    parser.add_argument('--output-subdir', type=str, default=None, help='Name of output subdirectory (default: auto-generated from input folder name)')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    if args.output_subdir:
        output_subdir_name = args.output_subdir
    else:
        # Extract base name: "fuel_truck_sampled" -> "fuel_truck_nerfstudio"
        base_name = os.path.basename(input_dir.rstrip('/'))
        if base_name.endswith('_sampled'):
            output_subdir_name = base_name.replace('_sampled', '_nerfstudio')
        else:
            output_subdir_name = base_name + '_nerfstudio'
    
    output_dir = os.path.join(input_dir, output_subdir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("="*60)
    print("POSE CSV → NERFSTUDIO JSON CONVERTER")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output subdirectory: {output_dir}")
    print(f"Coordinate system: Y-up, Right-hand (PyVista = Nerfstudio)")
    print("="*60)

    output_json_fn = os.path.join(output_dir, 'transforms.json')

    output_img2_dir = os.path.join(output_dir, './images_2')
    if not os.path.exists(output_img2_dir):
        os.mkdir(output_img2_dir)
    output_img4_dir = os.path.join(output_dir, './images_4')
    if not os.path.exists(output_img4_dir):
        os.mkdir(output_img4_dir)
    output_img8_dir = os.path.join(output_dir, './images_8')
    if not os.path.exists(output_img8_dir):
        os.mkdir(output_img8_dir)

    input_img_dir = os.path.join(input_dir, 'images')
    input_csv = os.path.join(input_dir, 'pose.csv')
    
    if not os.path.exists(input_csv):
        print(f"Error: pose.csv not found in {input_dir}")
        sys.exit(1)
    if not os.path.exists(input_img_dir):
        print(f"Error: images/ folder not found in {input_dir}")
        sys.exit(1)

    res_dict = {
        "w": 640,
        "h": 400,
        "fl_x": 705,
        "fl_y": 705,
        "cx": 320,
        "cy": 200,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "applied_transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ],
        "ply_file_path": "sparse_pc.ply",
        "camera_model": "OPENCV",
        "frames": []
    }

    data = np.genfromtxt(input_csv, dtype=float, delimiter=',', names=True)

    frames = []
    
    print(f"\nProcessing {data.shape[0]} frames...")
    
    for i in range(data.shape[0]):
        if (i+1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{data.shape[0]}")
        
        x, y, z, qw, qx, qy, qz, tx, ty, tz = data[i]
        input_img_fn = os.path.join(input_img_dir, f'img_{i:04d}.png')
        
        # Camera position and target in PyVista world coordinates
        # PyVista: +X=right, +Y=up, +Z=toward you (right-hand)
        C = np.array([x, y, z])      # Camera position
        F = np.array([tx, ty, tz])   # Target (what camera looks at)
        
        # Build camera coordinate frame
        # Forward: direction from camera to target
        forward = F - C
        forward = forward / np.linalg.norm(forward)
        
        # World up vector (Y is up in PyVista)
        up_world = np.array([0, 1, 0])
        
        # Right: perpendicular to forward and world up
        right = np.cross(forward, up_world)
        
        # Handle edge case: camera looking straight up or down
        if np.linalg.norm(right) < 1e-6:
            # Forward is parallel to up, use alternative
            right = np.cross(forward, np.array([1, 0, 0]))
            if np.linalg.norm(right) < 1e-6:
                right = np.cross(forward, np.array([0, 0, 1]))
        
        right = right / np.linalg.norm(right)
        
        # Up: perpendicular to both right and forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Build camera-to-world rotation matrix
        # Camera coordinate system: +X=right, +Y=up, +Z=backward (looking down -Z)
        # Nerfstudio expects camera-to-world transform
        # Columns of R are the world coordinates of camera's basis vectors
        R = np.column_stack([right, up, -forward])
        
        # Verify it's a proper rotation matrix
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            print(f"Warning frame {i}: det(R) = {det:.4f}, fixing...")
            # Orthonormalize using SVD
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

        # Build 4x4 transformation matrix (camera-to-world)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = C
        
        frame = {}
        # Point to original images folder (parent directory)
        frame['file_path'] = f'../images/img_{i:04d}.png'
        frame['colmap_im_id'] = i + 1
        frame['original_fn'] = os.path.normpath(input_img_fn)
        frame['transform_matrix'] = transform_matrix.tolist()
        frames.append(frame)

        # Only create downsampled versions in output subdirectory
        if not os.path.exists(input_img_fn):
            print(f"Warning: Image not found: {input_img_fn}")
            continue
            
        img = Image.open(input_img_fn)
        output_img2_fn = os.path.join(output_img2_dir, f'img_{i:04d}.png')
        scale_down_image(img, output_img2_fn, 2)
        output_img4_fn = os.path.join(output_img4_dir, f'img_{i:04d}.png')
        scale_down_image(img, output_img4_fn, 4)
        output_img8_fn = os.path.join(output_img8_dir, f'img_{i:04d}.png')
        scale_down_image(img, output_img8_fn, 8)

    res_dict['frames'] = frames 
    with open(output_json_fn, 'w+') as f:
        json.dump(res_dict, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated {len(frames)} frames")
    print(f"JSON saved to: {output_json_fn}")
    print(f"Downsampled images saved to: {output_subdir_name}/ subfolder")