## Abstract Rendering Data Collector

This repository contains a small, self‑contained pipeline to:

- Visualize a GLTF/GLB object and its coordinate axes.
- Render a synthetic multi‑view dataset suitable for Gaussian Splatting.
- Convert the rendered poses and images into a Nerfstudio‑compatible dataset.

All examples below use the provided frog asset: [frog.glb](frog.glb) and the sample data folder [frog_sampled](frog_sampled).

---

## 1. Visualize a GLTF/GLB model

Script: [visualize_gltf.py](visualize_gltf.py)

This script inspects a GLTF/GLB model and overlays the world axes at a chosen origin. It also prints the VTK actor hierarchy to help understand how the model is structured.

**Usage**

```bash
python visualize_gltf.py frog.glb \
	--translate-x [translation_x] \
	--translate-y [translation_y] \
	--translate-z [translation_z]
```

**Arguments**

- `gltf_file` (positional): Path to the `.gltf` or `.glb` file (for example, `frog.glb`).
- `--translate-x`: X translation of the axis origin in world coordinates.
- `--translate-y`: Y translation of the axis origin.
- `--translate-z`: Z translation of the axis origin.

The script:

	- Loads the GLTF/GLB with PyVista.
	- Prints the actor tree (useful for debugging model structure).
	- Draws RGB axes at `(translate_x, translate_y, translate_z)`.
	- Opens an interactive window for orbiting, zooming, and inspecting the model.

This visualization stage is typically used to determine how the model is oriented and where the logical center of the object should be placed. By adjusting `--translate-x`, `--translate-y`, and `--translate-z`, the axes overlay can be positioned relative to the geometry in a way that is convenient for subsequent data collection.

---

## 2. Collect Gaussian Splat data

Script: [collect_splat_data.py](collect_splat_data.py)

This script renders a set of views around a GLTF/GLB object and writes:

- A directory of rendered images.
- A `pose.csv` file describing camera positions and targets in a normalized world coordinate frame.

These are later converted into Nerfstudio’s `transforms.json` format.

**High‑level behavior**

- The camera orbits the object on a set of pitch/yaw angles.
- You can choose a circular or elliptical trajectory.
- For each camera position, the camera can look at one or several sub‑targets along a chosen world axis (useful for long objects like trucks or airplanes).
- Camera and target positions are normalized by a scale factor so that all scenes live at a consistent scale.

**Arguments**

```bash
python collect_splat_data.py frog.glb \
	--output-dir [output_name] \
	--trajectory [circular_or_elliptical] \
	--radius [radius] \
	--target-x [target_x] --target-y [target_y] --target-z [target_z] \
	--num-subtargets [num_subtargets] \
	--z-length [z_length] \
	--subtarget-axis [x|y|z]
```

Key options:

- `gltf_file` (positional): Path to the `.gltf`/`.glb` model.
- `--output-dir`: Base name for the output; the script creates `<output-dir>_sampled`.
- Trajectory shape:
	- `--trajectory {circular,elliptical}`: Orbit type.
	- Circular:
		- `--radius`: Single radius (same for X/Y/Z).
	- Elliptical:
		- `--radius-x`, `--radius-y`, `--radius-z`: Per‑axis radii of the orbit.
- Look‑at target:
	- `--target-x`, `--target-y`, `--target-z`: Main look‑at point (typically the object’s visual center). All orbits are centered around this point.
- Long‑object coverage:
	- `--num-subtargets`: Number of sub‑targets along the chosen axis per camera position.
	- `--z-length`: Approximate object length along that axis; sub‑targets are spaced around the main target over a range of length `z_length`.
	- `--subtarget-axis {x,y,z}`: Axis along which sub‑targets are placed (`z` by default). For example, for a long truck aligned with world Z use the default, while for an airplane aligned with world X use `--subtarget-axis x`.

**Output structure**

Running the command with an appropriate `--output-dir` produces a folder of the form:

- `[output-dir]_sampled/` (for the example, `frog_sampled/`)
	- `images/`
		- `img_0000.png`, `img_0001.png`, ...
	- `pose.csv`

`pose.csv` contains one row per rendered image with columns:

- `x, y, z`: Camera position in a normalized world frame.
- `qw, qx, qy, qz`: Camera orientation (quaternion) derived from the PyVista camera.
- `tx, ty, tz`: Target point (the look‑at position) in the same normalized frame.

The normalization uses a scale factor of $4 / R_\text{ref}$, where $R_\text{ref}$ is the chosen radius (or average of `radius-x/y/z` for elliptical orbits). This ensures that different objects and trajectories are mapped to a comparable global scale.

---

## 3. Convert to Nerfstudio format

Script: [generate_json.py](generate_json.py)

This script converts a `*_sampled` folder (images + `pose.csv`) into a Nerfstudio‑compatible dataset by creating:

- A `transforms.json` describing all camera poses and intrinsics.
- Downsampled image copies for faster visualization.

**Usage**

From the repository root (where `frog_sampled/` is located):

```bash
python generate_json.py frog_sampled
```

If the input directory name ends with `_sampled`, the script automatically creates a subdirectory ending in `_nerfstudio` inside the input directory. For the frog example, this yields:

- `frog_sampled/frog_nerfstudio/`
	- `transforms.json`
	- `images_2/` (half resolution)
	- `images_4/` (quarter resolution)
	- `images_8/` (eighth resolution)

**What generate_json.py does**

- Reads `pose.csv` and reconstructs a camera‑to‑world transform for each image:
	- Camera position = `x, y, z`.
	- Forward direction from camera to target = `F − C` using `tx,ty,tz`.
	- Builds a right/up frame so that the camera looks down its −Z axis (Nerfstudio convention).
- Writes camera intrinsics consistent with how the images were rendered:
	- `w = 640`, `h = 400`.
	- `fl_x = fl_y = 705`, `cx = 320`, `cy = 200`.
- Adds a `frames` entry for each image in `images/` with its corresponding `transform_matrix`.

The resulting `transforms.json` is intended to be used directly with Splatfacto without manual modification.

---

## 4. Train a Gaussian Splatting model with Nerfstudio

Once `frog_nerfstudio/` has been generated, you can train a Gaussian Splatting model (Splatfacto) using Nerfstudio.

From an environment where Nerfstudio is installed (for example, an appropriate conda environment):

```bash
ns-train splatfacto \
	--data /full/path/to/frog_sampled/frog_nerfstudio \
	--max-num-iterations [num_iterations]
```

The `--data` path should point to the local `frog_nerfstudio` directory, and `--max-num-iterations` can be chosen according to the desired training budget and quality. After training, the Nerfstudio viewer or export tools can be used to inspect and utilize the learned Gaussian Splat model.

As a concrete example, assuming this repository is located at `/home/<user>/Abstract-Rendering-Data-Collector`, one possible training command used in development was:

```bash
ns-train splatfacto \
	--data Abstract-Rendering-Data-Collector/frog_sampled/frog_nerfstudio \
	--max-num-iterations 30000
```

---

## Summary

1. Inspect and understand the GLTF/GLB model:
	- `python visualize_gltf.py frog.glb --translate-x [translation_x] --translate-y [translation_y] --translate-z [translation_z]`
2. Collect synthetic training views:
	- `python collect_splat_data.py frog.glb --output-dir [output_name] ...`
3. Convert to Nerfstudio format:
	- `python generate_json.py frog_sampled`
4. Train Splatfacto in Nerfstudio:
	- `ns-train splatfacto --data /full/path/to/frog_sampled/frog_nerfstudio --max-num-iterations [num_iterations]`

These steps describe one complete path from a GLTF/GLB asset to a trained Gaussian Splatting model using the frog example contained in this repository.
