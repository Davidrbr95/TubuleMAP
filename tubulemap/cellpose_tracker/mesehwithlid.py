import os
import math
import json
import re
import h5py
import numpy as np
from pathlib import Path
from skimage import measure
from scipy.ndimage import gaussian_filter1d
import trimesh
from trimesh.smoothing import filter_taubin
import traceback

# ============================================================
# Parameter Settings
# ============================================================
JSON_DIR = Path(
    r"/media/cfxuser/SSD2/Nephron_Tracking/GT/hand_drawn_masks/ground_truth_mask/Obj_nephron2/orthoplanes/corrected_points.json/Run_0"
)
HDF5_DIR = Path(
    r"/media/cfxuser/SSD2/Nephron_Tracking/GT/hand_drawn_masks/ground_truth_mask/Obj_nephron2/orthoplanes/corrected_points.json/Run_0"
)
OUTPUT_DIR = Path(
    r"/media/cfxuser/SSD2/Nephron_Tracking/GT/hand_drawn_masks/ground_truth_mask/Obj_nephron2/orthoplanes/corrected_points.json/Run_0/mesh"
)

# Default parameters
TARGET_CONTOUR_POINTS = 128
SMOOTH_ITER = 10
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
SKIP_EMPTY_SLICES = True
CAP_THRESHOLD_PX = 50  # Add caps when start/end segment centers are closer than this threshold (pixels).

ID_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# ============================================================
# Utility Functions
# ============================================================
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def rodrigues_rotate(v, k, theta):
    v = np.asarray(v, dtype=float)
    k = normalize(k)
    ct = math.cos(theta)
    st = math.sin(theta)
    return v * ct + np.cross(k, v) * st + k * np.dot(k, v) * (1 - ct)

# ============================================================
# Data Loading
# ============================================================
def _extract_zyx(point, source_axes=None):
    """
    Extract z,y,x from a point with optional axis metadata.

    Fallback behavior matches tracker conventions:
      - len >= 5 : [t, c, z, y, x]
      - else     : last 3 coordinates are [z, y, x]
    """
    coords = list(point)
    if len(coords) < 3:
        raise ValueError("Point must have at least 3 coordinates.")

    if source_axes is not None:
        axis_names = [str(axis).strip().lower() for axis in source_axes]
        axis_map = {axis_name: idx for idx, axis_name in enumerate(axis_names)}
        if {"z", "y", "x"}.issubset(axis_map):
            if len(coords) == len(axis_names):
                z = coords[axis_map["z"]]
                y = coords[axis_map["y"]]
                x = coords[axis_map["x"]]
                return float(z), float(y), float(x)

            if len(coords) == 3:
                spatial_order = [name for name in axis_names if name in {"z", "y", "x"}]
                if len(spatial_order) == 3:
                    spatial_map = {axis_name: idx for idx, axis_name in enumerate(spatial_order)}
                    z = coords[spatial_map["z"]]
                    y = coords[spatial_map["y"]]
                    x = coords[spatial_map["x"]]
                    return float(z), float(y), float(x)

    if len(coords) >= 5:
        z, y, x = coords[2], coords[3], coords[4]
    else:
        z, y, x = coords[-3], coords[-2], coords[-1]
    return float(z), float(y), float(x)


def load_centerline_points(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        points = data.get("points", [])
        point_axes = data.get("point_axes")
    else:
        points = data
        point_axes = None

    if not points:
        raise ValueError(f"{json_path} has no 'points' or it's empty.")

    if isinstance(point_axes, (list, tuple)):
        source_axes = [str(axis).strip().lower() for axis in point_axes]
    else:
        source_axes = None

    centerline_xyz = []
    for idx, point in enumerate(points):
        try:
            z, y, x = _extract_zyx(point, source_axes=source_axes)
        except Exception as exc:
            raise ValueError(f"Invalid point at index {idx} in {json_path}") from exc
        centerline_xyz.append([x, y, z])

    return np.asarray(centerline_xyz, dtype=float)

def load_masks(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith("mask_")],
                      key=lambda s: int(s.split("_")[-1]))
        masks = [np.array(f[k]) for k in keys]
    return masks

# ============================================================
# Geometry Construction
# ============================================================
def build_parallel_transport_frames(centerline):
    P = np.asarray(centerline, dtype=float)
    Np = len(P)
    Ts = np.zeros((Np, 3))
    Ns = np.zeros((Np, 3))
    Bs = np.zeros((Np, 3))

    for i in range(Np):
        if i == 0:
            t = P[1] - P[0] if Np > 1 else np.array([1.0, 0.0, 0.0])
        elif i == Np - 1:
            t = P[-1] - P[-2]
        else:
            t = P[i + 1] - P[i - 1]
        Ts[i] = normalize(t)

    arbitrary = np.array([0.0, 0.0, 1.0])
    if Np > 1 and np.linalg.norm(np.cross(Ts[0], Ts[1])) > 1e-8:
        arbitrary = normalize(np.cross(Ts[0], Ts[1]))
    Ns[0] = normalize(np.cross(Ts[0], np.cross(arbitrary, Ts[0])))
    Bs[0] = normalize(np.cross(Ts[0], Ns[0]))

    for i in range(1, Np):
        Ti_1, Ti = Ts[i - 1], Ts[i]
        k = np.cross(Ti_1, Ti)
        s = np.linalg.norm(k)
        if s < 1e-8:
            Ns[i] = Ns[i - 1]
        else:
            k /= s
            theta = math.acos(np.clip(np.dot(Ti_1, Ti), -1.0, 1.0))
            Ns[i] = rodrigues_rotate(Ns[i - 1], k, theta)
        Ns[i] = normalize(Ns[i] - np.dot(Ns[i], Ti) * Ti)
        Bs[i] = normalize(np.cross(Ti, Ns[i]))
    return Ts, Ns, Bs

def ring_to_world_from_xy(xy_px, center3d_px, N_axis, B_axis):
    """
    Map a 2D contour (x_px, y_px) into 3D: center + x*B + y*N.
    Note: all coordinates use pixel units (same space as JSON/HDF5).
    """
    return center3d_px[None, :] + xy_px[:, 1:2] * N_axis[None, :] + xy_px[:, 0:1] * B_axis[None, :]

def get_center_mask_contour(mask):
    """
    Extract the longest 2D contour from the connected component that contains
    the geometric center pixel of the mask.
    """
    H, W = mask.shape
    center_label = int(mask[H // 2, W // 2])
    if center_label <= 0:
        return None
    region_mask = (mask == center_label).astype(np.uint8)
    contours = measure.find_contours(region_mask, 0.5)
    if not contours:
        return None
    return max(contours, key=len)

def smooth_and_resample_contour(contour, n_points=128, smooth_sigma=2):
    """
    Smooth contour and uniformly resample to n_points.
    """
    P = np.asarray(contour, dtype=float)
    if len(P) < 5:
        return P
    y_s = gaussian_filter1d(P[:, 0], smooth_sigma, mode="wrap")
    x_s = gaussian_filter1d(P[:, 1], smooth_sigma, mode="wrap")
    P = np.stack([y_s, x_s], axis=1)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        idxs = np.linspace(0, len(P) - 1, n_points).astype(int)
        return P[idxs]
    t = np.linspace(0, total, n_points + 1)[:-1]
    resampled = []
    for ti in t:
        idx = np.searchsorted(s, ti) - 1
        idx = np.clip(idx, 0, len(seg) - 1)
        alpha = (ti - s[idx]) / (seg[idx] + 1e-12)
        p = (1 - alpha) * P[idx] + alpha * P[idx + 1]
        resampled.append(p)
    return np.asarray(resampled)

def align_rings(prev_ring, curr_ring):
    """
    Cyclically shift along the ring so adjacent slice rings align with
    minimum mean point-wise distance.
    """
    if prev_ring is None or curr_ring is None:
        return curr_ring
    if len(prev_ring) != len(curr_ring):
        M = len(prev_ring)
        idxs = np.linspace(0, len(curr_ring) - 1, M)
        curr_ring = curr_ring[np.round(idxs).astype(int)]
    M = len(curr_ring)
    best_shift = 0
    best_score = np.inf
    for shift in range(M):
        rolled = np.roll(curr_ring, shift, axis=0)
        score = np.mean(np.linalg.norm(rolled - prev_ring, axis=1))
        if score < best_score:
            best_score, best_shift = score, shift
    return np.roll(curr_ring, best_shift, axis=0)

# ============================================================
# Mesh Construction (segment-based strategy with end caps)
# ============================================================
def _split_segments_by_none(rings3d):
    """
    Split rings3d (which may include None) into contiguous non-None segments.
    Returns a list where each item is one segment (list of rings).
    """
    segments = []
    current = []
    for r in rings3d:
        if r is None:
            if len(current) > 0:
                segments.append(current)
                current = []
        else:
            current.append(r)
    if len(current) > 0:
        segments.append(current)
    return segments

def _build_strip_faces(start_base, n_rings, M):
    """
    Build side-wall triangles by splitting quads between adjacent rings.
    """
    faces = []
    for i in range(n_rings - 1):
        base0 = start_base + i * M
        base1 = start_base + (i + 1) * M
        for j in range(M):
            jn = (j + 1) % M
            faces.append([base0 + j, base1 + j, base0 + jn])
            faces.append([base0 + jn, base1 + j, base1 + jn])
    return faces
def build_loft_mesh(rings3d, centerline=None):
    """
    Build tube-like side walls from rings3d in order and apply Taubin smoothing
    to side walls only. Then add one cap at the first valid ring and one cap at
    the last valid ring to produce a closed surface. Caps are not smoothed so
    the visual appearance stays close to the original open tube.
    """
    # Handle all-None input.
    if not any(r is not None for r in rings3d):
        raise ValueError("Not enough valid slices to build a mesh")

    all_vertices = []          # Each element is an (N_i, 3) array.
    faces = []                 # Triangle indices (side walls only at this stage).
    ring_start_indices = []    # Per-ring start index in V, aligned with rings3d (or None).
    M_global = None            # Shared point count across all rings.
    prev_start = None          # Start index in V for the previous ring.
    prev_idx = None            # Index in rings3d for the previous ring.

    def add_side_faces(base0, base1, M):
        """Generate side-wall triangles between ring base0 and ring base1."""
        for j in range(M):
            jn = (j + 1) % M
            # Two triangles form one quad.
            faces.append([base0 + j,  base1 + j,  base0 + jn])
            faces.append([base0 + jn, base1 + j,  base1 + jn])

    # === Append ring vertices and connect side walls ===
    for i, ring in enumerate(rings3d):
        if ring is None:
            ring_start_indices.append(None)
            prev_start = None
            prev_idx = None
            continue

        ring = np.asarray(ring, dtype=float)

        # Resample each ring to M_global points.
        if M_global is None:
            M_global = ring.shape[0]
        elif ring.shape[0] != M_global:
            idxs = np.linspace(0, ring.shape[0] - 1, M_global)
            ring = ring[np.round(idxs).astype(int)]

        # Start index for this ring in global vertex array.
        base = sum(v.shape[0] for v in all_vertices)
        all_vertices.append(ring)
        ring_start_indices.append(base)

        # Connect side walls only when previous ring is valid and adjacent.
        if prev_start is not None and prev_idx is not None and i == prev_idx + 1:
            add_side_faces(prev_start, base, M_global)

        prev_start = base
        prev_idx = i

    if not all_vertices:
        raise ValueError("No valid rings to build mesh")

    # Stack all ring vertices into global V (side walls only so far).
    V = np.vstack(all_vertices)
    F = np.asarray(faces, dtype=np.int64)

    # === Smooth side walls first with Taubin ===
    mesh_side = trimesh.Trimesh(vertices=V.copy(), faces=F.copy(), process=False)
    if SMOOTH_ITER > 0 and len(mesh_side.vertices) > 0 and len(mesh_side.faces) > 0:
        filter_taubin(mesh_side, lamb=TAUBIN_LAMBDA, nu=TAUBIN_MU, iterations=SMOOTH_ITER)

    # Smoothed vertices/faces (still side walls only, no caps).
    V = mesh_side.vertices.copy()
    F = mesh_side.faces.copy()

    # === Find first/last valid rings in trace (caps only at these ends) ===
    first_i = next(i for i, s in enumerate(ring_start_indices) if s is not None)
    last_i  = max(i for i, s in enumerate(ring_start_indices) if s is not None)

    first_start = ring_start_indices[first_i]
    last_start  = ring_start_indices[last_i]

    # Debug: print cap ring indices and corresponding centerline endpoints.
    print("Cap ring indices:")
    print(f"   first_i = {first_i}, last_i = {last_i}")
    if centerline is not None:
        try:
            print("Corresponding centerline endpoint coordinates:")
            print(f"   centerline[first_i] = {centerline[first_i]}")
            print(f"   centerline[last_i]  = {centerline[last_i]}")
        except Exception as e:
            print(f"   Could not index centerline endpoints: {e}")

    M0 = M_global
    M1 = M_global

    first_ring_idx = np.arange(first_start, first_start + M0)
    last_ring_idx  = np.arange(last_start,  last_start  + M1)

    # Compute cap centers from smoothed V.
    first_center = V[first_ring_idx].mean(axis=0)
    last_center  = V[last_ring_idx].mean(axis=0)

    # Debug: print cap center coordinates.
    print("Cap center coordinates:")
    print(f"   first_center = {first_center}")
    print(f"   last_center  = {last_center}")

    # === Add two cap-center vertices ===
    c0_idx = len(V)
    c1_idx = len(V) + 1
    V = np.vstack([V, first_center[None, :], last_center[None, :]])

    # First cap: reverse order [center, b, a] to orient normal outward.
    cap_faces = []
    for j in range(M0):
        a = first_start + j
        b = first_start + (j + 1) % M0
        cap_faces.append([c0_idx, b, a])

    # Last cap: forward order [center, a, b].
    for j in range(M1):
        a = last_start + j
        b = last_start + (j + 1) % M1
        cap_faces.append([c1_idx, a, b])

    # Append cap faces to F (no additional smoothing of full mesh).
    if F.size == 0:
        F_final = np.asarray(cap_faces, dtype=np.int64)
    else:
        F_final = np.vstack([F, np.asarray(cap_faces, dtype=np.int64)])

    # Build final mesh (side walls smoothed, caps preserved).
    mesh = trimesh.Trimesh(vertices=V, faces=F_final, process=False)
    return mesh




def clean_mesh_inplace(mesh: trimesh.Trimesh):
    vmask = np.all(np.isfinite(mesh.vertices), axis=1)
    if not vmask.all():
        mesh.update_vertices(mask=vmask)
    if len(mesh.faces) > 0:
        fmask = np.all(mesh.faces < len(mesh.vertices), axis=1)
        mesh.update_faces(mask=fmask)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh became empty after cleaning")

# ============================================================
# Main Function: Reconstruct a Single Mesh
# ============================================================
def reconstruct_mesh_from_pair(json_path: Path, hdf5_path: Path):
    centerline = load_centerline_points(json_path)
    masks = load_masks(hdf5_path)

    n_points = len(centerline)
    n_masks = len(masks)

    # === Check whether number of points and masks match ===
    if n_points != n_masks:
        print(f"⚠️ Warning: {json_path.name} has {n_points} points, "
              f"but {hdf5_path.name} has {n_masks} masks. "
              f"→ Aligning using nearest center mapping.")

        # --- Step 1. Extract center point of each mask ---
        mask_centers = []
        for i, m in enumerate(masks):
            if m.max() > 0:
                ys, xs = np.nonzero(m)
                center_y, center_x = np.mean(ys), np.mean(xs)
                mask_centers.append(np.array([center_x, center_y, 0.0]))  # z represented by index
            else:
                mask_centers.append(None)

        # --- Step 2. Match each mask to the nearest centerline point ---
        matched_centerline = []
        for i, mc in enumerate(mask_centers):
            if mc is None:
                matched_centerline.append(None)
                continue
            z_hint = i / max(1, n_masks - 1)
            approx_idx = int(z_hint * (n_points - 1))
            diffs = centerline - centerline[approx_idx]
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argmin(dists)
            matched_centerline.append(centerline[nearest_idx])

        # --- Step 3. Remove None and create new centerline array ---
        valid_pairs = [(c, m) for c, m in zip(matched_centerline, masks) if c is not None and m.max() > 0]
        if len(valid_pairs) < 2:
            raise ValueError(f"Too few valid matched slices in {json_path.name}")

        centerline, masks = zip(*valid_pairs)
        centerline = np.array(centerline)
        masks = list(masks)
    else:
        n = min(n_points, n_masks)
        centerline = centerline[:n]
        masks = masks[:n]

    # === Debug: print centerline endpoints actually used for this reconstruction ===
    print("Centerline range used:")
    print(f"   start point (index 0): {centerline[0]}")
    print(f"   end   point (index {len(centerline) - 1}): {centerline[-1]}")


    # === Reconstruction process ===
    Ts, Ns, Bs = build_parallel_transport_frames(centerline)
    rings_world, prev_ring = [], None
    valid_count = 0

    for idx, mask in enumerate(masks):
        if SKIP_EMPTY_SLICES and mask.max() <= 0:
            rings_world.append(None)
            continue
        contour = get_center_mask_contour(mask)
        if contour is None or len(contour) < 10:
            rings_world.append(None)
            continue

        contour = smooth_and_resample_contour(contour, TARGET_CONTOUR_POINTS, smooth_sigma=2)
        centroid = contour.mean(axis=0)
        # Convert contour to centroid-centered 2D coordinates (pixels).
        y_px = contour[:, 0] - centroid[0]
        x_px = contour[:, 1] - centroid[1]
        xy_px = np.stack([x_px, y_px], axis=1)

        # Project to 3D (pixels).
        ring3d = ring_to_world_from_xy(xy_px, centerline[idx], Ns[idx], Bs[idx])
        if prev_ring is not None:
            ring3d = align_rings(prev_ring, ring3d)

        rings_world.append(ring3d)
        prev_ring = ring3d
        valid_count += 1

    if valid_count < 2:
        raise ValueError("Less than two valid slices")

    # === Core: segment-based loft + two end caps ===
    mesh = build_loft_mesh(rings_world, centerline=centerline)

    # === Cleanup ===
    clean_mesh_inplace(mesh)

    # === Watertight check ===
    try:
        is_water = mesh.is_watertight
        print("Watertight check result:")
        print(f"   is_watertight = {is_water}")
    except Exception as e:
        print(f"Watertight check failed: {e}")

    return mesh




# ============================================================
# Batch Matching and Export
# ============================================================
def collect_pairs(json_dir: Path, h5_dir: Path):
    json_files = list(json_dir.glob("*.json"))
    h5_files = list(h5_dir.glob("*.hdf5"))
    print(json_files)
    print(h5_files)
    id_pattern = re.compile(r"(\d{1,5})")
    def extract_id(path: Path):
        m = id_pattern.search(path.stem)
        return m.group(1) if m else None
    json_map = {extract_id(p): p for p in json_files if extract_id(p)}
    h5_map = {extract_id(p): p for p in h5_files if extract_id(p)}
    common_ids = sorted(set(json_map.keys()) & set(h5_map.keys()))
    print(f"✅ Found {len(common_ids)} matched pairs.")
    if not common_ids:
        print("⚠️ No numeric ID matches.")
    return [(nid, json_map[nid], h5_map[nid]) for nid in common_ids]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(JSON_DIR, HDF5_DIR)
    print(pairs)
    if not pairs:
        print("❌ No matched pairs found.")
        return

    print(f"✅ Found {len(pairs)} matched nephron pairs.")
    existing_meshes = {
        re.search(ID_PATTERN, p.name).group(1)
        for p in OUTPUT_DIR.glob("*.obj")
        if re.search(ID_PATTERN, p.name)
    }

    skipped, failed, completed = [], [], []

    for nid, jp, hp in pairs:
        out_path = OUTPUT_DIR / f"N{nid}_mesh.obj"
        if nid in existing_meshes or out_path.exists():
            print(f"⏭️  Skipping {nid}: mesh already exists → {out_path.name}")
            skipped.append(nid)
            continue

        try:
            print(f"\n🧩 Processing {nid}: {jp.name} + {hp.name}")
            mesh = reconstruct_mesh_from_pair(jp, hp)
            mesh.export(out_path)
            completed.append(nid)
            print(f"✅ Saved mesh → {out_path}")
        except Exception as e:
            failed.append(nid)
            print(f"⚠️ Failed to process {nid}: {e}")
            traceback.print_exc()

    # === Summary ===
    print("\n=== Summary ===")
    print(f"✅ Completed: {len(completed)} → {completed}")
    print(f"⏭️  Skipped (already done): {len(skipped)} → {skipped}")
    print(f"❌ Failed: {len(failed)} → {failed}")

    # === Check for missing items ===
    all_ids = {nid for nid, _, _ in pairs}
    done_ids = set(completed) | set(skipped)
    missing = sorted(all_ids - done_ids)
    if missing:
        print(f"\n⚠️ Missing (not processed): {len(missing)} → {missing}")
    else:
        print("\n🎯 All matched IDs have been processed or skipped.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()
