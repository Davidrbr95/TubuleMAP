# Boundary-safe Cellpose tracking change

This change prevents a partially out-of-volume sampling plane from presenting a
large artificial zero-filled region to Cellpose.

## What changed

- `geometry.get_frame()` calculates a matching 2-D validity mask from the slice
  transform, fills invalid pixels with the fifth-percentile background from
  valid pixels, and stores `trace.current_valid_mask`. It does not allocate a
  second 3-D volume.
- `segmentation.run_cellpose()` removes labels predicted outside that valid
  region.
- Rotation processing carries the matching validity mask for every rotated
  plane and retains the mask belonging to the selected plane.
- The main tracking loop stops only when the tracking centre itself is outside
  the real volume. Being near a boundary does not stop or downgrade tracking.
- If the primary Cellpose pass fails within one current tubule diameter of a
  volume face, eight 2-D planes are tested sequentially around two rotation
  axes. A successful plane continues the trace. If all eight fail, the existing
  points are saved and tracking stops without entering the memory-heavy 3-D
  Ultrack fallback. Interior failures still use the existing Ultrack workflow.

No tracking window size, step size, Cellpose threshold, or Ultrack setting was
changed.

## How to review or revert

Review the exact patch with:

```text
git diff -- tubulemap/cellpose_tracker/geometry.py tubulemap/cellpose_tracker/segmentation.py tubulemap/cellpose_tracker/plane_rotations.py tubulemap/cellpose_tracker/parameters.py tubulemap/cellpose_tracker/core.py docs/boundary_tracking_change.md
```

If these are the only uncommitted edits and they should all be removed, restore
the five tracked Python files with Git and delete this new documentation file.
Do not use that broad rollback when other desired edits overlap these files.
