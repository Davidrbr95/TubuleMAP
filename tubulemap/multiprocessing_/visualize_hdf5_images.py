import h5py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def add_scalebar(ax, pixel_size_um=0.42, bar_length_um=50, location="lower right",
                 color="white", linewidth=4, pad=0.4, sep=5, fontsize=10):
    """
    Add a scalebar to an axes in image pixel coordinates.
    """
    bar_length_px = bar_length_um / pixel_size_um  # um -> px
    fontprops = fm.FontProperties(size=fontsize)

    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length_px,
        f"{bar_length_um:g} µm",
        loc=location,
        pad=pad,
        borderpad=0.5,
        sep=sep,
        frameon=False,
        color=color,
        size_vertical=0,          # line instead of a filled rectangle
        fontproperties=fontprops,
    )
    # Make the line thicker
    scalebar.size_vertical = 0
    # scalebar.txt_label.set_color(color)
    ax.add_artist(scalebar)

    # AnchoredSizeBar doesn't directly expose linewidth cleanly for all mpl versions,
    # so we can post-adjust the children if needed:
    for child in scalebar.get_children():
        try:
            child.set_linewidth(linewidth)
        except Exception:
            pass


def load_and_plot_hdf5(hdf5_path, save_path="hdf5_plot.png", max_rows_per_figure=10,
                       pixel_size_um=0.42, scalebar_um=50):
    """Load and plot hdf5."""
    with h5py.File(hdf5_path, "r") as h5f:
        mask_keys = sorted([k for k in h5f.keys() if k.startswith("mask_")])
        raw_keys  = sorted([k for k in h5f.keys() if k.startswith("raw_")])

        num_pairs = min(len(mask_keys), len(raw_keys))
        if num_pairs == 0:
            print("No valid mask/raw pairs found in HDF5.")
            return

        num_figures = (num_pairs // max_rows_per_figure) + (1 if num_pairs % max_rows_per_figure else 0)

        for fig_idx in range(num_figures):
            start_idx = fig_idx * max_rows_per_figure
            end_idx = min((fig_idx + 1) * max_rows_per_figure, num_pairs)
            num_images = end_idx - start_idx

            fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
            if num_images == 1:
                axes = np.array([axes])

            for i, (mask_key, raw_key) in enumerate(zip(mask_keys[start_idx:end_idx], raw_keys[start_idx:end_idx])):
                mask_img = h5f[mask_key][:]
                raw_img  = h5f[raw_key][:]

                axes[i, 0].imshow(mask_img, cmap="gray")
                axes[i, 0].set_title(f"Mask {raw_key}")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(raw_img, cmap="gray")
                axes[i, 1].set_title(f"Raw {raw_key}")
                axes[i, 1].axis("off")

                # Add scalebar to BOTH panels (or remove one if you only want it on raw)
                add_scalebar(axes[i, 0], pixel_size_um=pixel_size_um, bar_length_um=scalebar_um,
                             location="lower right", color="white", linewidth=4, fontsize=10)
                add_scalebar(axes[i, 1], pixel_size_um=pixel_size_um, bar_length_um=scalebar_um,
                             location="lower right", color="white", linewidth=4, fontsize=10)

            plt.tight_layout()
            save_filename = f"{os.path.splitext(save_path)[0]}_{fig_idx}.png"
            plt.savefig(save_filename, dpi=300)
            plt.close(fig)
            print(f"Saved plot to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mask/raw pairs from an HDF5 file.")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--save_path", type=str, default="hdf5_plot.png", help="Base path to save the plots.")
    parser.add_argument("--max_rows_per_figure", type=int, default=10, help="Max number of mask/raw pairs per figure.")
    parser.add_argument("--pixel_size_um", type=float, default=0.42, help="Pixel size in microns (µm/px).")
    parser.add_argument("--scalebar_um", type=float, default=50.0, help="Scale bar length in microns.")
    args = parser.parse_args()

    load_and_plot_hdf5(args.hdf5_path, args.save_path, args.max_rows_per_figure,
                       pixel_size_um=args.pixel_size_um, scalebar_um=args.scalebar_um)

# example call:
# /Users/davidbrenes/anaconda3/envs/ultrack_tubule/bin/python "/Users/davidbrenes/Dropbox/Liu Lab/NephronSegmentation/TubuleTracker/visualize_hdf5_images.py" --save_path output_plot.png --max_rows_per_figure 10 "/Users/davidbrenes/Dropbox/Liu Lab/NephronSegmentation/TubuleTracker/Generate_ply/result_trace.json/Run_0/ortho_planes.hdf5"

# python "/home/cfxuser/src/tubule-tracker/tubulemap/visualize_hdf5_images.py" --save_path output_plot.png --max_rows_per_figure 10 "/home/cfxuser/src/tubule-tracker/ORTHO/GT_8.json/Run_0/ortho_planes.hdf5"

# python "/home/cfxuser/src/tubule-tracker/tubulemap/visualize_hdf5_images.py" --save_path output_plot.png --max_rows_per_figure 10 "/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/gt_nephron_ortho_planes/GT_1.json/Run_0/ortho_planes.hdf5"
