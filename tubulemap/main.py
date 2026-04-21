import os
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Keep napari/numba runtime defaults local and writable across launch contexts.
os.environ.setdefault("NAPARI_ASYNC", "1")
os.environ.setdefault("NAPARI_OCTREE", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

import napari
from qtpy.QtWidgets import QVBoxLayout, QWidget, QTabWidget
from tubulemap.widgets import (PointsWidget,
                     ZarrLoaderWidget,
                     TubuleTrackerWidget,
                     HumanInLoopWidget,
                     DownsampleControlWidget,
                    )
#from dask_image.imread import imread

# os.environ['NAPARI_ASYNC'] = '1'
# os.environ['NAPARI_OCTREE'] = '0'
# os.environ['NAPARI_OCTREE'] = './octree.json'


def create_widgets(viewer):
    """Create and return the custom widgets."""
    downsample_control_widget = DownsampleControlWidget(viewer)
    points_widget = PointsWidget(viewer)
    zarr_loader_widget = ZarrLoaderWidget(viewer)
    tubulemap_widget = TubuleTrackerWidget(viewer)
    human_in_loop_widget = HumanInLoopWidget(viewer)
    left_widgets = [downsample_control_widget, zarr_loader_widget]
    right_widgets = [points_widget, tubulemap_widget, human_in_loop_widget]
    return left_widgets, right_widgets

def setup_layout(left_widgets, right_widgets):
    """Setup the layout and add the provided widgets."""
    layout_left = QWidget()
    setup_layout = QVBoxLayout(layout_left)
    
    for widget in left_widgets:
        setup_layout.addWidget(widget)

    layout_right = QTabWidget()
    for widget in right_widgets:
        tab = QWidget()
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(widget)
        tab.setLayout(tab_layout)
        layout_right.addTab(tab, widget.name)

    return layout_left, layout_right

def main():
    # Initialize the napari viewer
    """Launch napari and register the Tubule Tracker widgets."""
    viewer = napari.Viewer()

    # Create custom widgets
    left_widgets, right_widgets = create_widgets(viewer)

    # Setup the main layout and include custom widgets
    layout_left, layout_right = setup_layout(left_widgets, right_widgets)

    # Add the custom UI to the napari viewer window
    viewer.window.add_dock_widget(layout_left, area='left', name='Main operations')
    viewer.window.add_dock_widget(layout_right, area='right', name='Tracking')


    # Josh 
    # zarr_store = tifffile.imread("/media/cfxuser/SSD1/Nephron_Tracking/Che_mouse_kidney_data/oldeci.tif", aszarr=True)
    # data = dask.array.from_zarr(zarr_store)
    # viewer.add_image(data, rgb=False, contrast_limits=[0, 2**16])

    # David 
    # zarr_file = '/media/cfxuser/SSD1/Nephron_Tracking/Che_mouse_kidney_data/halfkidney.zarr/fused_tp_0_ch_0'
    # zarr_store = zarr.open(zarr_file, mode='r')
    # print(zarr_store)
    # print('here')
    # data = da.array.from_zarr(zarr_store)
    # viewer.add_image(data, rgb=False)

    # zarr_file = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr/0"
    # zarr_store = zarr.open(zarr_file, mode='r')
    # print(zarr_store)
    # data = da.array.from_zarr(zarr_store)
    # viewer.add_image(data, rgb=False, contrast_limits=[0, 2**16])

    

    # zarr_store = tifffile.imread(r"D:\Datasets\fused_tp_0_ch_1.tif", aszarr=True)
    # data = dask.array.from_zarr(zarr_store)
    # viewer.add_image(data, rgb=False, contrast_limits=[0, 2**16])

    # David 
    # zarr_file = '/Users/davidbrenes/Dropbox/Liu Lab/NephronSegmentation/test_2.zarr'
    # zarr_store = zarr.open(zarr_file, mode='r')
    # print(zarr_store)
    # print('here')
    # data = da.array.from_zarr(zarr_store)
    # viewer.add_image(data, rgb=False)
    # ome_zarr_path = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr"
    # viewer.open(ome_zarr_path, plugin="napari-ome-zarr")

    # Start the napari event loop
    napari.run()


if __name__ == "__main__":
    main()

# zarr_store.close()
