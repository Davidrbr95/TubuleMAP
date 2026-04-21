<h1>
    <div>
        <img src="icon.png" alt="TubuleMAP" width="50" height="50">
    </div>
    TubuleMAP
</h1>

## Abstract
Advances in tissue clearing and lightsheet microscopy enable mesoscale imaging of intact and convoluted tubular networks, yet analytical tools to map tubule continuity and assess injury patterns within and across tubules are limited. Here, we introduce TubuleMAP, a semi-automated pipeline for 3D tubule tracking and reconstruction that adapts to various morphological and staining patterns, leverages parallel processing of terabyte-scale data for large-scale analysis of tubular networks, and uses a napari interface for human oversight. Using TubuleMAP, we reconstruct 1,000 mouse nephrons in ~1-millimeter-thick kidney slab with ~400-fold higher throughput and <1% human effort compared to prior approaches. These reconstructions enable analysis of mesoscale nephron organization, quantitative profiling of pathologic morphologies, whole-nephron cytometry, and identification of rare morphologies at unprecedented scales. We demonstrate generalizability by reconstructing all seminiferous tubules in a mouse testis within a day. TubuleMAP is released as an open-source Python package.  
## Quick demo
[Watch the demo video](./videos/demo.mp4)
## Installation

1. Create a virtual environment and activate it:
    On Windows:

    ```bash
    conda create -n tubulemap python=3.11.0 ipython
    conda activate tubulemap
    ```

    or

    ```bash
    python -m venv tubulemap
    .\tubulemap\Scripts\activate
    ```

2. [On mac silicon only] Install higra from conda-forge.
    ```bash
    conda install -c conda-forge higra=0.6.10
    ```

3. To develop the code, run:

    ```bash
    pip install -e "git+ssh://git@github.com/Davidrbr95/TubuleTracker.git@Feb2025Refactor#egg=tubulemap[dev]"
    ```
