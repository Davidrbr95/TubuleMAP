<h1>
    <div>
        <img src="icon.png" alt="TubuleMAP" width="50" height="50">
    </div>
    TubuleMAP
</h1>

## Abstract
Advances in tissue clearing and lightsheet microscopy enable mesoscale imaging of intact and convoluted tubular networks, yet analytical tools to map tubule continuity and assess injury patterns within and across tubules are limited. Here, we introduce TubuleMAP, a semi-automated pipeline for 3D tubule tracking and reconstruction that adapts to various morphological and staining patterns, leverages parallel processing of terabyte-scale data for large-scale analysis of tubular networks, and uses a napari interface for human oversight. Using TubuleMAP, we reconstruct 1,000 mouse nephrons in ~1-millimeter-thick kidney slab with ~400-fold higher throughput and <1% human effort compared to prior approaches. These reconstructions enable analysis of mesoscale nephron organization, quantitative profiling of pathologic morphologies, whole-nephron cytometry, and identification of rare morphologies at unprecedented scales. We demonstrate generalizability by reconstructing all seminiferous tubules in a mouse testis within a day. TubuleMAP is released as an open-source Python package.  
## Quick demo
[Watch the demo video](./videos/highlevel_overview.mp4)
## Installation

1. Create a virtual environment with Python 3.11.0–3.11.11 and activate it.

    On Windows using Conda:

    ```bash
    conda create -n tubulemap python=3.11.11 ipython
    conda activate tubulemap
    ```

    Alternatively, if you already have a compatible Python version installed, you can use `venv`:

    ```bash
    python -m venv tubulemap
    .\tubulemap\Scripts\activate
    ```

    Verify that the Python version is between 3.11.0 and 3.11.11:

    ```bash
    python --version
    ```

2. [On mac silicon only] Install higra from conda-forge.
    ```bash
    conda install -c conda-forge higra=0.6.10
    ```

3. To develop the code, clone the repository and install it in editable mode:

    ```bash
    git clone https://github.com/Davidrbr95/TubuleMAP.git
    cd TubuleMAP
    pip install -e ".[dev]"
    ```


## Demo

A small example dataset and a predefined seed point are provided in the `demos` directory for testing the TubuleMAP tracking workflow.

The demo files include:

```text
demos/
├── main_demo.py
├── Kidney_demo.zarr/
├── test_seed_points/
│   └── test1.json
└── expected_results/
```

To run the demo, activate the TubuleMAP environment and run:

```bash
conda activate tubulemap
python demos/main_demo.py
```

The demo automatically loads `Kidney_demo.zarr`, uses the provided seed point in `test_seed_points/test1.json`, and configures the tracking parameters for the example dataset.

Tracking results are saved to:

```text
demos/test_tracking_results/
```

A representative expected tracking result is provided in:

```text
demos/expected_results/
```

The expected output is a 3D reconstructed tubule trajectory starting from the provided seed point.

Typical demo runtime: approximately **5 minutes** on a GPU-enabled workstation.
