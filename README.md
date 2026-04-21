<h1>
    <div>
        <img src="tubi_icon.png" alt="TubuleMAP" width="50" height="50">
    </div>
    Tubi-Tracker
</h1>

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
