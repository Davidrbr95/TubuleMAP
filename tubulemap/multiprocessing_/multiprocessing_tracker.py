import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process

# Allow running this script via absolute path (outside repo root cwd),
# e.g.: python /path/to/tubulemap/multiprocessing_tracker.py
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tubulemap.cellpose_tracker.tracking import run_core_function
from tubulemap.cellpose_tracker.parameters import ALL_PARAMETERS

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 1) A small helper that calls run_core_function's generator:
def run_trace_core_wrapper(params):
    """
    Wraps run_core_function, consuming its generator yield
    so we don't have to iterate ourselves.
    """
    for _ in run_core_function(**params):
        pass

# 2) A helper to read job status from the JSON file:
def read_status(status_path):
    """Return the 'status' field from the JSON file, or 'unknown' on error."""
    if not os.path.exists(status_path):
        return "unknown"
    try:
        with open(status_path, 'r') as f:
            data = json.load(f)
        return data.get("status", "unknown")
    except:
        return "unknown"

def find_corrected_points_from_status_json(status_json_path):
    """
    Given the path to a status JSON that has a 'job_name' field,
    return the full path to 'corrected_points.json' inside that job folder,
    e.g. job_name='Human_in_loop/Track_nephron_1.json/Run_0'
    -> 'Human_in_loop/Track_nephron_1.json/Run_0/corrected_points.json'.

    Returns the path if it exists, else None.
    """
    if not os.path.exists(status_json_path):
        return None

    try:
        with open(status_json_path, 'r') as f:
            data = json.load(f)
        job_name = data.get("job_name", "")
        if not job_name:
            return None

        corrected_path = os.path.join(job_name, "corrected_points.json")
        if os.path.exists(corrected_path):
            return corrected_path
    except Exception as e:
        print(f"Error reading status JSON: {e}")

    return None

def configure_parameters(param):
    """Configure parameters."""
    final_params = {}
    for k, v in param.items():
        final_params[k] = v["default"]
    return final_params


def _job_stem(name):
    """Return job stem used by setup_logging_and_folders for status file naming."""
    job_name = str(name)
    return job_name[:-5] if job_name.endswith(".json") else job_name

if __name__ == '__main__':
    # kp_folder = '/media/cfxuser/SSD2/Nephron_Tracking/GT/seedpoints'
    # volume = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr/0"
    kp_folder = '/home/cfxuser/src/tubule-tracker/tubulemap/Demo/starting_points'
    volume = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/demo_data/oldeci_bbox_cropped.zarr"
    
    ### Lets say you complete your tracking and you edit your session - if resume == true then you will run the tracks that have been flagged for "rerun"
    resume = False

    names = []
    # default_params = None

    for z in os.listdir(kp_folder):
        if z != '.DS_Store':
            names.append(z)

    # with open(default_parameters, 'r') as file:
    #     default_params = json.load(file)

    parameters_list = []
    for idx, value in enumerate(names):
        # "value" might be something like "my_keypoints.json"
        default_params_copy = configure_parameters(ALL_PARAMETERS.copy())
        default_params_copy['kp_path'] = os.path.join(kp_folder, value)
        default_params_copy['data_set_path'] = volume
        default_params_copy['name'] = value  # used to build the status JSON
        default_params_copy['multiprocessing'] = False
        default_params_copy['save_dir'] = 'test_demo_new'  # location of status files you need to change it for each new session ()
        default_params_copy['starting_model'] = 'PCT_ECI'
        default_params_copy['iterations'] = 20
        default_params_copy['stepsize'] = 15
        default_params_copy['jitter'] = 30
        default_params_copy['diameter'] = 81
        default_params_copy['use_adaptive_diameter'] = True
        default_params_copy['use_ultrack'] = True
        default_params_copy['use_rotations'] = True
        default_params_copy['dim'] = 200
        parameters_list.append(default_params_copy)

    # Set the maximum number of processes to run at any time
    max_concurrent = 2

    # Dictionary that will map job_name -> process
    running_processes = {}

    # We'll keep looping until all jobs have status "done"/"error" (and no more "rerun").
    while True:
        all_finished = True
        for params in parameters_list:
            job_name = params['name']  # e.g. "my_keypoints.json"
            print(job_name, 'job name')
            # Construct the status.json path from the job_name
            status_path = os.path.join(params['save_dir'], _job_stem(job_name) + '_status.json')
            status = read_status(status_path)

            proc = running_processes.get(job_name)
            # If we have a process, see if it's still alive
            if proc is not None:
                if not proc.is_alive():
                    proc.join()
                    running_processes.pop(job_name)
                    # re-check status after finishing
                    status = read_status(status_path)

            # If resume == True, we only run rerun.
            if not resume:
                # Original logic
                if status in ["unknown", "running"]:
                    if job_name not in running_processes and len(running_processes) < max_concurrent:
                        print(f"Starting job: {job_name} with status={status}")
                        p = Process(target=run_trace_core_wrapper, args=(params,))
                        p.start()
                        running_processes[job_name] = p
                    all_finished = False

                elif status == "rerun":
                    if job_name not in running_processes and len(running_processes) < max_concurrent:
                        corrected_json_path = find_corrected_points_from_status_json(status_path)
                        if corrected_json_path:
                            params['kp_path'] = corrected_json_path
                        print(f"Re-running job: {job_name}")
                        p = Process(target=run_trace_core_wrapper, args=(params,))
                        p.start()
                        running_processes[job_name] = p
                    all_finished = False

                elif status in ["done", "error", "all_complete"]:
                    pass
                else:
                    # unexpected status
                    all_finished = False
            else:
                # resume == True -> only run processes if status == 'rerun'
                if status == "rerun":
                    if job_name not in running_processes and len(running_processes) < max_concurrent:
                        corrected_json_path = find_corrected_points_from_status_json(status_path)
                        if corrected_json_path:
                            params['kp_path'] = corrected_json_path
                        print(f"[RESUME MODE] Re-running job: {job_name} with status='rerun'")
                        p = Process(target=run_trace_core_wrapper, args=(params,))
                        p.start()
                        running_processes[job_name] = p
                    all_finished = False
                else:
                    # For 'unknown', 'running', 'done', 'error', we do nothing
                    pass

        # If everything is "done" or "error" (and no "rerun"), we can break
        if all_finished and not running_processes:
            print("All jobs are done (or errored) and no processes are running. Exiting.")
            break

        time.sleep(5)

    print("All processes have completed (no more rerun jobs).")
    print("All processes have completed (no more rerun jobs).")
