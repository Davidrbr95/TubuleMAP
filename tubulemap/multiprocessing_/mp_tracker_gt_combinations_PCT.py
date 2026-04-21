from magicgui import magicgui
from pathlib import Path
import pandas as pd
import json
import numpy as np
import os
import logging
import time
from cellpose import models as Models
import shutil
import napari
from tubulemap.utils import *
from multiprocessing import Process
from tubulemap.cellpose_tracker.tracking import run_core_function
import re
from tubulemap.cellpose_tracker.io_utils import get_max_run_number
from tubulemap.cellpose_tracker.parameters import ALL_PARAMETERS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import product

def configure_parameters(param):
    """Configure parameters."""
    final_params = {}
    for k, v in param.items():
        final_params[k] = v["default"]
    return final_params

def run_trace_core_wrapper(params):
    """Run trace core wrapper."""
    for _ in run_core_function(**params):
        pass

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

        corrected_path = os.path.join(job_name, "Next_Starting_Points.json")
        if os.path.exists(corrected_path):
            return corrected_path
    except Exception as e:
        print(f"Error reading status JSON: {e}")

    return None

def find_corrected_diameter_from_status_json(status_json_path):
    """Find corrected diameter from status json."""
    if not os.path.exists(status_json_path):
        return None

    with open(status_json_path, 'r') as f:
        data = json.load(f)

    return float(data["diameter"])    


if __name__ == '__main__':
    kp_folder = '/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/Ground_Truth_Nephrons/250405_Runs/PCT_GT'
    volume = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr/0"
    MAX_RUN_THRESHOLD = 20
    ## modification to add ability to setup cominations
    base_directory = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/Ground_Truth_Nephrons/260120_Runs/single_models/PCT"
    os.makedirs(base_directory)
    flag_combinations = list(product([False, True], repeat=3))


    resume = False

    names = []

    for z in os.listdir(kp_folder):
        if z != '.DS_Store':
            names.append(z)

    ### This is where we should define the parameter files for testing different values for variables. 
    parameters_list = []

    for combo_idx, (ul_flag, rot_flag, adapt_flag) in enumerate(flag_combinations, start=1):
        combo_dir_name = f"combination_UL_{ul_flag}_rot_{rot_flag}_adapt_{adapt_flag}"
        combo_save_dir = os.path.join(base_directory, combo_dir_name)
        os.makedirs(combo_save_dir)
        for idx, value in enumerate(names):
            ### Redefine any input parameters here (for example you could do a foorloop across different step sizes)
            # I suggest adkust the parameter "value", which is the name of the save_dir folder based on what is being tested
            # For example, say if you are testing diamter value should read something like "Nephron_1_dimater_16"
            # In summary, the forloops to test different parameters would go here. 
            default_params_copy = configure_parameters(ALL_PARAMETERS.copy())
            default_params_copy['kp_path'] = None
            default_params_copy['save_dir'] = combo_save_dir
            default_params_copy['data_set_path'] = volume
            default_params_copy['name'] = value
            default_params_copy['multiprocessing'] = True
            default_params_copy['ground_truth'] = os.path.join(kp_folder, value)
            default_params_copy['starting_model'] = 'PCT_ECI'
            default_params_copy['stepsize'] = 15
            default_params_copy['jitter'] = 30
            default_params_copy['break_distance'] = 100
            default_params_copy['diameter'] = 81.34510381079271

            ### Modifiy for combination
            default_params_copy['use_adaptive_diameter'] = adapt_flag
            default_params_copy['use_ultrack'] = ul_flag
            default_params_copy['use_rotations'] = rot_flag
            ### 

            default_params_copy['dim'] = 200

            parameters_list.append(default_params_copy)

    # Set the maximum number of processes to run at any time
    max_concurrent = 20

    start_time = time.time()

    # Dictionary that will map job_name -> process
    running_processes = {}

    # We'll keep looping until all jobs have status "done"/"error" (and no more "rerun").
    while True:
        all_finished = True
        for params in parameters_list:
            job_name = params['name']  # e.g. "my_keypoints.json"
            # Construct the status.json path from the job_name
            status_path = os.path.join(params['save_dir'], job_name[:-5] + '_status.json')
            # print(status_path)
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
                        if os.path.exists(status_path):
                            with open(status_path, 'r') as f:
                                status_data = json.load(f)
                            job_name_in_status = status_data.get("job_name", "")
                            # print('job_name_in_status', job_name_in_status)
                            m = re.search(r'/Run_(\d+)$', job_name_in_status)
                            # print('M', m)
                            if m:
                                run_num = int(m.group(1))
                                if run_num == MAX_RUN_THRESHOLD:
                                    status_data["status"] = "all_complete"
                                    with open(status_path, 'w') as f:
                                        json.dump(status_data, f, indent=4)
                                    print(f"Job {job_name} reached Run_{MAX_RUN_THRESHOLD}. Updating status to 'all_complete'.")
                                    continue
                        print(f"Starting job: {job_name} with status={status}")
                        p = Process(target=run_trace_core_wrapper, args=(params,))
                        p.start()
                        running_processes[job_name] = p
                    all_finished = False

                elif status == "rerun":
                    with open(status_path, 'r') as f:
                        status_data = json.load(f)
                    job_name_in_status = status_data.get("job_name", "")
                    m = re.search(r'/Run_(\d+)$', job_name_in_status)
                    if m:
                        run_num = int(m.group(1))
                        if run_num == MAX_RUN_THRESHOLD:
                            status_data["status"] = "all_complete"
                            with open(status_path, 'w') as f:
                                json.dump(status_data, f, indent=4)
                            print(f"Job {job_name} reached Run_{MAX_RUN_THRESHOLD}. Updating status to 'all_complete'.")
                            continue
                    if job_name not in running_processes and len(running_processes) < max_concurrent:
                        corrected_json_path = find_corrected_points_from_status_json(status_path)
                        if corrected_json_path:
                            params['ground_truth'] = corrected_json_path
                        corrected_diameter = find_corrected_diameter_from_status_json(status_path)
                        if corrected_diameter:
                            params['diameter'] =  corrected_diameter
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
                    with open(status_path, 'r') as f:
                        status_data = json.load(f)
                    job_name_in_status = status_data.get("job_name", "")
                    m = re.search(r'/Run_(\d+)$', job_name_in_status)
                    if m:
                        run_num = int(m.group(1))
                        if run_num == MAX_RUN_THRESHOLD:
                            status_data["status"] = "all_complete"
                            with open(status_path, 'w') as f:
                                json.dump(status_data, f, indent=4)
                            print(f"Job {job_name} reached Run_{MAX_RUN_THRESHOLD}. Updating status to 'all_complete'.")
                            continue
                    if job_name not in running_processes and len(running_processes) < max_concurrent:
                        corrected_json_path = find_corrected_points_from_status_json(status_path)
                        if corrected_json_path:
                            params['ground_truth'] = corrected_json_path
                        corrected_diameter = find_corrected_diameter_from_status_json(status_path)
                        if corrected_diameter:
                            params['diameter'] =  corrected_diameter
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
    
    
    end_time = time.time()
    time_result = end_time-start_time
    print("Execution time for tracking was", time_result, " seconds.")
