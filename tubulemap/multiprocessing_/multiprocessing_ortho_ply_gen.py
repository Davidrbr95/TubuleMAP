from pathlib import Path
import pandas as pd
import json
import numpy as np
import os
import time
from cellpose import models as Models
from tubulemap.utils import *
from multiprocessing import Process
from tubulemap.cellpose_tracker.post_processing import run_post_process_trace
from tubulemap.cellpose_tracker.parameters import ALL_PARAMETERS
import torch
import re

# hide all GPUs
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     n = torch.cuda.device_count()
#     for idx in range(n):
#         if idx < n:
#             # Optional: you can also print out the device name
#             name = torch.cuda.get_device_name(idx)
#             print(f"Using CUDA device {idx}: {name}")
#             torch.device(f"cuda:{idx}")
#         else:
#             print(f"CUDA available but only {n} device(s) found. Falling back to CPU.")
# else:
#     print("CUDA not available. Using CPU.")



print(torch.device('cuda:0'), torch.device('cuda:1'))
print("GPUs seen:", torch.cuda.device_count())

# 1) A small helper that calls run_core_function's generator:
def run_trace_core_wrapper(params):
    """
    Wraps run_core_function, consuming its generator yield
    so we don't have to iterate ourselves.
    """
    for _ in run_post_process_trace(**params):
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

import os
import json

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

if __name__ == '__main__':
    # kp_folder = "/media/cfxuser/SSD1/Nephron_Tracking/Che_mouse_kidney_data/Completed_Tracks/whole_final"
    kp_folder = "/media/cfxuser/SSD2/Nephron_Tracking/GT/hand_drawn_masks/ground_truth_mask/Obj_nephron2/keypoints"
    # kp_folder = "/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/2025_oct_rerun_nephrons"
    # volume =  "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/M3.zarr"
    volume = "/media/cfxuser/SSD11/Nephron_Tracking/Che_mouse_kidney_data/oldeci.zarr/0"
    # default_parameters = 'tubulemap/cellpose_tracker/default_values.json'
    
    ### Lets say you complete your tracking and you edit your session - if resume == true then you will run the tracks that have been flagged for "rerun"
    resume = False
    MAX_RUN_THRESHOLD = 1

    names = []
    # default_params = None

    for z in os.listdir(kp_folder):
        if z != '.DS_Store':
            names.append(z)


    parameters_list = []
    # previous_runs1 = os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/all_nephron_ply")
    # previous_runs1 = [f.replace('_status', '') for f in previous_runs1]

    # previous_runs2 = os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/all_nephron_ply_2")
    # previous_runs2 = [f.replace('_status', '') for f in previous_runs2]
    
    # previous_runs3 = os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/all_nephron_ply_3")
    # previous_runs3 = [f.replace('_status', '') for f in previous_runs3]
    
    # previous_runs4 = os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/all_nephron_ply_4")
    # previous_runs4 = [f.replace('_status', '') for f in previous_runs4]
    
    # previous_runs5 = os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/all_nephron_ply_5")
    # previous_runs5 = [f.replace('_status', '') for f in previous_runs5]
    
    
    # previous_runs = set(previous_runs1 + previous_runs2 + previous_runs3 + previous_runs4+previous_runs5)
    # names = list(set(names)-previous_runs)
    # names = ['N1358.json', 'N1211.json', 'N1290.json', 'N0823.json', 'N1396.json', 'N1301.json', 'N0757.json', 'N0816.json', 
    #          'N1411.json', 'N1482.json', 'N1375.json', 'N1462.json', 'N1410.json', 'N1377.json', 'N1314.json', 'N1471.json', 
    #          'N1364.json', 'N1395.json', 'N1361.json', 'N1407.json', 'N1363.json', 'N1475.json', 'N1316.json', 'N0742.json', 
    #          'N1464.json', 'N1446.json', 'N1401.json', 'N0881.json', 'N1259.json', 'N1483.json', 'N1437.json', 'N0868.json', 
    #          'N1362.json', 'N1392.json', 'N1431.json', 'N1485.json', 'N1438.json', 'N1434.json', 'N1394.json', 'N0430.json', 
    #          'N1442.json', 'N1359.json', 'N1399.json', 'N1428.json', 'N1230 no thinlimb.json', 'N1269.json', 'N0940.json', 
    #          'N1417.json', 'N1205.json', 'N0358.json', 'N1374.json', 'N1360.json', 'N1317.json', 'N1478.json', 
    #          'N1309.json', 'N1391.json', 'N1315.json', 'N1304.json', 'N0579.json', 'N1412 no thinlimb.json', 'N0604.json']
    # names = ['GT_1.json']
    # ply_file_names = [i.split('_')[0]+'.json' for i in os.listdir("/media/cfxuser/SSD2/Nephron_Tracking/Che_mouse_kidney_data/cloud_files")]
    # print(ply_file_names)
    # print(names[:4], ply_file_names[:4])
    # names = list(set(names) - set(ply_file_names))
    # print('FINAL NAMES', len(names))
    # exit()
    for idx, value in enumerate(names):
        # if 'N0358' not in value:
        #     continue
        # "value" might be something like "my_keypoints.json"
        default_params_copy = configure_parameters(ALL_PARAMETERS.copy())
        default_params_copy['kp_path'] = os.path.join(kp_folder, value)
        default_params_copy['data_set_path'] = volume
        default_params_copy['name'] = value  # used to build the status JSON
        default_params_copy['multiprocessing'] = True
        default_params_copy['save_dir'] = "/media/cfxuser/SSD2/Nephron_Tracking/GT/hand_drawn_masks/ground_truth_mask/Obj_nephron2/orthoplanes"  # location of status files you need to change it for each new session ()
        default_params_copy['model_suite'] = 'tubulemap/cellpose_tracker/models'
        default_params_copy['write_ply'] = False
        default_params_copy['resample_step_size'] = 10
        default_params_copy['starting_model'] = 'PCT_ECI'
        default_params_copy['stepsize'] = 15
        default_params_copy['jitter'] = 30
        default_params_copy['diameter'] = 81.34510381079271


        ### Modifiy for combination
        default_params_copy['use_adaptive_diameter'] = True
        default_params_copy['use_ultrack'] = True
        default_params_copy['use_rotations'] = True
        ### 

        default_params_copy['dim'] = 200
    
        parameters_list.append(default_params_copy)

    # Set the maximum number of processes to run at any time
    max_concurrent = 5

    # Dictionary that will map job_name -> process
    running_processes = {}

    # We'll keep looping until all jobs have status "done"/"error" (and no more "rerun").
    while True:
        print('len parameters list', len(parameters_list))
        # all_finished = True
        for params in parameters_list:
            job_name = params['name']  # e.g. "my_keypoints.json"
            # Construct the status.json path from the job_name
            status_path = os.path.join(params['save_dir'], job_name[:-5] + '_status.json')
            status = read_status(status_path)
            if os.path.isfile(status_path):
                with open(status_path, 'r') as f:
                    status_data = json.load(f)
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
                        if os.path.isfile(status_path):
                            job_name_in_status = status_data.get("job_name", "")
                            m = re.search(r'/Run_(\d+)$', job_name_in_status)
                            if m:
                                run_num = int(m.group(1))
                                if run_num >= MAX_RUN_THRESHOLD:
                                    status_data["status"] = "all_complete"
                                    with open(status_path, 'w') as f:
                                        json.dump(status_data, f, indent=4)
                                    print(f"Job {job_name} reached Run_{MAX_RUN_THRESHOLD}. Updating status to 'all_complete'.")
                                    continue
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
                        if os.path.isfile(status_path):
                            job_name_in_status = status_data.get("job_name", "")
                            m = re.search(r'/Run_(\d+)$', job_name_in_status)
                            if m:
                                run_num = int(m.group(1))
                                if run_num >= MAX_RUN_THRESHOLD:
                                    status_data["status"] = "all_complete"
                                    with open(status_path, 'w') as f:
                                        json.dump(status_data, f, indent=4)
                                    print(f"Job {job_name} reached Run_{MAX_RUN_THRESHOLD}. Updating status to 'all_complete'.")
                                    continue
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
