import os
import subprocess
import multiprocessing
from venv import create
import regex as re
import json
from sklearn.metrics import coverage_error
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")

lr_list = [
    0.05,
]
epoch_list = [
    300,
]
num_inducing_list = [
    150,
]
samples_list = [
    7,
]
fold_list = [0, 1, 2]

# lr_list=[0.1]
# epoch_list=[1,]
# num_inducing_list=[12]
# samples_list = [3]
# fold_list=[2,]


results_file = "results.jl"
base_path = os.path.dirname(__file__)


class Config:
    n_processes_multiprocessing = 5


def get_precision_values(metrics_output):
    rmse = float(re.findall(r"[0-9]+.[0-9]+", metrics_output)[-4])
    msll = float(re.findall(r"[0-9]+.[0-9]+", metrics_output)[-3])
    nlpd = float(re.findall(r"[0-9]+.[0-9]+", metrics_output)[-2])
    coverage_error = float(re.findall(r"[0-9]+.[0-9]+", metrics_output)[-1])
    return rmse, msll, nlpd, coverage_error


def get_results(params_dict):
    try:
        os.chdir(base_path)
        script_command = f"python doubly_stochastic.py --num_inducing {params_dict['num_inducing']} --lr {params_dict['lr']} --epochs {params_dict['epoch']} --samples {params_dict['samples']} --fold {params_dict['fold']}"
        print(params_dict["iter_id"])
        print(script_command)

        # return

        # Create Model Directory
        program_output = os.popen(script_command).read()
        # print(metric_output)
        rmse, msll, nlpd, coverage_error = get_precision_values(program_output)

        return params_dict, (rmse, msll, nlpd, coverage_error)
    except Exception as err:
        print("ERROR", err)
        print(os.popen("pwd").read())
        # print(dataset_creation_output)
        print(params_dict)
        return None


parameters_dict_list = []
iter_id = 0


for epoch in epoch_list:
    for lr in lr_list:
        for num_inducing in num_inducing_list:
            for samples in samples_list:
                for fold in fold_list:
                    dict_obj = {}
                    dict_obj["epoch"] = epoch
                    dict_obj["lr"] = lr
                    dict_obj["num_inducing"] = num_inducing
                    dict_obj["samples"] = samples
                    dict_obj["iter_id"] = iter_id
                    dict_obj["fold"] = fold
                    parameters_dict_list.append(dict_obj)
                    iter_id += 1


print(len(parameters_dict_list))
np.random.shuffle(parameters_dict_list)


pool = multiprocessing.Pool(Config.n_processes_multiprocessing)
for x in tqdm(
    pool.imap(get_results, parameters_dict_list), total=len(parameters_dict_list)
):
    if x is None:
        continue
    parameters_dict, (rmse, msll, nlpd, coverage_error) = x
    parameters_dict["rmse"] = rmse
    parameters_dict["msll"] = msll
    parameters_dict["nlpd"] = nlpd
    parameters_dict["coverage_error"] = coverage_error
    json_str = json.dumps(parameters_dict)
    with open(results_file, "a") as jl_file:
        print(json_str, file=jl_file)

pool.close()
