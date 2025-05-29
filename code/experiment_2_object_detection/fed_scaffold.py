#!/usr/bin/env python
# coding: utf-8

# <h4> This code is part of FLAUTO. It implements FedScaffold. Date: 01/09/2025 </h4>
# <h4> Contact: rakibul.haque@utsa.edu </h4>  
# <h4> Cite as: R. U. Haque and P. Markopoulos,"Federated Learning with Automated Dual-Level Hyperparameter Tuning", 2025 <h4>

# In[ ]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO
import shutil
import os
import time
import pickle
import json
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# print function
def Print(string, dictionary):
    first_key = next(iter(dictionary))
    first_value = dictionary[first_key]
    print(f"{string}:{first_key}: {first_value[0][0]}\n")

# deleting run folder for saving spaces
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully!")
    else:
        print(f"Folder '{folder_path}' does not exist.")

#averaging the weihgts


# In[ ]:


def average_weights_function(dicts):
    # Initialize an empty dictionary to store the summed weights
    summed_weights = {}

    # Initialize an empty dictionary to keep track of how many times each key is seen
    key_occurrences = {}

    # Iterate through all the dictionaries
    for d in dicts:
        for key, value in d.items():
            if key in summed_weights:
                summed_weights[key] += value
                key_occurrences[key] += 1
            else:
                summed_weights[key] = value
                key_occurrences[key] = 1

    # Create a dictionary to store the final averaged weights
    averaged_weights = {}

    # Iterate through the summed weights and divide by the number of occurrences for each key
    for key, value in summed_weights.items():
        if key_occurrences[key] > 1:
            averaged_weights[key] = value / key_occurrences[key]
        else:
            averaged_weights[key] = value  # If only present in one dict, keep as is

    return averaged_weights


# In[ ]:


def training(i_w, E, r, c_no, learning_rate, ci, c, device):
    # Declare local model and move to the specified device
    local_model = YOLO("/data/ex2/initial_weights.pt").to(device)
    local_model.load_state_dict(i_w)
    print(local_model.info())

    # Initialize local model with the global model weights
    yi = {k: v.clone().float().to(device) for k, v in i_w.items()}

    # Ensure control variates are float
    ci = {k: v.float().to(device) for k, v in ci.items()}
    c = {k: v.float().to(device) for k, v in c.items()}

    # Adjust gradients with control variate (c - ci)
    with torch.no_grad():  # Ensure no gradient tracking for this operation
        for k in local_model.state_dict().keys():
            param = local_model.state_dict()[k].float()
            local_model.state_dict()[k] = param - (ci[k] - c[k])

    # Epochs
    local_model.train(
        data=f"/data/ex2/{fl_a}/{set_up}/c{c_no}.yaml", 
        project=f"{dst_folder}/train/round_{r}_client_{c_no}", 
        workers=0, 
        epochs=E, 
        imgsz=512, 
        lr0=learning_rate, 
        batch=4, 
        optimizer=opti, 
        val=True,
        warmup_epochs=0
    )
    # Collecting final weights
    client_final_weights = {k: v.clone().float().to(device) for k, v in local_model.state_dict().items()}

    # Update client control variate ci using Option II
    new_ci = {
        k: ci[k] - c[k] + (1 / (E * learning_rate)) * (i_w[k].float() - client_final_weights[k])
        for k in ci.keys()
    }

    return client_final_weights, new_ci


# In[ ]:


def federated_learning(i_w, C, P, R, E, learning_rate, b_size):
    # Initialize global model and control variates
    global global_model
    global validation_dict
    global dst_folder
    global device

    global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
    global_model.load_state_dict(i_w)
    c = {k: torch.zeros_like(v, dtype=torch.float).to(device) for k, v in i_w.items()}  # Initialize global control variate to zero
    ci_list = [{k: torch.zeros_like(v, dtype=torch.float).to(device) for k, v in i_w.items()} for _ in range(C)]  # Initialize client control variates to zero

    # Loop for each round
    for r in range(1, R + 1):
        models = []
        i_w = {k: v.clone().to(device).float() for k, v in global_model.state_dict().items()}

        # Loop for each client
        for n_no in range(C):
            c_i = ci_list[n_no]  # Fetch the client-specific control variate

            # Training
            clients_weight, new_ci = training(i_w, E, r, n_no + 1, learning_rate, c_i, c, device)
            models.append(clients_weight)

            # Update the client-specific control variate
            ci_list[n_no] = new_ci

        # Compute average weights for the global model
        average_weights = average_weights_function(models)

        # Save the averaged weights
        os.makedirs(os.path.join(dst_folder, "weights"), exist_ok=True)
        torch.save(average_weights, f'{dst_folder}/weights/after_round_{r}_weights.pt')

        # Update global model and control variate
        global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        global_model.load_state_dict(average_weights)

        # Ensure the summation result is in float type
        for k in c.keys():
            # Convert to float before summing and assigning
            ci_sum = sum([ci_list[n_no][k].float() for n_no in range(C)])
            c[k] += (1 / C) * ci_sum

        val_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        val_model.load_state_dict(average_weights)

        # Perform round validations
        validation_results = val_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_{r}", imgsz=512, batch=4, split='val',workers=0)
        validation_dict[f"round_{r}"] = validation_results

        print("round", r, "completed")


# In[ ]:


#===========================Parameters==============================================================
round_no=30
client_no=4
participating_client=client_no
learning_rate=0.01
batch_size=4
epochs=5
opti='SGD'


fl_a="yaml"

set_up="IID"
# set_up="limited_data"

forname=set_up

#===========other variables=============================================
validation_dict = {}

dst_folder = f"{fl_a}_{forname}_Fed_Scaffold_{learning_rate}_{opti}"
delete_folder(dst_folder)

#===================================loading the saved weight list====================================================
global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
global_model.info()
initial_weights = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
print(len(initial_weights))
Print("Model's initial weights", initial_weights)
# global_model.save('current.pt')


# In[ ]:


l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
#server validation rounds
validation_results = l_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_0", imgsz=512, batch=4,split='val', workers=0)
validation_dict["round_0"] = validation_results
print(validation_results)

#=================================================================client_1====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c1.yaml", project=f"{dst_folder}/train/round_0_client_1", imgsz=512, batch=4, split='train', workers=0)

#=================================================================client_2====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c2.yaml", project=f"{dst_folder}/train/round_0_client_2", imgsz=512, batch=4, split='train', workers=0)

#=================================================================client_3====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c3.yaml", project=f"{dst_folder}/train/round_0_client_3", imgsz=512, batch=4, split='train', workers=0)

#=================================================================client_4====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c4.yaml", project=f"{dst_folder}/train/round_0_client_4", imgsz=512, batch=4, split='train',workers=0)
# clear_output(wait=False)

federated_learning(initial_weights, client_no, participating_client, round_no, epochs, learning_rate, batch_size)

# Convert the dict to a serializable format
def dict_to_serializable(d):
    serializable_dict = {}
    for key, value in d.items():
        if isinstance(value, (int, float, str, list, dict)):
            serializable_dict[key] = value
        else:
            serializable_dict[key] = str(value)  # Convert non-serializable types to string
    return serializable_dict

# Save as JSON
save_dir = dst_folder
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, 'validation_dict.json')

with open(file_path, 'w') as f:
    json.dump(dict_to_serializable(validation_dict), f, indent=4)

print(f"Validation dictionary saved to {file_path}")

file_path = os.path.join(save_dir, 'validation_dict.json')

# Load the JSON file
with open(file_path, 'r') as f:
    loaded_dict = json.load(f)

# Print the loaded dictionary
print("Validation dictionary loaded successfully")

