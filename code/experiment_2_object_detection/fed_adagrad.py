#!/usr/bin/env python
# coding: utf-8


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
from IPython.display import clear_output
import time
import pickle
import json
# from IPython.display import clear_output
import copy
from copy import deepcopy
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


def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg


def training(i_w, E, r, c):
    global learning_rate, epochs
    # Initialize the local model
    local_model = YOLO("/data/ex2/initial_weights.pt").to(device)
    local_model.load_state_dict(i_w)

    # Perform local training
    local_model.train(data=f"/data/ex2/{fl_a}/{set_up}/c{c}.yaml", 
                      project=f"{dst_folder}/train/round_{r}_client_{c}", 
                      workers=0, 
                      epochs=epochs,  # Ensure epochs is an integer
                      imgsz=512, 
                      lr0=learning_rate,  # Ensure learning_rate is float
                      split='train',
                      batch=4, 
                      optimizer=opti,  # Ensure optimizer is correctly specified
                      val=True, device=0, warmup_epochs=0)

    # Collect final weights
    client_final_weights = {k: v.clone().float().to(device) for k, v in local_model.state_dict().items()}

    model_update = {}
    for key in local_model.state_dict():
        model_update[key] = torch.sub(i_w[key], client_final_weights[key])

    return model_update


def federated_learning(i_w, C, P, R, E, b_size, lr=0.001, tau=1e-3):
    global global_model
    global_model.load_state_dict(i_w)
    G = None

    for r in range(1, R + 1):
        delta = []

        # Create a copy of the current global model's weights
        #G = {k: v.clone().float() for k, v in global_model.state_dict().items()}
        i_w = {k: v.clone().float() for k, v in global_model.state_dict().items()}
        Print("Model's initial weights:", {k: v.float() for k, v in i_w.items()})

        # Loop for selected clients
        for c in range(1, C + 1):
            # Training on the client
            clients_delta = training(i_w, E, r, c)
            delta.append(clients_delta)

        # Average the gradients from clients
        # average_gradients = average_function(delta)
        # update_avg = average_updates(all_clients_updates, data_size)
        average_gradients = average_updates(delta, data_size)  #average_function(delta)

        Print("Weights difference:", average_gradients)

        if G is None:
            G = {key: torch.zeros_like(param).float() for key, param in average_gradients.items()}

        for key in i_w:
            # Accumulate the sum of squares of gradients
            G[key] += average_gradients[key] ** 2

            # Update weights using Adagrad rule
            i_w[key] = i_w[key] - ( lr * average_gradients[key] / (torch.sqrt(G[key]) + tau))


        global_model.load_state_dict(i_w)

        updated_weights = {k: v.clone().float() for k, v in global_model.state_dict().items()}
        Print(f"Updated global model after round {r}:", {k: v.float() for k, v in updated_weights.items()})

        # Save the updated weights
        os.makedirs(os.path.join(dst_folder, "weights"), exist_ok=True)
        torch.save(global_model, f'{dst_folder}/weights/after_round_{r}_weights.pt')

        val_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        val_model.load_state_dict(updated_weights)
        # Perform validation
        validation_results = val_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_{r}", imgsz=512, batch=4, split='val', workers=0, device=0)
        validation_dict[f"round_{r}"] = validation_results

        print("Round", r, "completed")
        # clear_output(wait=False)

#===========================Parameters==============================================================
round_no=30
client_no=4
participating_client=client_no
learning_rate=0.01
batch_size=4
epochs=5
opti='SGD'
# momentum=0.937
# weight_decay=0.0005
data_size=[]

#===========other variables=============================================
validation_dict = {}

fl_a="yaml"

set_up="IID"
# set_up="limited_data"

if set_up=="IID":
    data_size.append(120)
    data_size.append(120)
    data_size.append(120)
    data_size.append(120)
else:
    data_size.append(120)
    data_size.append(120)
    data_size.append(120)
    data_size.append(43)


forname=set_up

dst_folder = f"{fl_a}_{forname}_Fed_adagrad_{learning_rate}_{opti}"
delete_folder(dst_folder)

#===================================loading the saved weight list====================================================
global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
global_model.info()
initial_weights = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
print(len(initial_weights))
Print("Model's initial weights", initial_weights)
# global_model.save('current.pt')


l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
#server validation rounds
validation_results = l_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_0", imgsz=512, batch=4,split='val', workers=0,device=0)
validation_dict["round_0"] = validation_results
print(validation_results)

#=================================================================client_1====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c1.yaml", project=f"{dst_folder}/train/round_0_client_1", imgsz=512, batch=4, split='train',  workers=0,device=0)

#=================================================================client_2====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c2.yaml", project=f"{dst_folder}/train/round_0_client_2", imgsz=512, batch=4, split='train',  workers=0,device=0)


#=================================================================client_3====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c3.yaml", project=f"{dst_folder}/train/round_0_client_3", imgsz=512, batch=4, split='train',  workers=0,device=0)


#=================================================================client_4====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c4.yaml", project=f"{dst_folder}/train/round_0_client_4", imgsz=512, batch=4, split='train',  workers=0,device=0)
# clear_output(wait=False)


federated_learning(initial_weights, client_no, participating_client, round_no, epochs, batch_size)


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


