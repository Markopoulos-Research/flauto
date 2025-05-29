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


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully!")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def smoothed_weiszfeld(models_dict, weights_alpha, initial_weights_dict, nu=1e-5, R_weiszfeld=10):
    global device

    # Ensure all tensors are on the same device and of float type
    v = {k: v.clone().detach().to(device).float() for k, v in initial_weights_dict.items()}

    for _ in range(int(R_weiszfeld)):
        # Compute distances between the current aggregated weights and each model's weights
        distances = []
        for model in models_dict:
            # Move model weights to the same device as v and ensure they are of float type
            model = {k: m.to(device).float() for k, m in model.items()}
            distance = sum(
                torch.norm(v[k] - model[k]).item()  # Convert to Python number for easy use
                for k in v.keys()
            )
            distances.append(distance)

        distances = torch.tensor(distances, device=device).float()

        print(len(weights_alpha), len(distances))

        # Avoid division by zero
        beta = torch.tensor([alpha_i / max(nu, dist) for alpha_i, dist in zip(weights_alpha, distances)], device=device).float()

        # Compute weighted sum
        weighted_sum = {k: torch.zeros_like(v[k], device=device).float() for k in v.keys()}
        for beta_i, model in zip(beta, models_dict):
            model = {k: m.to(device).float() for k, m in model.items()}  # Ensure model weights are on the same device and of float type
            for k in weighted_sum.keys():
                weighted_sum[k] += beta_i * model[k]

        # Normalize weights
        total_beta = torch.sum(beta)
        for k in weighted_sum.keys():
            weighted_sum[k] /= total_beta

        # Update v with the new aggregated weights
        v = weighted_sum

    return v

def training( i_w, E, r, c, learning_rate):
    #declear local model
    local_model=YOLO("/data/ex2/initial_weights.pt").to(device)
    local_model.load_state_dict(i_w)
    print(local_model.info())
    #local_model.load_state_dict(i_w,strict=False)
    checking_weights = local_model.state_dict()
    Print(f"Client {c} functions weights", checking_weights)

    local_model.train(data=f"/data/ex2/{fl_a}/{set_up}/c{c}.yaml", project=f"{dst_folder}/train/round_{r}_client_{c}", workers=0, epochs=E, imgsz=512, lr0=learning_rate, batch=4, optimizer=opti, val=True, device=0, warmup_epochs=0)

    #checking initial weights
    Print(f"Client {c} initial weights", i_w)
    #colleting final weights
    client_final_weights = {k: v.clone() for k, v in local_model.state_dict().items()}#local_model.state_dict()
    Print(f"Client {c} final weights",client_final_weights)    
    #clear_output(wait=False)
    return client_final_weights

def federated_learning(i_w, C, P, R, E, learning_rate, b_size):
    global global_model
    global validation_dict
    global dst_folder
    global device
    global_model.load_state_dict(i_w)

    for r in range(1,R+1):
        models=[]
        weights_alpha = []
        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print("Model's initial weights", i_w)
        #loop for clients
        for c in range(1,C+1):
            #training
            clients_weight=training(i_w, E, r, c, learning_rate)
            models.append(clients_weight)
            weights_alpha.append(1.0)

        average_weights = smoothed_weiszfeld(models, weights_alpha, i_w)
        os.makedirs(os.path.join(dst_folder, "weights"), exist_ok=True)  # Creates both Fed_Avg and weights if needed
        # Now you can save the weights as usual:
        torch.save(average_weights, f'{dst_folder}/weights/after_round_{r}_weighs.pt')

        global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        global_model.load_state_dict(average_weights)
        val_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        val_model.load_state_dict(average_weights)
        #chceking averaged weights
        c_weight= {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print(f"updated global model after round {r}",c_weight)

        #performing round validations
        validation_results = val_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_{r}", imgsz=512, batch=4, split='val',workers=0)
        #save validation results into dict 
        validation_dict[f"round_{r}"] = validation_results

        print("round", r, "completed")


#===========================Parameters==============================================================
round_no=30
client_no=4
participating_client=client_no
learning_rate=0.01
batch_size=4
epochs=5
opti='SGD'

#===========other variables=============================================
validation_dict = {}

# ===========================================FL type=======
fl_a="yaml"

set_up="IID"
# set_up="limited_data"

forname=set_up

dst_folder = f"{fl_a}_{forname}_Fed_RFA_{learning_rate}_{opti}"
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
validation_results = l_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_0", imgsz=512, batch=4,split='val')
validation_dict["round_0"] = validation_results
# print(validation_results)

#=================================================================client_1====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c1.yaml", project=f"{dst_folder}/train/round_0_client_1", imgsz=512, batch=4, split='train')

#=================================================================client_2====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c2.yaml", project=f"{dst_folder}/train/round_0_client_2", imgsz=512, batch=4, split='train')


#=================================================================client_3====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c3.yaml", project=f"{dst_folder}/train/round_0_client_3", imgsz=512, batch=4, split='train')


#=================================================================client_4====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c4.yaml", project=f"{dst_folder}/train/round_0_client_4", imgsz=512, batch=4, split='train')


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

