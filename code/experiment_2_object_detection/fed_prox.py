#!/usr/bin/env python
# coding: utf-8

# <h4> This code is part of FLAUTO. It implements FedProx. Date: 01/09/2025 </h4>
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


#averaging the weihgts
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

def training(i_w, E, r, c, learning_rate, mu=0.001):
    # Declare local model
    local_model = YOLO("/data/ex2/initial_weights.pt").to(device)
    local_model.load_state_dict(i_w)
    print(local_model.info())

    # Move initial weights to the same device as the model
    i_w = {key: value.to(device) for key, value in i_w.items()}

    # Train the model
    local_model.train(data=f"/data/ex2/{fl_a}/{set_up}/c{c}.yaml", project=f"{dst_folder}/train/round_{r}_client_{c}",
                      workers=0, epochs=E, imgsz=512, lr0=learning_rate, batch=4, split='train',
                      optimizer=opti, val=True, device=0, warmup_epochs=0)

    # Apply FedProx proximal term after training
    final_weights = local_model.state_dict()
    for name in final_weights.keys():
        # Ensure final_weights are also on the correct device
        final_weights[name] = final_weights[name].to(device)
        # Apply the proximal term: w_new = w - mu * (w - w_t)
        final_weights[name] = final_weights[name] - mu * (final_weights[name] - i_w[name])

    # Check final weights after FedProx adjustment
    Print(f"Client {c} final weights after FedProx adjustment:", final_weights)

    return final_weights


def federated_learning(i_w, C, P, R, E, learning_rate, b_size):
    # declearning global validtion dictionary
    global global_model
    global validation_dict
    global dst_folder
    global device
    global_model.load_state_dict(i_w)
    #global average_weights
    #loop for round
    for r in range(1,R+1):

        models=[]
        #global_model = YOLO(f"Fed_Avg/weights/round_{r-1}_weighs").to(device)
        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print("Model's initial weights", i_w)
        # averaged training weights per rounds
        #average_weights = {}
        #loop for clients
        for c in range(1,C+1):

            #training
            clients_weight=training(i_w, E, r, c, learning_rate)
            models.append(clients_weight)

        average_weights=average_weights_function(models)
        #torch.save(average_weights, f'{dst_folder}/weights/round_{r}_weighs.pt')
        os.makedirs(os.path.join(dst_folder, "weights"), exist_ok=True)  # Creates both Fed_Avg and weights if needed
        # Now you can save the weights as usual:
        torch.save(average_weights, f'{dst_folder}/weights/after_round_{r}_weighs.pt')

        global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        global_model.load_state_dict(average_weights)
        val_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        val_model.load_state_dict(average_weights)
        #global_model = load_model_weights_partial("yolov8n-obb.yaml", average_weights, device)

        #chceking averaged weights
        c_weight= {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print(f"updated global model after round {r}",c_weight)

        #performing round validations
        validation_results = val_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_{r}", imgsz=512, batch=4, split='val',workers=0,device=0)
        #save validation results into dict 
        validation_dict[f"round_{r}"] = validation_results

        print("round", r, "completed")
        #clear_output(wait=False)


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

fl_a="yaml"

set_up="IID"
# set_up="limited_data"

forname=set_up

dst_folder = f"{fl_a}_{forname}_Fed_prox_{learning_rate}_{opti}"
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

