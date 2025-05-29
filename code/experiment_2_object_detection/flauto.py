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
# from IPython.display import clear_output
import time
import pickle
import json
from IPython.display import clear_output
import copy
from collections import deque
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# print function
def Print(string, dictionary):
    first_key = next(iter(dictionary))
    first_value = dictionary[first_key]
    print(f"{string}:{first_key}: {first_value[0][0]}\n")

# forbinus norm function
def forbinus_norm_function(w_i):
    value = 0
    for k in w_i.keys():
        value += torch.linalg.norm(w_i[k])
    return value.item()

# model deviation function
def model_deviation_function(m1, m2):

    #model deviation code
    w_i = {key: value.to(device) for key, value in m1.items()}
    w_f = {key: value.to(device) for key, value in m2.items()}

    model_deviation = 0
    for k in w_i.keys():
        model_deviation += torch.linalg.norm(w_f[k].to(torch.float) - w_i[k].to(torch.float)) / (torch.linalg.norm(w_i[k].to(torch.float) +1) )
    return model_deviation.item()


# data path identification
def on_train_epoch_end(trainer):
    """Custom logic for additional metrics logging at the end of each training epoch."""
    global path
    path=trainer.csv

# define function to add data from another DataFrame
def add_data_to_client(client_id, new_data):
    global clients
    if client_id in clients:
        # Append new data to the client's DataFrame
        clients[client_id] = pd.concat([clients[client_id], new_data], ignore_index=True)
    else:
        print(f"Client {client_id} does not exist.")


# deleting run folder for saving spaces
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully!")
    else:
        print(f"Folder '{folder_path}' does not exist.")


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



def training( i_w, E, r, c, l_rate):

    print("\n\n\n=======current learning rate", l_rate, "====\n\n\n\n")
    #declear local model
    global path, count_e, client_limit, min_b, max_b, opti
    local_model=YOLO("/data/ex2/initial_weights.pt").to(device)
    local_model.load_state_dict(i_w)
    print(local_model.info())
    total_val_loss=[]
    total_train_loss=[]
    c_limit=client_limit
    batch_s=min_b

    Step_Size_b=(max_b - batch_s) / (client_limit)

    rmd_pe=[]
    #rmd_pe.append(0)
    flag=0
    le_rate=l_rate
    #local_model.load_state_dict(i_w,strict=False)
    checking_weights = local_model.state_dict()
    Print(f"Client {c} functions weights", checking_weights)

    #epochs
    for e in range(0,E):
        count_e=count_e+1
        local_model.add_callback("on_train_epoch_end", on_train_epoch_end)
        local_model.train(data=f"/data/ex2/{fl_a}/{set_up}/c{c}.yaml", project=f"{dst_folder}/train/round_{r}_client_{c}/{e}", 
                          workers=0, epochs=1, imgsz=512, 
                          split='train', lr0=le_rate, batch=batch_s, optimizer=opti, 
                          val=True, device=0)

        fff_weights = {k: v.clone() for k, v in local_model.state_dict().items()}

        modev_pe=model_deviation_function(i_w,fff_weights)
        rmd_pe.append(modev_pe)

        print("model deviation", modev_pe)

        #collecting results
        df = pd.read_csv(path)
        #print(path)
        #print(df)
        df.columns = df.columns.str.strip()

        t_loss= df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        total_train_loss.append(t_loss.iloc[0])#print(df['train/box_loss'])

        v_loss= df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        total_val_loss.append(v_loss.iloc[0])#print(df['train/box_loss'])

        if e>0:
            rmd_change=abs(rmd_pe[e]-rmd_pe[e-1])/rmd_pe[e-1]
            print("rmd_change", rmd_change)
            if( ((rmd_pe[e] <1) and (rmd_change*100) <20 ) or (total_train_loss[e] >= total_train_loss[e-1]) ):
                c_limit=c_limit-1
                if(c_limit<=0):
                    break
                batch_s= round(batch_s + Step_Size_b)
                if batch_s >= max_b:
                    batch_s = max_b

    #checking initial weights
    Print(f"Client {c} initial weights", i_w)
    #colleting final weights
    client_final_weights = {k: v.clone() for k, v in local_model.state_dict().items()}#local_model.state_dict()
    Print(f"Client {c} final weights",client_final_weights)    
    #clear_output(wait=False)
    #return client_final_weights, total_val_loss, v_loss.iloc[0]
    return client_final_weights, v_loss.iloc[0]


def federated_learning(i_w, C, P, R, E, b_size):

    # declearning global validtion dictionary
    global global_model, min_LR, max_LR, global_Step_Size_LR, server_limit
    global validation_dict
    global dst_folder
    global device

    model_weights_queue = deque(maxlen=2)

    global_model.load_state_dict(i_w)
    c_lr=min_LR
    r_val_list=[]
    r_val_list.append(0)
    #global average_weights
    #loop for round
    for r in range(1,R+1):
        models=[]
        c_val_list=[]
        #global_model = YOLO(f"Fed_Avg/weights/round_{r-1}_weighs").to(device)
        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print("Model's initial weights", i_w)
        # averaged training weights per rounds
        #average_weights = {}
        #loop for clients
        for c in range(1,C+1):

            #training
            clients_weight, val_loss=training(i_w, E, r, c, c_lr)
            models.append(clients_weight)
            c_val_list.append(val_loss)

        all_final_weights=average_weights_function(models)
        r_val_list.append( sum(c_val_list)/len(c_val_list) )

        model_weights_queue.append(all_final_weights)

        #torch.save(average_weights, f'{dst_folder}/weights/round_{r}_weighs.pt')
        os.makedirs(os.path.join(dst_folder, "weights"), exist_ok=True)  # Creates both Fed_Avg and weights if needed
        # Now you can save the weights as usual:
        torch.save(all_final_weights, f'{dst_folder}/weights/after_round_{r}_weighs.pt')

        val_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        val_model.load_state_dict(all_final_weights)
        #global_model = load_model_weights_partial("yolov8n-obb.yaml", average_weights, device)

        #chceking averaged weights
        c_weight= {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print(f"updated global model after round {r}",c_weight)

        #performing round validations
        validation_results = val_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_{r}",imgsz=512, batch=4, split='val', device=0, workers=0)

        #save validation results into dict 
        validation_dict[f"round_{r}"] = validation_results

        if r>1:
            if server_limit > 0:
                #round_LR_context.append( (iid_val_loss[i] > iid_val_loss[i-1]) and (min_LR > 0.0001) )
                if (r_val_list[r] > r_val_list[r-1]) and (min_LR > 0.0001):
                    max_LR=min_LR
                    min_LR=round(min_LR - global_Step_Size_LR,4)
                    min_LR=min_LR
                    server_limit=server_limit-1
                    global_Step_Size_LR = (max_LR - min_LR) / (2 ** server_limit)
                    all_final_weights = model_weights_queue.popleft()
                    model_weights_queue.clear()
                    model_weights_queue.append(all_final_weights)
                    #c_lr=min_LR
                elif r_val_list[r] <= r_val_list[r-1]:
                    min_LR=round(min_LR + global_Step_Size_LR,4)
                    if min_LR>=max_LR:
                        min_LR=max_LR
                c_lr=min_LR

        global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
        global_model.load_state_dict(all_final_weights)

        #chceking averaged weights
        c_weight= {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
        Print(f"updated global model after round {r}",c_weight)

        #Print("updated global model",i_w)

        print("round", r, "completed")
        # clear_output(wait=False)


#===========================Parameters==============================================================
round_no=30
client_no=4
participating_client=client_no
# learning_rate=0.01
batch_size=4
epochs=5
opti='Adam'
count_e=0


min_LR=0.0001
min_b=4
max_LR=0.09
max_b=6


client_limit=4 #4 is used to generate all the graphs 
server_limit=4 #4 is sued to generate al the graphs


global_Step_Size_LR = (max_LR - min_LR) / (( 2 ** server_limit))

#===========other variables=============================================
validation_dict = {}
# Define the destination folder

fl_a="yaml"

set_up="IID"
# set_up="limited_data"

forname=set_up

dst_folder = f"{fl_a}_{forname}_Flauto_{opti}"
# dst_folder = f"Fed_Avg"
delete_folder(dst_folder)


path=""

#client results storage
clients = {f'client_{i}': pd.DataFrame(columns=['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
       'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
       'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
       'lr/pg0', 'lr/pg1', 'lr/pg2']) for i in range(1,10)}

#===================================loading the saved weight list====================================================
global_model = YOLO("/data/ex2/initial_weights.pt").to(device)
global_model.info()
initial_weights = {k: v.clone() for k, v in global_model.state_dict().items()}#global_model.state_dict()
print(len(initial_weights))
Print("Model's initial weights", initial_weights)
# global_model.save('current.pt')


# <h1><b>Round 0</b></h1>

# In[ ]:


l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
#server validation rounds
validation_results = l_model.val(data=f"/data/ex2/{fl_a}/c5.yaml", project=f"{dst_folder}/val/round_0", imgsz=512, batch=4,split='val', device=0,workers=0)
validation_dict["round_0"] = validation_results
# print(validation_results)

#=================================================================client_1====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c1.yaml", project=f"{dst_folder}/train/round_0_client_1", imgsz=512, batch=4, split='train', device=0, workers=0)

#=================================================================client_2====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c2.yaml", project=f"{dst_folder}/train/round_0_client_2", imgsz=512, batch=4, split='train', device=0, workers=0)


#=================================================================client_3====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c3.yaml", project=f"{dst_folder}/train/round_0_client_3", imgsz=512, batch=4, split='train',device=0, workers=0)


#=================================================================client_4====================
l_model = YOLO("/data/ex2/initial_weights.pt").to(device)
l_model.val(data=f"/data/ex2/{fl_a}/{set_up}/c4.yaml", project=f"{dst_folder}/train/round_0_client_4", imgsz=512, batch=4, split='train', device=0, workers=0)
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


# In[ ]:


file_path = os.path.join(save_dir, 'validation_dict.json')

# Load the JSON file
with open(file_path, 'r') as f:
    loaded_dict = json.load(f)

# Print the loaded dictionary
print("Validation dictionary loaded successfully")

