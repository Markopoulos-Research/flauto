#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from sklearn.metrics import confusion_matrix
from torchvision.datasets import CIFAR10
from collections import Counter
import random
import copy
import torch.nn.functional as F
import os
import sys
import time
import pickle
import pandas as pd
from torch import Tensor
from typing import Type
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# Set font family for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



subset_dataset = torch.load('/data/ex1/val_dataset.pth',weights_only=False)
remaining_dataset = torch.load('/data/ex1/test_dataset.pth',weights_only=False)

# Create DataLoaders
val_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(remaining_dataset, batch_size=32, shuffle=False)


def accuracy(outp, target):
    """Computes accuracy"""
    with torch.no_grad():
        pred = torch.argmax(outp, dim=1)
        correct = pred.eq(target).float().sum().item()
        return 100.0 * correct / target.size(0)


def Print(string, dictionary):
    first_key = next(iter(dictionary))
    first_value = dictionary[first_key]
    print(f"{string}:{first_key}: {first_value[0][0]}\n")


def forbinus_norm_function(w_i):
    value = 0
    for k in w_i.keys():
        value += torch.linalg.norm(w_i[k])
    return value.item()


def model_deviation_function(w_i, w_f):
    model_deviation = 0
    for k in w_i.keys():
        model_deviation += torch.linalg.norm(w_f[k].to(torch.float) - w_i[k].to(torch.float)) / torch.linalg.norm(w_i[k].to(torch.float))
    #print(model_deviation.item())
    return model_deviation.item()


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # Change output to 10 for CIFAR-10 classes

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.relu5(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


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
        
        #print(len(weights_alpha), len(distances))
        
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


def train(i_weights, epochs, train_loader, le_rate, cli,roun, epoch_flag):
    global opti
    
    local_model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    if opti=="adam":
        optimizer = torch.optim.Adam(local_model.parameters(), lr=le_rate)
    elif opti=="sgd":
        optimizer = torch.optim.SGD(local_model.parameters(), lr=le_rate)
    
    epoch_train_accuracy=0 
    epoch_train_loss=0
    epoch_test_accuracy=0
    epoch_test_loss=0
    epoch_rmd=0

    local_model.load_state_dict(i_weights)

    local_model.train()  # Set the model to training mode

    # initial weights cathing and printing
    initial_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
    #Print("Model's inside the function Initial weights for client",initial_weights)

    # Training loop
    for epoch in range(epochs):
        epoch_flag=epoch_flag+1
        # gradients_this_epoch = {}
        total_samples = 0
        total_loss=0
        correct_samples = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            _, predicted = outputs.max(1)  # Get the index of the maximum value in outputs (predicted class)
            total_samples += labels.size(0)
            correct_samples += predicted.eq(labels).sum().item()
        
        if(total_samples!=0 and len(train_loader)!=0):
            epoch_accuracy = 100 * correct_samples / total_samples
            epoch_loss = total_loss / len(train_loader)
        else:
            epoch_accuracy = 100 * correct_samples / (total_samples+1)
            epoch_loss = total_loss / (len(train_loader)+1)
        print(f"Round {roun}, cleint {cli+1}, epoch {epoch+1}: epoch_accuracy {epoch_accuracy}, epoch_loss {epoch_loss} ")
    
    f_weights = {k: v.clone() for k, v in local_model.state_dict().items()}

    #print(f"\n Round {roun}, cleint {cli}: epoch_accuracy {epoch_accuracy}, epoch_loss {epoch_loss} \n")
    epoch_train_accuracy=epoch_accuracy
    epoch_train_loss=epoch_loss
    epoch_test_accuracy, epoch_test_loss= test(f_weights, test_loader)
    
    
    epoch_rmd=model_deviation_function(initial_weights,f_weights)
    
    #saving data into dataframe
    epoch_data = [epoch_train_accuracy, epoch_train_loss, epoch_test_accuracy, epoch_test_loss, epoch_rmd]
    epoch_results.loc[len(epoch_results)] = epoch_data
    
    return epoch_accuracy,epoch_loss, f_weights, epoch_flag



def test(w,data):
    lmodel = model().to(device)
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task
    #optimizer = torch.optim.SGD(lmodel.parameters(), lr=learning_rate)
    lmodel.load_state_dict(w)
    lmodel.eval()

    #checking the weights
    tw = lmodel.state_dict()
    #Print("Model's before testing the weights in global model",tw)

    # Evaluation phase for test set
    acc_list = []
    loss_list = []

    with torch.no_grad():
        for j, data in enumerate(data, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            out = lmodel(images)
            # Calculate loss
            loss = criterion(out, labels)
            loss_list.append(loss.item())
            #calculate accuracy
            acc = accuracy(out, labels)
            acc_list.append(acc)
    test_loss = np.mean(loss_list)
    test_accuracy = np.mean(acc_list)
    #print("Model's Test accuracy : {:.2f}%".format(test_accuracy))
    return test_accuracy, test_loss


def federated_learning(i_w, data_client, C, P, R, E, learning_rate, b_size):
    
    global total_clients_list, participating_client_list
    
    global_model.load_state_dict(i_w)
    #Print("Model's initial weights", i_w)

    #loop for round
    for r in range(1,R+1):
        round_train_accuracy=0
        round_train_loss=0
        round_test_accuracy=0
        round_test_loss=0
        epoch_flag=0

        models=[]
        weights_alpha = []

        #saving initial weights for spiking model
        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}
        #Print("Model's initial weights", i_w)
        
        #colleting weights and results
        all_final_weights={}
        train_accuracy_list=[]
        train_loss_list=[]
        
        # Randomly select clients
        selected_clients = random.sample(total_clients_list, P)
        participating_client_list.append(selected_clients)

        #loop for client
        for c, data in enumerate(data_client):
            
            if(c in selected_clients):
                
                train_loader = torch.utils.data.DataLoader(data, batch_size=b_size, shuffle=True, drop_last=True)
                
                #train model
                train_accuracy, train_loss, c_f_weights, epoch_flag = train(i_w, E, train_loader, learning_rate, c, r,epoch_flag)

                models.append(c_f_weights)
                weights_alpha.append(1.0) 
            
                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(train_loss)

            else:
                print(f"client {c+1} is not selectecd")

        all_final_weights = smoothed_weiszfeld(models, weights_alpha, i_w)
        round_epoch=(epoch_flag)
        
        #print("Total number of selected clients is", client_counter)
        round_train_loss=sum(train_loss_list)/len(train_loss_list)
        round_train_accuracy=sum(train_accuracy_list)/len(train_accuracy_list)

        print(f"Model's Round: {r}, train accuracy of model: {round_train_accuracy}, train loss of model: {round_train_loss} \n\n")

        round_test_accuracy, round_test_loss=test(all_final_weights, test_loader)
        print(f"Model's Round: {r}, test accuracy of model: {round_test_accuracy}, test loss of model: {round_test_loss} \n\n")

        #model deviation code
        round_rmd=model_deviation_function(i_w, all_final_weights)
        #print("Model deviation values: ", model_deviation)

        #saving data into dataframe
        round_data = [round_train_accuracy, round_train_loss, round_test_accuracy, round_test_loss, round_rmd, round_epoch]
        round_results.loc[len(round_results)] = round_data
            
        global_model.load_state_dict(all_final_weights)
        print("round", r, "completed")


#===========================Parameters==============================================================
client_no=20
participating_client=20
epochs=5
learning_rate=0.01
round_no=30
batch_size=128

# distributions = "non_iid" # 'non_iid'
distributions = "iid" # 'non_iid'

data_class=10

# alpha=0.5
alpha="infinity"

opti="sgd" # or SGD

method="fed_rfa"

# List of clients
total_clients_list = list(range(0, client_no))
# print(total_cleints_list)
participating_client_list=[]

# Define dataframe for round results
round_columns = ['train_accuracy', 'train_loss', 'test_accuracy', 'test_loss', 'rmd', 'epoch']
round_results = pd.DataFrame(columns=round_columns)

# Define dataframe for epoch results
epoch_columns = ['train_accuracy', 'train_loss', 'test_accuracy', 'test_loss', 'rmd']
epoch_results = pd.DataFrame(columns=epoch_columns)

#===================================loading the saved weight list====================================================
global_model = model().to(device)
# initial_weights={k: v.clone() for k, v in global_model.state_dict().items()}
# Save the initial weights
file_path = "/data/ex1/s_cnn.pth"
# torch.save(initial_weights, file_path)
initial_weights=torch.load(file_path,weights_only=True)
Print("Model's initial weights", initial_weights)


# Load client_datasets from a file
if distributions == 'iid':
    with open('/data/ex1/20_client_datasets_IID.pkl', 'rb') as f:
        client_datasets = pickle.load(f)

elif distributions == 'non_iid' and alpha==0.5:
    with open('/data/ex1/20_client_datasets_non_IID_0_5.pkl', 'rb') as f:
        client_datasets = pickle.load(f)
     
print("client_datasets loaded successfully.")


#train accuracy for cleints
round_train_accuracy=0
round_train_loss=0

train_accuracy_list=[]
train_loss_list=[]
for c, data in enumerate(client_datasets):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    train_accuracy, train_loss=test(initial_weights, train_loader)
    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)
round_train_accuracy=(sum(train_accuracy_list)/len(train_accuracy_list))
round_train_loss=(sum(train_loss_list)/len(train_loss_list))


#test accuracy for server
round_test_accuracy=0
round_test_loss=0
test_accuracy,test_loss=test(initial_weights,test_loader)
round_test_accuracy=(test_accuracy)
round_test_loss=(test_loss)

round_rmd=0
round_epoch=0

round_data = [round_train_accuracy, round_train_loss, round_test_accuracy, round_test_loss, round_rmd, round_epoch]
round_results.loc[len(round_results)] = round_data

Print("initial_weights", initial_weights)
print(f' train accuracy: {round_train_accuracy}\n train_loss: {round_train_loss}\n test_accuracy: {round_test_accuracy}\n test_loss: {round_test_loss}')



federated_learning(initial_weights, client_datasets, client_no, participating_client, round_no, epochs, learning_rate, batch_size)



# Define the folder and file name
folder_name = f"{method}_{opti}_{learning_rate}_{participating_client}_{client_no}_{distributions}_{alpha}"  # Folder where the Excel file will be saved
file_name = "round_results.xlsx"


# Check if the folder exists, if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Full path where the Excel file will be saved
file_path = os.path.join(folder_name, file_name)

round_results.to_excel(file_path, index=False)

print("DataFrame successfully written for round results.")



# Define the folder and file name
folder_name =  f"{method}_{opti}_{learning_rate}_{participating_client}_{client_no}_{distributions}_{alpha}"   # Folder where the Excel file will be saved
file_name = "epoch_results.xlsx"

# Check if the folder exists, if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Full path where the Excel file will be saved
file_path = os.path.join(folder_name, file_name)

epoch_results.to_excel(file_path, index=False)

print("DataFrame successfully written for epoch results.")

