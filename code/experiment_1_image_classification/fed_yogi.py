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
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import torchvision.models as models
from collections import Counter
from torchvision import datasets, transforms
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
            
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            
            total_loss += loss.item()
            optimizer.step()

            _, predicted = outputs.max(1)  # Get the index of the maximum value in outputs (predicted class)
            total_samples += labels.size(0)
            correct_samples += predicted.eq(labels).sum().item()
            
            #print(loss)
            if torch.isnan(loss):
                raise ValueError("Training stopped: NaN detected in loss.")
        
        if(total_samples!=0 and len(train_loader)!=0):
            epoch_accuracy = 100 * correct_samples / total_samples
            epoch_loss = total_loss / len(train_loader)
            #print("=========",epoch_loss, len(train_loader) )
        else:
            epoch_accuracy = 100 * correct_samples / (total_samples+1)
            epoch_loss = total_loss / (len(train_loader)+1)
            #print("=========",epoch_loss, len(train_loader)+1 )
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
    
    model_update = {}
    for key in local_model.state_dict():
        model_update[key] = torch.sub(i_weights[key], f_weights[key])
    
    return epoch_accuracy,epoch_loss, epoch_flag, model_update


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


def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg


def federated_learning(i_w, data_client, C, P, R, E, learning_rate, b_size, beta_1=0.9, beta_2=0.999, tau=1e-3, epsilon=1e-8):
    
    global total_clients_list, participating_client_list, m, v, t
    r_flag=0
    global_model.load_state_dict(i_w)
    t = 0  # Time step for FedYogi

    # loop for round
    for r in range(1, R+1):
        round_train_accuracy = 0
        round_train_loss = 0
        round_test_accuracy = 0
        round_test_loss = 0
        epoch_flag = 0

        # Saving initial weights for spiking model
        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}

        # Collecting weights and results
        data_size = []
        all_clients_updates = []
        train_accuracy_list = []
        train_loss_list = []
        
        # Randomly select clients
        selected_clients = random.sample(total_clients_list, P)
        participating_client_list.append(selected_clients)

        # Loop for client
        for c, data in enumerate(data_client):
            if c in selected_clients:
                train_loader = torch.utils.data.DataLoader(data, batch_size=b_size, shuffle=True, drop_last=False)
                
                # Train model
                train_accuracy, train_loss, epoch_flag, model_update = train(i_w, E, train_loader, learning_rate, c, r, epoch_flag)

                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(train_loss)
                
                all_clients_updates.append(model_update)
                data_size.append(len(train_loader))
            else:
                print(f"Client {c+1} is not selected")
        
        round_epoch = epoch_flag
        round_train_loss = sum(train_loss_list) / len(train_loss_list)
        round_train_accuracy = sum(train_accuracy_list) / len(train_accuracy_list)
        print(f"Model's Round: {r}, train accuracy of model: {round_train_accuracy}, train loss of model: {round_train_loss} \n\n")

        # Aggregate the updates using a weighted average
        update_avg = average_updates(all_clients_updates, data_size)

        # Initialize moment vectors if not done
        if m is None:
            m = {key: torch.zeros_like(param) for key, param in update_avg.items()}
            v = {key: torch.zeros_like(param) for key, param in update_avg.items()}

        # Yogi update rule for each parameter
        t += 1
        for key in i_w:
            # Update biased first moment estimate
            m[key] = beta_1 * m[key] + (1 - beta_1) * update_avg[key]
            
            # Update biased second raw moment estimate (Yogi adjustment)
            v[key] = v[key] - (1 - beta_2) * (update_avg[key] ** 2) * torch.sign(v[key] - update_avg[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m[key] / (1 - beta_1 ** t)
            
            # Compute bias-corrected second moment estimate (with Yogi adjustment)
            v_hat = v[key] / (1 - beta_2 ** t)
            
            # Update weights using Yogi rule
            i_w[key] -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

        # Test the model on global test set
        round_test_accuracy, round_test_loss = test(i_w, test_loader)
        print(f"Model's Round: {r}, test accuracy of model: {round_test_accuracy}, test loss of model: {round_test_loss} \n\n")
        
        round_val_accuracy, round_val_loss=test(i_w, val_loader)
        print(f"Model's Round: {r}, val accuracy of model: {round_val_accuracy}, val loss of model: {round_val_loss} \n\n")
        
        list_accuracy.append(round_train_accuracy)
        list_loss.append(round_train_loss)
        list_val_accuracy.append(round_val_accuracy)
        list_val_loss.append(round_val_loss)
        list_test_accuracy.append(round_test_accuracy)
        list_test_loss.append(round_test_loss)
        
        
        # Model deviation calculation
        round_rmd = model_deviation_function(i_w, global_model.state_dict())
        
        # Save data into dataframe
        round_data = [round_train_accuracy, round_train_loss, round_test_accuracy, round_test_loss, round_rmd, round_epoch, round_val_accuracy, round_val_loss]
        round_results.loc[len(round_results)] = round_data
        
        print((list_val_loss[r] > list_val_loss[r-1]) and (list_loss[r] > list_loss[r-1]))

        # Load the updated weights into the global model
        global_model.load_state_dict(i_w)
        print("Round", r, "completed")


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

opti="sgd"
# opti="adam"


v=None
m=None
t=0

method="fed_yogi"

# List of clients
total_clients_list = list(range(0, client_no))
# print(total_cleints_list)
participating_client_list=[]


list_accuracy=[]
list_loss=[]
list_test_accuracy=[]
list_test_loss=[]
list_val_accuracy=[]
list_val_loss=[]

# Define dataframe for round results
round_columns = ['train_accuracy', 'train_loss', 'test_accuracy', 'test_loss', 'rmd', 'epoch', 'val_accuracy', 'val_loss']
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


list_accuracy.append(round_train_accuracy)
list_loss.append(round_train_loss)

#test accuracy for server
round_test_accuracy=0
round_test_loss=0
test_accuracy,test_loss=test(initial_weights,test_loader)
val_accuracy,val_loss=test(initial_weights,val_loader)

print(f"{test_accuracy}, {test_loss}, {val_accuracy}, {val_loss}  ")
round_test_accuracy=(test_accuracy)
round_test_loss=(test_loss)

round_rmd=0
round_epoch=0

round_data = [round_train_accuracy, round_train_loss, round_test_accuracy, round_test_loss, round_rmd, round_epoch, val_accuracy,val_loss]
round_results.loc[len(round_results)] = round_data

list_val_accuracy.append(val_accuracy)
list_val_loss.append(val_loss)
list_test_accuracy.append(test_accuracy)
list_test_loss.append(test_loss)


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

