#!/usr/bin/env python
# coding: utf-8


# Librearies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from collections import Counter
import random
import copy
import torch.nn.functional as F
import sys
import time
import pickle
import pandas as pd
import random
import queue
from collections import deque
from torch import Tensor
from typing import Type
from sklearn.metrics import confusion_matrix
import os
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


subset_dataset = torch.load('/data/ex1/val_dataset.pth',weights_only=False)
remaining_dataset = torch.load('/data/ex1/test_dataset.pth',weights_only=False)

# Create DataLoaders
val_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(remaining_dataset, batch_size=32, shuffle=False)



#Print the distribution of dataset
def print_distribution(client_data):
    # Check distribution
    for l, client_data_value in enumerate(client_data):
        print(f"Client {l + 1} data size: {len(client_data_value)}")
        class_counts = {j: 0 for j in range(10)}
        for _, label in client_data_value:
            class_counts[label] += 1
        print(f"Class distribution: {class_counts}")


# necessary functions
def Print(string, dictionary):
    first_key = next(iter(dictionary))
    first_value = dictionary[first_key]
    print(f"{string}:{first_key}: {first_value[0][0]}\n")


def model_deviation_function(w_i, w_f):
    model_deviation = 0
    for k in w_i.keys():
        model_deviation += torch.linalg.norm(w_f[k].to(torch.float) - w_i[k].to(torch.float)) / (torch.linalg.norm(w_i[k].to(torch.float)) +1)
    #print(model_deviation.item())
    return model_deviation.item()
    

def forbinus_norm_function(w_i):
    value = 0
    for k in w_i.keys():
        value += torch.linalg.norm(w_i[k])
    return value.item()

def accuracy(outp, target):
    """Computes accuracy"""
    with torch.no_grad():
        pred = torch.argmax(outp, dim=1)
        correct = pred.eq(target).float().sum().item()
        return 100.0 * correct / target.size(0)


def c_rmd_line_graph_generator(data, x_gg, x_label, y_label, graph_title, filename ):
    x_g = list(range(1, x_gg + 1))

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Plot the data
    ax1.plot(x_g, data, color='red')

    # Labeling the axes
    ax1.set_xlabel(x_label, fontsize=8, labelpad=0)
    ax1.set_ylabel(y_label, fontsize=8, labelpad=0)

    # Setting the range for x and y axes
    ax1.set_ylim(0, 1.6)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax1.set_xticks(range(1, x_gg + 1))

    # Adding grid lines
    grid_color = 'grey'
    ax1.grid(True, axis='x', color=grid_color)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)

    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Show the left and bottom spines in gray
    ax1.tick_params(axis='x', labelsize=7)  # Reduced tick label font size to 7
    ax1.tick_params(axis='y', labelsize=7)  # Reduced tick label font size to 7

    # Adding title
    plt.title(graph_title, fontsize=8)

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PDF file
    save_path = filename + ".pdf"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()

#===batch size graphs
def DY_b_line_graph_generator(data1, data2, x_gg, x_label, y_label1, graph_title, filename):
    x_g = list(range(1, x_gg + 1))

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Add text above the graph
    text_x = 0.41
    text_y = 0.84
    text = 'β update criteria met'
    text_obj = plt.figtext(text_x, text_y, text, ha='center', va='top', fontsize=7,
                           bbox=dict(facecolor=(1, 1, 1, 0.8), edgecolor='none'))

    # Plotting the learning rates
    ax1.set_xlabel(str(x_label), fontsize=8, labelpad=0)
    ax1.set_ylabel(str(y_label1), color='black', fontsize=8, labelpad=0)
    ax1.plot(x_g, data1, color='blue')  # Dashed line for data1
    ax1.tick_params(axis='y', labelcolor='black', labelsize=7)
    ax1.tick_params(axis='x', labelsize=7)

    # Setting y-axis limits and ticks
    ax1.set_ylim(0, 300)
    ax1.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax1.set_xticks(range(1, x_gg + 1))

    # Adding grid lines
    grid_color = 'grey'
    #ax1.grid(True, axis='x', color=grid_color)
    plt.gca().spines['left'].set_color(grid_color)
    plt.gca().spines['bottom'].set_color(grid_color)

    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Get the position of the text center in display coordinates
    bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_data = bbox.transformed(fig.dpi_scale_trans.inverted())

    # Convert the text center coordinates to data coordinates
    text_x_data, text_y_data = 1.5, 130

    # Plotting vertical lines for data2 and markers
    for i in range(len(data2)):
        if data2[i] == 1:
            ax1.axvline(x=x_g[i], color='red', linestyle='--', linewidth=1.5)
            #ax1.plot(x_g[i], data1[i])  # Add marker

    # Adding title
    plt.title(graph_title, fontsize=8)

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PDF file
    save_path = filename + ".png"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()

#===rmd graphs
def d_rmd_line_graph_generator(c_data, s_data, x_gg, x_label, y_label, c_label, s_label, filename):
    x_g = list(range(1, x_gg + 1))

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Plotting the data
    ax1.plot(x_g, c_data, label=str(c_label), color='red',linestyle='--')  # Plot train accuracy
    ax1.plot(x_g, s_data, label=str(s_label), color='blue',linestyle='-')  # Plot train accuracy

    # Labeling the axes
    ax1.set_xlabel(str(x_label), fontsize=8,labelpad=0)
    ax1.set_ylabel(str(y_label), fontsize=8,labelpad=0)

    # Setting the range for x and y axes
    ax1.set_ylim(0, 12)
    ax1.set_yticks([0, 2, 4, 6, 8, 10, 12])
    #ax1.set_xticks(range(0, x_gg + 1,5))
    ax1.set_xticks([ 1, 5, 10, 15, 20, 25, 30])

    # Adding grid lines
    grid_color = 'grey'
    ax1.grid(True, axis='x', color=grid_color)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)

    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Show the left and bottom spines in gray
    ax1.tick_params(axis='x', labelsize=7)  # Reduced tick label font size to 7
    ax1.tick_params(axis='y', labelsize=7)  # Reduced tick label font size to 7

    # Adding legend
    ax1.legend(fontsize=7)
    plt.title('Server rmd & average client rmd vs round', fontsize=8)

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PDF file
    save_path = filename + ".pdf"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()

#====epoch graphs
def epoch_line_graph_generator(data, x_gg, x_label, y_label, filename):
    x_g = list(range(1, x_gg + 1))

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Plot the data
    ax1.plot(x_g, data, color='red')  # Plot train accuracy

    # Labeling the axes
    ax1.set_xlabel(str(x_label), fontsize=8)
    ax1.set_ylabel(str(y_label), fontsize=8)

    ax1.set_ylim(1, 6)
    ax1.set_yticks([1, 2, 3, 4, 5,6])
    ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])

    # Adding grid lines
    grid_color = 'grey'
    ax1.grid(True, axis='x', color=grid_color)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)

    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Set font size for ticks
    ax1.tick_params(axis='x', labelsize=7)  # X-axis tick label font size
    ax1.tick_params(axis='y', labelsize=7)  # Y-axis tick label font size

    # Add title with font size 7
    plt.title('Average epoch per client vs round', fontsize=8)

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PDF file
    save_path = filename + ".pdf"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()

#====batch size grpahs
def a_m_b_line_graph_generator(data, x_gg, x_label, y_label, filename):
    x_g = list(range(1, x_gg + 1))

    # Specify desired xtick positions
    xticks = [1, 5, 10, 15, 20, 25, 30]

    # Ensure x_g includes all specified xtick positions
    # Filter data to match xticks
    x_ticks_with_data = [i for i in xticks if i <= len(data)]
    filtered_data = [data[i-1] for i in x_ticks_with_data]

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Plot the data without markers
    ax1.plot(x_g, data, color='red')  # Plot train accuracy

    # Add markers only at the specified x-tick positions
    #ax1.scatter(x_ticks_with_data, filtered_data, color='tab:blue')  # Plot markers at specified positions

    # Labeling the axes
    ax1.set_xlabel(str(x_label), fontsize=8, labelpad=0)
    ax1.set_ylabel(str(y_label), fontsize=8, labelpad=0)

    ax1.set_ylim(0, 300)
    ax1.set_yticks([0, 30, 60, 90, 120, 150])
    ax1.set_xticks(xticks)  # Set x-ticks to desired positions

    # Adding grid lines
    grid_color = 'grey'
    ax1.grid(True, axis='x', color=grid_color)
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)

    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Show the left and bottom spines in gray
    ax1.tick_params(axis='x', labelsize=7)  # Reduced tick label font size to 7
    ax1.tick_params(axis='y', labelsize=7)  # Reduced tick label font size to 7

    plt.title('Average max β per client vs rounds', fontsize=8)
    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PDF file
    save_path = filename + ".pdf"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()

#===== learning rate tuning graphs
def DY_LR_line_graph_generator(data1, data2, x_gg, x_label, y_label1, filename):
    x_g = list(range(1, x_gg + 1))

    fig, ax1 = plt.subplots(figsize=(2.84, 2.22))

    # Add text above the graph
    text_x = 0.55
    text_y = 0.5
    plt.figtext(text_x, text_y, 'λ update criteria met', ha='center', va='top', fontsize=7,
                bbox=dict(facecolor=(1, 1, 1, .7), edgecolor='none'))

    # Plotting the learning rates
    ax1.set_xlabel(str(x_label), fontsize=8, labelpad=0)
    ax1.set_ylabel(str(y_label1), fontsize=8, labelpad=0)
    ax1.plot(x_g, data1, color='blue')  # Removed label to exclude it from the legend
    #ax1.tick_params(axis='y', labelcolor='black', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)
    ax1.tick_params(axis='x', labelsize=7)

    # Setting y-axis limits and ticks
    ax1.set_ylim(0, 0.006)
    ax1.set_yticks([0.0000,  0.0012, 0.0024, 0.0036, 0.0048, 0.006])
    # ax1.set_yticks([0,  0.005, 0.008, 0.01, 0.015, 0.02])
    ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])

    # Adding grid lines
    grid_color = 'grey'
    ax1.spines['left'].set_color(grid_color)
    ax1.spines['bottom'].set_color(grid_color)
    # Removing the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plotting vertical lines for data2 and markers
    for i in range(len(data2)):
        if data2[i] == 1:
            ax1.axvline(x=x_g[i], color='red', linestyle='--', linewidth=1.5)

    # Add title
    plt.title('λ & λ update criteria vs round', fontsize=8)

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a PNG file
    save_path = filename + ".png"
    plt.savefig(save_path, dpi=1000)
    #plt.show()
    plt.close()


# In[ ]:


#====CNN model
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Convolutional layers
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

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust input size here
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # Add a linear layer for classification

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


def train(i_weights, epochs, data_c, lea_rate,max_LR, nsv, mome, wd, cli,roun, batch_ss, max_b, epoch_count,client_limit,epoch_flag):
    global opti
    rmd_pe=[]
    loss_pe=[]
    batch_list=[]
    
    batch_list_context=[]
    batch_list_context.append(0)
    
    c_limit=client_limit
    rmd_change=0
    
    batch_s=batch_ss
    le_rate=lea_rate
    
    Step_Size_b=(max_b - batch_s) / (client_limit)

    train_loader = torch.utils.data.DataLoader(data_c, batch_size=batch_s, shuffle=True, drop_last=True)
    rmd_pe.append(0)
    aa,ll=test(i_weights,train_loader)
    loss_pe.append(ll)
    # print("epoch zero loss:",ll,"rmd: 0")

    local_model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    if opti=="adam":
        optimizer = torch.optim.Adam(local_model.parameters(), lr=le_rate)
    elif opti=="sgd":
        optimizer = torch.optim.SGD(local_model.parameters(), lr=le_rate)
        
    local_model.load_state_dict(i_weights)

    local_model.train()  # Set the model to training mode

    # initial weights cathing and printing
    initial_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
     
    # Training loop
    for epoch in range(1,epochs+1):
        
        epoch_count=epoch_count+1
        epoch_flag=epoch_flag+1
        # gradients_this_epoch = {}
        total_samples = 0
        total_loss=0
        correct_samples = 0
        
        batch_list.append(batch_s)
        
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


        fff_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
        modev_pe=model_deviation_function(initial_weights,fff_weights)
        
        print(f"Round {roun}, client {cli},  Epoch {epoch }, rmd: {modev_pe} Learning rate: {le_rate}, batch size: {batch_s}, Loss: {epoch_loss},  accuracy: {epoch_accuracy:.2f}%, ")

        rmd_pe.append(modev_pe)
        loss_pe.append(epoch_loss)
        
        if epoch>1:
            rmd_change=abs(rmd_pe[epoch]-rmd_pe[epoch-1])/rmd_pe[epoch-1]
        
        batch_list_context.append( ( (rmd_pe[epoch] <1) and (rmd_change*100) <20 ) or (loss_pe[epoch] >= loss_pe[epoch-1]) )
        # print("batch size change condition: ",((rmd_pe[epoch] <1) and (rmd_change*100) <20 ) or (loss_pe[epoch] >= loss_pe[epoch-1]))
        if( ((rmd_pe[epoch] <1) and (rmd_change*100) <20 ) or (loss_pe[epoch] >= loss_pe[epoch-1]) ):
            c_limit=c_limit-1
            if(c_limit<=0):
                break
            batch_s= round(batch_s + Step_Size_b)
            if batch_s >= max_b:
                batch_s = max_b
            # print("updated batch :",batch_s)
            train_loader = torch.utils.data.DataLoader(data_c, batch_size=batch_s, shuffle=True, drop_last=True)
            
    rmd_pe = rmd_pe[1:]
    c_rmd_line_graph_generator(rmd_pe, len(rmd_pe),
                               "epoch",
                               f"rmd",
                               f'rmd vs epoch (c={cli}, r={roun})',
                               f"client no {cli} rmd at round {roun} accross epoch")

    DY_b_line_graph_generator(batch_list, batch_list_context[1:], len(batch_list),
                              "epoch",
                              f"β",
                              f'β & β update criteria vs epoch (c={cli}, r={roun})',
                              f"client no {cli}, batch and batch change condition at round {roun} accross epoch dual_y_axis_plot_")
    
    max_batch=max(batch_list)
    total_epoch_batch=sum(batch_list)
    avg_epoch_batch=sum(batch_list)/len(batch_list)
    avg_epoch_rmd=sum(rmd_pe)/len(rmd_pe)    

    if nsv==0:
        f_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
    # else:
    #     temp_f_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
    #     f_weights = add_gaussian_noise_to_dict(temp_f_weights, nsv)
        
    modev=model_deviation_function(initial_weights,f_weights)

    return epoch_accuracy,epoch_loss, f_weights, epoch_count, epoch_flag, max_batch, total_epoch_batch, avg_epoch_batch, avg_epoch_rmd  


#test
def test(w,data):
    lmodel = model().to(device)
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task
    #optimizer = optim.SGD(lmodel.parameters(), lr=min_LR)
    lmodel.load_state_dict(w)
    lmodel.eval()

    #checking the weights
    tw = lmodel.state_dict()

    # Evaluation phase for test set
    acc_list = []
    loss_list = []

    with torch.no_grad():
        for j, data in enumerate(test_loader, 0):
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
    return test_accuracy, test_loss


#=== Fl structure
def federated_learning(i_w, c, p, r, e, data_client, min_LR,max_LR, min_b,max_b, global_Step_Size_LR, server_limit, mome, wd, distribution, alpha_v, noise_strength_value, straggler_prob,client_limit):
    counter=0
    global_model.load_state_dict(i_w)

    model_weights_queue = deque(maxlen=2)

    # List of clients
    clients = list(range(1, c+1))
    

    epoch_count=0
    
    #loop for round
    for i in range(1,r+1):

        round_LR.append(min_LR)

        r_time_s = time.time()

        epoch_flag=0

        client_counter=0

        i_w = {k: v.clone() for k, v in global_model.state_dict().items()}

        all_gradients={}
        all_final_weights={}

        train_accuracy_list=[]
        train_loss_list=[]

        # Initialize the dictionary to track non-stragglers
        non_stragglers = {client: 1 for client in clients}

        # Randomly select clients
        selected_clients = random.sample(clients, p)
        #print(f"Round: {i}, Total client: {c}, Participating client {p}, Selected clients: {selected_clients}")

        participating_client_list.append(selected_clients)

        # Check for stragglers and update the non_stragglers dictionary
        straggler_count = 0
        for client in selected_clients:
            if random.uniform(0, 1) < straggler_prob:
                non_stragglers[client] = 0
                straggler_count += 1


        # Ensure at least one non-straggler
        if straggler_count == len(selected_clients):
            # Randomly choose one client to be a non-straggler
            non_straggler_client = random.choice(selected_clients)
            non_stragglers[non_straggler_client] = 1
 
        c_max_batch=0
        c_total_epoch_batch=0
        c_avg_epoch_batch=0
        c_avg_epoch_rmd=0
        
        #loop for client
        for j, data in enumerate(data_client):

            if(j+1 in selected_clients and non_stragglers[j+1]):

                client_counter=client_counter+1

                #train model
                train_accuracy, train_loss, c_f_weights, epoch_count,epoch_flag, max_batch, total_epoch_batch, avg_epoch_batch, avg_epoch_rmd=train(i_w, e, data, min_LR, max_LR, noise_strength_value, mome, wd,j+1,i,min_b,max_b, epoch_count,client_limit,epoch_flag)
                
                c_max_batch= c_max_batch + max_batch 
                c_total_epoch_batch = c_total_epoch_batch + total_epoch_batch 
                c_avg_epoch_batch = c_avg_epoch_batch + avg_epoch_batch 
                c_avg_epoch_rmd = c_avg_epoch_rmd + avg_epoch_rmd

                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(train_loss)

                # Accumulate weights for the selected client
                for param_name, param_grad in c_f_weights.items():
                    if param_name in all_final_weights:
                        all_final_weights[param_name] += param_grad
                    else:
                        all_final_weights[param_name] = param_grad

            else:
                print(f"client {j+1} is not selectecd")
        
        c_max_b_round_list.append(c_max_batch/p)
        c_t_b_round_list.append(c_total_epoch_batch/p)
        c_e_b_round_list.append(c_avg_epoch_batch/p)
        c_e_rmd_round_list.append(c_avg_epoch_rmd/p)
        
        round_epoch.append(epoch_flag)
        
        averaged_train_loss=sum(train_loss_list)/len(train_loss_list)
        averaged_train_accuracy=sum(train_accuracy_list)/len(train_accuracy_list)
        
        for param_name in all_final_weights:
            all_final_weights[param_name] = all_final_weights[param_name].float() / client_counter

        #validation_code
        val_accuracy, val_loss=test(all_final_weights, val_loader)

        #test code
        test_accuracy,test_loss=test( all_final_weights, test_loader)
        #print(f"model's Round: {i}, test accuracy of : {test_accuracy}, test loss of : {test_loss}, train accuracy of : {averaged_train_accuracy} \n\n")

        #model deviation code
        model_deviation=model_deviation_function(i_w,  all_final_weights)
        model_weights_queue.append(all_final_weights)

        print(f"model's Round: {i}, Rmd: {model_deviation}, test accuracy: {test_accuracy}, test loss: {test_loss}, train accuracy: {averaged_train_accuracy} \n\n")

        # if distribution=="iid":
        iid_accuracy.append(averaged_train_accuracy)
        iid_loss.append(averaged_train_loss)
        iid_val_loss.append(val_loss)
        iid_val_accuracy.append(val_accuracy)
        iid_test_loss.append(test_loss)
        iid_test_accuracy.append(test_accuracy)
        iid_model_deviation.append(model_deviation)

        if server_limit > 0:
            round_LR_context.append( (iid_val_loss[i] > iid_val_loss[i-1]) and (min_LR > 0.0001) )
            if (iid_val_loss[i] > iid_val_loss[i-1]) and (min_LR > 0.0001):
                max_LR=min_LR
                min_LR=round(min_LR - global_Step_Size_LR,4)
                min_LR=min_LR
                server_limit=server_limit-1
                global_Step_Size_LR = (max_LR - min_LR) / (2 ** server_limit)
                all_final_weights = model_weights_queue.popleft()
                model_weights_queue.clear()
            elif iid_val_loss[i] <= iid_val_loss[i-1]:
                min_LR=round(min_LR + global_Step_Size_LR,4)
                if min_LR>=max_LR:
                    min_LR=max_LR
                    
        global_model.load_state_dict(all_final_weights)
        print("round: ", i, " completed ", " total epoch: ", epoch_count)


#===========================Parameters model==============================================================
client_no=20
participating_client=20
epochs=5

momentum=0.95
weight_decay=5e-4
round_no=30

global_epoch=(epochs * round_no)
rounds = list(range(0, round_no+1))
distributions = "iid" #or non_iid 
data_class=10
alpha='infinity' #or 0.5 
client_limit=4 #4 is used to generate all the graphs 
server_limit=4 #4 is used to generate al the graphs

opti="adam" # or sgd
# opti="sgd" # or sgd

c_max_b_round_list=[]
c_e_b_round_list=[]
c_t_b_round_list=[]
c_e_rmd_round_list=[]

min_LR=0.0001
min_b=64
max_LR=0.09
max_b=256

global_Step_Size_LR = (max_LR - min_LR) / (( 2 ** server_limit))
participating_client_list=[]
p_c_l=[]
iid_accuracy=[]
iid_loss=[]
iid_test_accuracy=[]
iid_test_loss=[]
iid_val_accuracy=[]
iid_val_loss=[]
iid_model_deviation=[]
iid_model_deviation.append(0)
round_epoch=[]
iid_epoch_test_accuracy=[]
iid_epoch_test_loss=[]
round_LR=[]
round_LR_context=[]
round_LR_context.append(0)

# Create a list to store the row index
row_index = []
# Generate row index data
for round_num in range(1, round_no + 1):
    for epoch in range(0, epochs):
        row_index.append((round_num, epoch))

# Create a DataFrame filled with zeros
epoch_b = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(row_index, names=['r', 'e']), columns=[f"c_{i}" for i in range(1, client_no)])
epoch_b_context = pd.DataFrame('-', index=pd.MultiIndex.from_tuples(row_index, names=['r', 'e']), columns=[f"c_{i}" for i in range(1, client_no)])

noise_strength = 0
straggler_prob_value = 0

roun = [f"{i}" for i in range(round_no+1)]
global_model = model().to(device)

file_path = "/data/ex1/s_cnn.pth"
initial_weights=torch.load(file_path, weights_only=True)
Print("Model's initial weights", initial_weights)


# Load client_datasets from a file
if distributions == 'iid':
    with open('/data/ex1/20_client_datasets_IID.pkl', 'rb') as f:
        clients = pickle.load(f)

elif distributions == 'non_iid' and alpha==0.5:
    with open('/data/ex1/20_client_datasets_non_IID_0_5.pkl', 'rb') as f:
        clients = pickle.load(f)

        
print("client_datasets loaded successfully.")
# print_distribution(clients)



#reound zero
train_accuracy_list=[]
train_loss_list=[]
for j, data in enumerate(clients):
    train_loader = torch.utils.data.DataLoader(data, batch_size=min_b, shuffle=True, drop_last=True)
    train_accuracy, train_loss=test(initial_weights, train_loader)
    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)

iid_accuracy.append(sum(train_accuracy_list)/len(train_accuracy_list))
iid_loss.append(sum(train_loss_list)/len(train_loss_list))


val_accuracy,val_loss=test(initial_weights,val_loader)
iid_val_accuracy.append(val_accuracy)
iid_val_loss.append(val_loss)

test_accuracy,test_loss=test(initial_weights,test_loader)
iid_test_accuracy.append(test_accuracy)
iid_test_loss.append(test_loss)

Print("initial_weights", initial_weights)
print(f' train accuracy: {train_accuracy}\n train_loss: {train_loss}\n test_accuracy: {test_accuracy}\n test_loss: {test_loss}')


federated_learning(initial_weights, client_no, participating_client, round_no, epochs, clients, min_LR, max_LR, min_b, max_b, global_Step_Size_LR, server_limit, momentum, weight_decay, distributions, str(alpha), noise_strength, straggler_prob_value,client_limit)



avg_round_epoch = [item / 20 for item in round_epoch]
epoch_line_graph_generator(avg_round_epoch, len(avg_round_epoch), "round", "epoch count", "------averaged number of epoch run across rounds")

#average max beta
a_m_b_line_graph_generator(c_max_b_round_list, len(c_max_b_round_list), "round", "β", "------------average clients max β across rounds")

#dual rmd
d_rmd_line_graph_generator(c_e_rmd_round_list, iid_model_deviation[1:], len(c_e_rmd_round_list), "round", "rmd", "average client rmd", "server rmd", "-------dual_Y_axis_plot_for_rmd")

#learning rate
DY_LR_line_graph_generator(round_LR, round_LR_context[1:], len(round_LR), "round", "λ", "-----dual_Y_axis_plot_for_lamda")



tag="CNN, c="+str(client_no)+ ", p="+ str(participating_client)+ ", e="+ str(epochs)+ ", r="+ str(round_no) +", lr="+ str(min_LR) + ", bs=" + str(min_b)
roun = [f"{i}" for i in range(round_no+1)]
roun_int = [int(value) for value in roun]
rounds = list(range(0, round_no+1))


roun_int = [int(value) for value in roun]
plt.figure(figsize=(10, 6))
plt.plot(roun_int, iid_model_deviation, label="rmd", marker='o',color='red')  # Plot test accuracy
# Labeling the axes and the plot
plt.xlabel("Round", fontsize=15)
plt.ylabel("rmd", fontsize=15)
plt.legend()
# Set x-axis and y-axis label intervals to a difference of 10
plt.xticks(range(0, max(roun_int) + 1, 5))
# plt.yticks(range(0.0, 0.5))
plt.ylim(0, 0.5)
# Show grid if you prefer
plt.grid(True)
plt.title(tag)
# Show the plot
plt.show()



roun_int = [int(value) for value in roun]
plt.figure(figsize=(10, 6))
plt.plot(roun_int, iid_test_accuracy, label="Test accuracy", marker='o',color='red')  # Plot test accuracy
plt.plot(roun_int, iid_accuracy, label="Train accuracy", marker='x',color='black')  # Plot train accuracy
# Labeling the axes and the plot
plt.xlabel("Round", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend()
# Set x-axis and y-axis label intervals to a difference of 10
plt.xticks(range(0, max(roun_int) + 1, 10))
plt.yticks(range(0, 101, 10))
# Show grid if you prefer
plt.grid(True)
plt.title(tag)
# Show the plot
plt.show()


data_collection = pd.DataFrame({'round': roun_int, 'train_accuracy': iid_accuracy, 'test_accuracy': iid_test_accuracy, 'train_loss': iid_loss, 'test_loss': iid_test_loss, 'rmd': iid_model_deviation, 'val_accuracy': iid_val_accuracy, 'val_loss': iid_val_loss})
# Save DataFrame to CSV file
data_collection.to_csv(f'Method_1__Scratched_Adam_1_.csv', index=False)
# data_collection
print(round_epoch)
x_g= range(1,len(round_epoch)+1)
plt.figure(figsize=(18, 10))
plt.plot(x_g, round_epoch, marker='o',color='red')  # Plot train accuracy
plt.xlabel("Rounds",  fontsize=24)
plt.ylabel("Epoch number",  fontsize=24)
plt.legend()
plt.yticks(range(0, 101, 10))
plt.grid(True)
plt.show()

