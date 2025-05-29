# FLAUTO 🎼🎶

## The Method

Method Name: FLAUTO 🎼🎶  
(Stands for [F]ederated [L]earning [AUTO]mated — also "flauto" is the Italian word for “flute.”)

Method Description:  
FLAUTO is a federated learning method that performs automated hyperparameter tuning at both the server and client levels. It dynamically adjusts the global learning rate based on validation loss and locally tunes each client’s batch size and number of epochs using model deviation metrics. This dual-level approach accelerates convergence, reduces communication and energy costs, and eliminates the need for manual hyperparameter initialization. FLAUTO is compatible with various optimizers and aggregation methods and consistently outperforms state-of-the-art baselines across diverse datasets and settings.

---

## The Paper

Paper title: Federated Learning with Automated Dual-Level Hyperparameter Tuning  
Authors: Rakib Ul Haque and Dr. Panagiotis Markopoulos (contact)  
Affiliation: The University of Texas at San Antonio, Machine Learning Optimization Laboratory
Emails: rakibul.haque@utsa.edu, panagiotis.markopoulos@utsa.edu


---

## This Repo

About:  
This repository contains all code and data required to exactly reproduce the results of the two experiments presented in our paper. The code is organized into two folders — one per experiment.

---

## Experiments

### Experiment 1: Image Classification

- Dataset: CIFAR-10  
  - 10 classes, 60,000 images  
  - IID and non-IID settings (Dirichlet α = 0.5)  
  - 20 clients, each with 2,500 training images  
- Model: Custom Convolutional Neural Network (CNN)  
  - 4 convolutional layers with ReLU activation  
  - 2 max pooling layers  
  - 2 dropout layers  
  - 2 fully connected layers  
  - Output: 10 softmax nodes (one per class)

---

### Experiment 2: Object Detection

- Dataset: Customized subset of DOTA v2.0  
  - 4 selected object classes: Harbor, Plane, Swimming Pool, Tennis Court  
  - 4 clients, each with 70 training and 50 validation images  
  - Server uses 50 test images  
  - Tested under both IID and non-IID settings  
- Model: YOLOv11-Nano  
  - Lightweight object detection model  
  - Backbone: C3k2 block for fast inference  
  - Neck: Multi-scale feature fusion using C3k2  
  - Includes SPPF and C2PSA (spatial attention)

---

## Dependencies

Experiment 1:  
os, sys, time, random, copy, pickle, collections, typing, torch, torchvision, matplotlib, numpy, seaborn, pandas

Experiment 2:  
os, time, pickle, json, copy, shutil, pandas, torch, torchvision, ultralytics

---

Code by: rakibul.haque@utsa.edu
