import os
import subprocess

#****** Demo Run ******

# Path to the specific file you want to run
script_path = os.path.join("experiment_1_image_classification", "fed_avg.py")

# Run fedavg.py
print(f"\nRunning {script_path}...")
subprocess.run(["python", script_path])


#****** To reproduce all results from the paper, simply uncomment the lines below ******

# # Path to the folder containing .py files you want to run
# scripts_folder = "/code/experiment_1_image_classification"

# # Loop through all files in the folder
# for filename in os.listdir(scripts_folder):
#     if filename.endswith(".py"):
#         filepath = os.path.join(scripts_folder, filename)
#         print(f"\nRunning {filename}...")
#         subprocess.run(["python", filepath])
        

# scripts_folder = "/code/experiment_2_object_detection"

# # Loop through all files in the folder
# for filename in os.listdir(scripts_folder):
#    if filename.endswith(".py"):
#        filepath = os.path.join(scripts_folder, filename)
#        print(f"\nRunning {filename}...")
#        subprocess.run(["python", filepath])
