# hash:sha256:8985eb054004f4a4a9a0b6e0e81ff65937c8af12b65d2ae07a1ee81f830a5071
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04


ARG DEBIAN_FRONTEND=noninteractive




# Install common data science and machine learning packages
RUN pip install -U --no-cache-dir \
 numpy==1.23.5 \
 pandas==2.0.3 \
 scikit-learn==1.2.2 \
 matplotlib==3.7.2 \
 seaborn==0.13.0\
 openpyxl==3.1.2




# Install PyTorch with CUDA 11.7 support
# We use pip3 here explicitly for consistency, although pip usually defaults to python3's pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117






