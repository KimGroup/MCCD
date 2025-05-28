# 🧠 Multi-Core Circuit Decoder (MCCD)

This repository accompanies the paper:  
**"Learning to decode logical circuits"**  
by Yiqing Zhou, Chao Wan, Yichen Xu, Jin Peng Zhou, Kilian Q. Weinberger, and Eun-Ah Kim.  
📄 arXiv: [arXiv:2504.16999](https://arxiv.org/abs/2504.16999)

We introduce **MCCD**, a modular machine learning-based decoder designed for logical quantum circuits. MCCD handles both single- and entangling logical gates using gate-specific decoder modules. 

---
# System requirements 
## OS requirements 
The code has been tested on the following macOS and Linux systems. 
Linux: Ubuntu 20.04.6 LTS  
macOS: Sequoia 15.5

## GPU compatibility 
The code is compatible with both CPU and CUDA GPU. 

## Python dependencies 
```
numpy==1.21.6
torch==1.13.1
```
# Install MCCD
Before running the scripts (see [Curriculum Training](#curriculum-training-demo) section below), you need to install the `mccd` package.  
The package can be installed with `pip` using the command below. 
```bash
pip install mccd
```
Typical installation time should be within 1 minute on a normal desktop computer. 


# Curriculum Training (Demo)
## Sample dataset
The full dataset was generated using third-party software that has not been open-sourced yet, but the generated dataset is available upon request.
For demonstration purposes, we provide a small dataset stored in the `cached_qec_data_small` folder to help users explore the data structure and experiment with MCCD.

We use a two-step curriculum training process, which can be executed by sequentially running the following two commands.

## Train single-qubit modules
In the first stage, we train single-qubit decoder modules. 
```bash
bash train_1q.sh
```
After running the script, the resulting model will be saved to `trained_models/c3_d3/model_0.pt`. The runtime on a CPU is approximately 100 seconds.

## Train two-qubit modules
In the second stage, we train the two-qubit decoder module. This script should be run after the single-qubit modules have been trained, as described in the [previous subsection](#train-single-qubit-modules). 
```bash
bash train_2q.sh
```
After running the script, the resulting model will be saved to `trained_models/c3_d4/model_0.pt`. The runtime on a CPU is approximately 200 seconds. 

# License 
This project is made available under **MIT License**.
