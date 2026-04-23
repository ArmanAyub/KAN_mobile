# Project Summary: FastKAN Human-vs-NonHuman Classifier

This document provides a comprehensive overview of the `KAN_mobile` project for use in AI assistants (like Claude) to quickly regain context.

---

## 🎯 Project Goal
The goal of this project is to build a highly efficient and interpretable **binary vision classifier** that distinguishes between **Humans** and **Non-Humans**. It specifically explores the use of **Kolmogorov-Arnold Networks (KAN)** as a modern alternative to traditional Multi-Layer Perceptrons (MLPs).

## 🧠 Core Technology: Why KAN?
- **FastKAN Integration:** Uses the `fast-kan` library which employs Radial Basis Functions (RBF) for efficient computation.
- **Interpretability:** Unlike MLPs with fixed activation functions, KANs have learnable activation functions on the edges, allowing for better visualization of what the model learns.
- **Efficiency:** KANs often achieve similar or better accuracy than MLPs with fewer parameters.
- **Architecture:** The project uses a **frozen MobileNetV2 backbone** (pre-trained on ImageNet) and replaces the final classification head with either an MLP (Baseline) or a FastKAN (KAN) layer.

## 📁 Project Structure

### 🛠️ Core Scripts (`src/`)
- **`model.py`**: Defines `BaselineClassifier` (MLP head) and `KANClassifier` (FastKAN head).
- **`train.py`**: The main training loop. Handles logging to Weights & Biases, validation, and saving the best models.
- **`dataset.py`**: Data loading logic using `ImageFolder`. Handles resizing, normalization, and basic augmentations.
- **`split_data.py`**: Utility to physically split the raw `data/training_set` into `data/train` and `data/val`.
- **`check_leakage.py`**: (Likely) checks for image overlap between train/val/test sets to ensure valid metrics.
- **`onnx_export.py`**: Exports the trained PyTorch KAN model to ONNX format for mobile/edge deployment.
- **`demo.py`**: A Gradio web interface for interactive inference.
- **`utils.py`**: Contains visualization utilities, such as `plot_kan_curves` to see the learned KAN functions.

### ⚙️ Configuration (`configs/`)
- **`config_baseline.yaml`**: Hyperparameters for training the MLP-based model.
- **`config_kan.yaml`**: Hyperparameters for training the KAN-based model.

### 📊 Data (`data/`)
- **`train/`**, **`val/`**: Folders containing `humans/` and `non-humans/` images.
- **`test_set/`**: Independent test images for final evaluation.
- **`training_set/`**: The original source data before splitting.

### 🚀 Automation
- **`run_sprint.ps1`**: A PowerShell script to run both Baseline and KAN training sessions sequentially.

## 📈 Workflow & Instructions

### 1. Data Preparation
If the data isn't split yet:
```bash
python src/split_data.py
```

### 2. Training
To train the models and compare them:
```bash
# Run both experiments automatically
./run_sprint.ps1

# Or run individually
python src/train.py --config configs/config_baseline.yaml
python src/train.py --config configs/config_kan.yaml
```

### 3. Monitoring
Check the **Weights & Biases (W&B)** dashboard for live plots of accuracy, loss, and the learned KAN curves.

### 4. Export & Deployment
To export the best KAN model:
```bash
python src/onnx_export.py --model-path models/kan_best.pth
```

## 📍 Current State
- **Backbone:** MobileNetV2 is the chosen feature extractor (frozen).
- **Weights:** Only `models/baseline_best.pth` exists so far. 
- **Next Steps:** You have successfully trained the baseline (MLP) model. Your next immediate task was likely to **train the KAN-based model** using `python src/train.py --config configs/config_kan.yaml` to compare its accuracy and parameter efficiency against the baseline. 
- **Goal:** Once the KAN model is trained, you would use the Gradio `demo.py` and the `analysis.ipynb` notebook to visualize the learned curves and evaluate if the model provides more interpretable results for identifying humans.

---

## 🤖 Instructions for Claude
When interacting with this project:
1.  **Always prefer `src/train.py`** for experimentation as it uses the YAML configs.
2.  **Refer to `src/model.py`** to adjust the FastKAN `hidden_dim` or layer structure.
3.  **Check `configs/*.yaml`** before starting a run to ensure `epochs` and `lr` are correct.
4.  **Use `utils.plot_kan_curves`** to generate "explanation" images for the KAN model.
