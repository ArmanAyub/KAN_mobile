# FastKAN Human-vs-NonHuman: Interpretable Binary Vision Classification

[![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![FastKAN](https://img.shields.io/badge/Fast--KAN-Efficient-blue)](https://github.com/ZiyaoLi/fast-kan)

A high-performance binary vision classifier using **FastKAN** (Kolmogorov-Arnold Networks) for human/non-human detection. This project demonstrates KAN's superior interpretability and parameter efficiency compared to traditional MLPs.

## 🚀 Key Features
- **FastKAN Integration:** Near-real-time inference with KAN's unique learnable activation curves.
- **W&B Tracking:** Professional experiment logging (accuracy, loss, gradients).
- **Interpretable Analysis:** A dedicated Jupyter Notebook for visualizing learned KAN functions.
- **Mobile-Ready:** ONNX export pipeline included for edge deployment.

## 🧠 Learning Highlights (What you'll find in the code)
- **MLP vs. KAN:** Understanding fixed node activations vs. learnable edge activations.
- **RBF Basis:** How Fast-KAN uses Radial Basis Functions for speed.
- **Parameter Efficiency:** Comparative study of model sizes.

## 🛠️ Setup & Execution

### 1. Conda Environment (Recommended)
This project uses Conda for easy cross-deployment and dependency management.

```bash
# Create the environment from the .yml file
conda env create -f environment.yml

# Activate the environment
conda activate kan-human-env

# Log in to Weights & Biases
wandb login
```

### 2. Dataset
Download the [Human and Not Human Dataset](https://www.kaggle.com/datasets/aliasgartaksali/human-and-non-human) and place it in the `data/` folder.

### 3. Training & Tracking
```bash
# Train the Baseline (MLP head)
python src/train.py --model baseline --epochs 10

# Train the FastKAN version
python src/train.py --model kan --epochs 10
```

### 4. Interactive Analysis
Open `analysis.ipynb` to compare models and visualize the learned curves.

## 📊 Demo
Launch the Gradio interface for live inference:
```bash
python src/demo.py
```
