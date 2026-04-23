# FastKAN Human-vs-NonHuman: Interpretable Binary Vision Classification

[![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![FastKAN](https://img.shields.io/badge/Fast--KAN-Interpretable-blue)](https://github.com/ZiyaoLi/fast-kan)

A binary vision classifier using **FastKAN** (Kolmogorov-Arnold Networks) for human/non-human detection. This project compares a standard MLP head against a FastKAN head — both on a frozen MobileNetV2 backbone — with a focus on **interpretability**: KAN learns a unique activation curve per edge that can be visualized and inspected, unlike the fixed activations in a standard MLP.

> **Note on parameters:** KAN edges carry learnable spline weights (RBF basis functions) in addition to base weights, so the KAN head has significantly more parameters than the MLP head at this input size (1280 features). The comparison here is about **accuracy parity and interpretability**, not parameter count.

## 🚀 Key Features
- **Interpretable by design:** Visualize exactly what each KAN edge learned as a univariate function.
- **Honest A/B comparison:** Same frozen backbone, same data — only the classification head differs.
- **W&B Tracking:** Full experiment logging — accuracy, loss, F1, precision, recall, mAP, latency.
- **ONNX export:** Pipeline included for deploying the trained model to edge devices.

## 🧠 What this project explores
- **MLP vs. KAN:** Fixed node activations (MLP) vs. learnable edge activation curves (KAN).
- **RBF Basis:** How FastKAN uses Radial Basis Functions for efficient spline computation.
- **Interpretability in practice:** What do the learned KAN curves actually look like on real vision features?

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
python src/train.py --config configs/config_baseline.yaml

# Train the FastKAN version
python src/train.py --config configs/config_kan.yaml
```

### 4. Interactive Analysis
Open `analysis.ipynb` to compare models and visualize the learned curves.

## 📊 Results

Both models trained on the same frozen MobileNetV2 backbone (10 epochs, Adam lr=0.001).

| Metric | Baseline (MLP) | KAN |
|--------|---------------|-----|
| Trainable params | 2,562 | 741,186 |
| Best val accuracy | 100.00% | 100.00% |
| **Test accuracy** | **99.93%** | 99.60% |
| Test mAP | 1.0000 | 1.0000 |
| Test F1 | 0.9992 | 0.9958 |
| Test Precision | 0.9985 | 0.9917 |
| Test Recall | **1.0000** | **1.0000** |
| Inference latency | 2.71ms | **1.92ms** |
| Epochs to 100% val | 9 | **2** |

**Key findings:**
- KAN matched the baseline within 0.33% test accuracy despite a fundamentally different architecture
- KAN converged 4× faster (100% val accuracy at epoch 2 vs epoch 9)
- Both models achieve perfect recall — no human is ever misclassified as non-human
- KAN is faster at inference (1.92ms vs 2.71ms) due to efficient RBF computation

## 🔍 KAN Interpretability

Unlike MLP weights (uninterpretable scalars), KAN edges learn visualizable univariate functions. Below are the learned activation curves from the final classification layer:

**Human class (Output 0):**
![KAN curves - Human](results/kan_curves_output_0_human.png)

**Non-Human class (Output 1):**
![KAN curves - Non-Human](results/kan_curves_output_1_non_human.png)

The Non-Human curves show clear **sigmoid-like step functions** — the model learned sharp thresholds on specific hidden features, revealing the decision logic that a standard MLP cannot expose.

## 🖥️ Demo
Launch the Gradio interface for live inference:
```bash
python src/demo.py
```

## 📤 ONNX Export
Export the trained KAN model for edge deployment:
```bash
python src/onnx_export.py --model-path models/kan_best.pth
```
