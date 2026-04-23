# Edge-Deployed KAN: Deploying a Kolmogorov-Arnold Network Head on an Edge Device

[![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![FastKAN](https://img.shields.io/badge/Fast--KAN-RBF%20Splines-blue)](https://github.com/ZiyaoLi/fast-kan)
[![HF Space](https://img.shields.io/badge/HF%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/armanmayub/KAN-mobile)

This project explores replacing a standard MLP classification head with a **FastKAN** (Kolmogorov-Arnold Network) head on a frozen MobileNetV2 backbone, and deploying it to run in real time on an Android device.

FastKAN replaces each linear layer with learnable univariate functions approximated using RBF spline weights — one function per edge — making the network intrinsically interpretable. The project compares FastKAN against a standard MLP head across accuracy, convergence speed, latency, and interpretability, then deploys the FastKAN model to Android via ONNX Runtime.

> **Note on parameters:** FastKAN edges carry both base weights and RBF spline weights, so the FastKAN head has significantly more parameters than the MLP at this input size (1280 backbone features). The comparison is about **accuracy parity, interpretability, and mobile deployment** — not parameter efficiency.

## Setup

### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate kan-human-env
wandb login
```

### 2. Dataset
Download the [Human and Not Human Dataset](https://www.kaggle.com/datasets/aliasgartaksali/human-and-non-human) and place it in the `data/` folder.

### 3. Training
```bash
# Train the Baseline (MLP head)
python src/train.py --config configs/config_baseline.yaml

# Train the FastKAN head
python src/train.py --config configs/config_kan.yaml
```

### 4. Interactive Analysis
Open `analysis.ipynb` to compare models and visualize the learned curves.

## Results

Both models trained on the same frozen MobileNetV2 backbone (10 epochs, Adam lr=0.001).

| Metric | Baseline (MLP) | FastKAN |
|--------|---------------|---------|
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
- FastKAN matched the baseline within 0.33% test accuracy despite a fundamentally different architecture
- FastKAN converged 4x faster (100% val accuracy at epoch 2 vs epoch 9)
- Both models achieve perfect recall — no human is ever misclassified as non-human
- FastKAN is faster at inference (1.92ms vs 2.71ms) due to efficient RBF computation

## KAN Interpretability

Unlike MLP weights (uninterpretable scalars), FastKAN edges learn visualizable univariate functions. Below are the learned activation curves from the final classification layer:

**Human class (Output 0):**
![KAN curves - Human](results/kan_curves_output_0_human.png)

**Non-Human class (Output 1):**
![KAN curves - Non-Human](results/kan_curves_output_1_non_human.png)

The Non-Human curves show clear **sigmoid-like step functions** — the model learned sharp thresholds on specific hidden features, revealing the decision logic that a standard MLP cannot expose.

## Demo
```bash
python src/demo.py
```

Or try the live web demo on [Hugging Face Spaces](https://huggingface.co/spaces/armanmayub/KAN-mobile).

## ONNX Export

Two variants are available depending on the deployment target:

| Variant | Output | Use case |
|---------|--------|----------|
| Standard | `logits [2]` | HF Space, general inference |
| With hidden | `logits [2]` + `hidden [64]` | Android live visualizer |

```bash
# Standard
python src/onnx_export.py --model-path models/kan_best.pth

# With hidden activations — for Android visualizer
python src/onnx_export.py --model-path models/kan_best.pth \
    --with-hidden --output-path models/kan_model_android.onnx
```

## Android Deployment

The `android/` directory is a complete Android Studio project (minSdk 26) that runs the FastKAN model on live camera video.

**Stack:**
- CameraX 1.3.x — live camera preview and frame analysis
- ONNX Runtime Android 1.17.0 — on-device FastKAN inference
- Custom `KANVisualizerView` — 64 lerp-smoothed bars driven by the FastKAN hidden activations, blue for Human, red for Non-Human

**To build and run:**
1. Open `android/` in Android Studio
2. Let Gradle sync
3. Enable USB Debugging on your Android phone and hit Run

The ONNX model is bundled as an asset and loaded at startup. Inference runs entirely on-device with no network calls.
