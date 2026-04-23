import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def plot_kan_curves(model, input_idx=0, output_idx=0, grid_range=(-1, 1), num_points=100):
    """
    Extracts and plots the learned activation curves from a FastKAN layer.
    """
    model.eval()
    kan_layer = None
    
    # Find the FastKAN layer in the model
    for module in model.modules():
        if type(module).__name__ == 'FastKANLayer':
            kan_layer = module
            break
    
    if kan_layer is None:
        print("No FastKANLayer found in the model.")
        return None

    # Create grid of input values
    x = torch.linspace(grid_range[0], grid_range[1], num_points).to(next(model.parameters()).device)
    
    with torch.no_grad():
        in_dim  = kan_layer.input_dim
        out_dim = kan_layer.base_linear.weight.shape[0]
        input_idx  = min(input_idx,  in_dim  - 1)
        output_idx = min(output_idx, out_dim - 1)

        # 1. Base contribution (learned base_activation, default SiLU)
        base_act = kan_layer.base_activation(x)
        w_base   = kan_layer.base_linear.weight[output_idx, input_idx]
        y_base   = w_base * base_act

        # 2. Total layer output via one-hot style input (vary one dim at a time)
        input_vec = torch.zeros(num_points, in_dim).to(x.device)
        input_vec[:, input_idx] = x
        y_total = kan_layer(input_vec)[:, output_idx]
        
    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x.cpu().numpy(), y_total.cpu().numpy(), label='Total Learned Function', linewidth=3, color='blue')
    plt.plot(x.cpu().numpy(), y_base.cpu().numpy(), '--', label='Base (SiLU) Contribution', alpha=0.6, color='gray')
    
    plt.title(f"KAN Learned Curve: Input Dim {input_idx} -> Output Dim {output_idx}")
    plt.xlabel("Input Value")
    plt.ylabel("Activation Output")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return fig

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
