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
        # FastKAN usually computes: y = base_weight * activation(x) + spline_weight * RBF(x)
        # We simulate a single input dimension's contribution
        
        # 1. Base contribution (usually SiLU/Linear)
        # Most FastKAN implementations use SiLU as the base activation
        base_act = nn.functional.silu(x)
        # Extract the specific weight for this input-output pair
        w_base = kan_layer.base_weight[output_idx, input_idx]
        y_base = w_base * base_act
        
        # 2. Spline/RBF contribution
        # This is more complex as it involves multiple basis functions
        # For simplicity in visualization, we can compute the total layer output 
        # for a "one-hot" style input where only one dimension varies.
        
        input_vec = torch.zeros(num_points, kan_layer.input_dim).to(x.device)
        input_vec[:, input_idx] = x
        
        # Get the full output from the layer
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
