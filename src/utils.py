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

def save_kan_curves(model, save_dir="results", grid_range=(-1, 1), num_points=100, num_inputs=12):
    """
    Saves a grid of learned KAN curves (sampled input dims x both output classes)
    as PNG files in save_dir. Useful for GitHub inspection.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    kan_layer = None
    for module in model.modules():
        if type(module).__name__ == 'FastKANLayer':
            kan_layer = module
            break

    if kan_layer is None:
        print("No FastKANLayer found.")
        return

    device = next(model.parameters()).device
    x = torch.linspace(grid_range[0], grid_range[1], num_points).to(device)
    in_dim  = kan_layer.input_dim
    out_dim = kan_layer.base_linear.weight.shape[0]

    # Sample evenly spaced input dims
    step = max(1, in_dim // num_inputs)
    input_indices = list(range(0, in_dim, step))[:num_inputs]

    with torch.no_grad():
        for out_idx in range(out_dim):
            cols = 4
            rows = (len(input_indices) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            axes = axes.flatten()

            for plot_i, in_idx in enumerate(input_indices):
                base_act = kan_layer.base_activation(x)
                w_base   = kan_layer.base_linear.weight[out_idx, in_idx]
                y_base   = w_base * base_act

                input_vec = torch.zeros(num_points, in_dim).to(device)
                input_vec[:, in_idx] = x
                y_total = kan_layer(input_vec)[:, out_idx]

                ax = axes[plot_i]
                ax.plot(x.cpu().numpy(), y_total.cpu().numpy(), color='steelblue', linewidth=2, label='Total')
                ax.plot(x.cpu().numpy(), y_base.cpu().numpy(),  color='gray', linewidth=1, linestyle='--', label='Base')
                ax.set_title(f"In {in_idx} → Out {out_idx}", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

            # Hide unused subplots
            for ax in axes[len(input_indices):]:
                ax.set_visible(False)

            axes[0].legend(fontsize=8)
            class_name = ["Human", "Non-Human"][out_idx]
            fig.suptitle(f"KAN Learned Activation Curves — Output: {class_name}", fontsize=13, fontweight='bold')
            plt.tight_layout()

            path = os.path.join(save_dir, f"kan_curves_output_{out_idx}_{class_name.lower().replace('-', '_')}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {path}")

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
