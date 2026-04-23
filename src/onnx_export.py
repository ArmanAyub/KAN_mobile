import torch
import onnx
from model import KANClassifier

def export_to_onnx(model_path, output_path="models/kan_model.onnx"):
    """
    Exports the trained KAN model to ONNX format.
    """
    device = torch.device("cpu") # Export on CPU for compatibility
    model = KANClassifier()
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export trained KAN model to ONNX")
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--output-path', type=str, default='models/kan_model.onnx', help='Output path for ONNX file')
    
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.output_path)
