import torch
import torch.nn as nn
from model import KANClassifier


class KANWithHidden(nn.Module):
    """Wrapper that outputs both final logits and the 64-dim hidden activations
    from the first FastKAN layer. Used for the Android live visualizer."""
    def __init__(self, kan_model):
        super().__init__()
        self.backbone   = kan_model.backbone
        self.kan_layer0 = kan_model.kan_head.layers[0]  # 1280 -> 64
        self.kan_layer1 = kan_model.kan_head.layers[1]  # 64   -> 2

    def forward(self, x):
        features = self.backbone(x)
        hidden   = self.kan_layer0(features)
        output   = self.kan_layer1(hidden)
        return output, hidden


def export_to_onnx(model_path, output_path, with_hidden=False):
    device = torch.device("cpu")
    base_model = KANClassifier()

    try:
        base_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {model_path}")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    base_model.eval()

    if with_hidden:
        model        = KANWithHidden(base_model)
        output_names = ['output', 'hidden']
        dynamic_axes = {
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
            'hidden': {0: 'batch_size'},
        }
    else:
        model        = base_model
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting {'(with hidden activations) ' if with_hidden else ''}to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("Export successful!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Export trained KAN model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard export (for HF Space / general inference):
  python src/onnx_export.py --model-path models/kan_best.pth

  # Export with hidden activations (for Android live visualizer):
  python src/onnx_export.py --model-path models/kan_best.pth --with-hidden \\
      --output-path models/kan_model_android.onnx
        """
    )
    parser.add_argument('--model-path',   type=str, required=True,
                        help='Path to trained .pth weights')
    parser.add_argument('--output-path',  type=str, default='models/kan_model.onnx',
                        help='Output .onnx path (default: models/kan_model.onnx)')
    parser.add_argument('--with-hidden',  action='store_true',
                        help='Also output 64-dim hidden KAN activations (for Android visualizer)')
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.output_path, with_hidden=args.with_hidden)
