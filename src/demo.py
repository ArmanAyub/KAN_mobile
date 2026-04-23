import os
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import KANClassifier, BaselineClassifier
import numpy as np

# Load the models (assuming they are trained)
# In practice, you'd load the .pth files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type='kan', path=None):
    if model_type == 'kan':
        model = KANClassifier()
    else:
        model = BaselineClassifier()
    
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Global model variables
current_model = None
classes = ['humans', 'non-humans']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    if current_model is None:
        return "Model not loaded. Please train the model first.", None
        
    # Convert PIL Image to Tensor
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = current_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probabilities, 0)
        
    result = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    # Uncertainty check
    if conf < 0.7:
        status = "Uncertain"
    else:
        status = "Confident"
        
    label_text = f"Prediction: {classes[idx]} ({status}, {conf:.2%})"
    
    # Placeholder for KAN curves plot
    # In a real demo, you'd generate a plot using utils.plot_kan_curves and return it as an image
    curve_plot = np.zeros((300, 400, 3), dtype=np.uint8) # Dummy plot
    
    return result, label_text

# Define Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilities"),
        gr.Text(label="Status"),
        # gr.Image(label="KAN Explanation Curves") # Uncomment when plot logic is ready
    ],
    title="FastKAN Human-vs-NonHuman Classifier",
    description="Interpretable Binary Vision Classification with a FastKAN Head."
)

if __name__ == "__main__":
    kan_path = 'models/kan_best.pth'
    baseline_path = 'models/baseline_best.pth'
    if os.path.exists(kan_path):
        current_model = load_model('kan', path=kan_path)
    elif os.path.exists(baseline_path):
        current_model = load_model('baseline', path=baseline_path)
    demo.launch()
