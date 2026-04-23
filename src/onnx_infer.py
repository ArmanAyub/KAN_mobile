import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = ['humans', 'non-humans']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)          # HWC -> CHW
    img = np.expand_dims(img, axis=0)     # add batch dim
    return img

def run(model_path, image_path):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    img = preprocess(image_path)
    logits = session.run(None, {input_name: img})[0][0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    pred = int(np.argmax(probs))
    print(f"Prediction : {CLASSES[pred]}")
    print(f"Confidence : {probs[pred]*100:.2f}%")
    print(f"  humans   : {probs[0]*100:.2f}%")
    print(f"  non-humans: {probs[1]*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/kan_model.onnx')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    run(args.model, args.image)
