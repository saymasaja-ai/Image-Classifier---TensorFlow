import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import load_model
import tensorflow_hub as hub
from PIL import Image

def process_image(image_path):
    """
    Load an image and preprocess it for the model:
    Resize to 224x224, normalize to 0-1, return as NumPy array.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

def predict(image_path, model, class_names, top_k=5):
    """
    Predict the top K classes for an image.
    """
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    probs = model.predict(image)[0]        # Get probabilities
    top_k_indices = probs.argsort()[-top_k:][::-1]

    top_probs = probs[top_k_indices]
    top_labels = [class_names[str(i + 1)] for i in top_k_indices]

    return top_probs, top_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower class using trained model")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="model.h5", help="Path to trained model (.h5)")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="label_map.json", help="Path to JSON mapping file")

    args = parser.parse_args()

    # Load Keras model
    from tensorflow import keras
    model = keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model loaded successfully!")

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    probs, labels = predict(args.image_path, model, class_names, top_k=args.top_k)

    print(f"\nTop {args.top_k} Predictions:")
    for i in range(len(labels)):
        print(f"{i + 1}: {labels[i]} -> {probs[i] * 100:.2f}%")
