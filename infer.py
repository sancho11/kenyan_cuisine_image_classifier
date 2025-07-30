"""
infer.py

Script to perform inference on a single image using a trained KenyanFood model.
It loads the checkpoint, processes the image, and prints the predicted class
and its confidence score.

Usage:
    python infer.py /path/to/checkpoint.pt /path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from common import (
    DataConfiguration,
    TrainingConfiguration,
    KenyanFood_DataModule,
    load_model_for_TransferLearningandFT
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a single‐image inference with a KenyanFood model checkpoint."
    )
    p.add_argument(
        "checkpoint",
        type=str,
        help="Path to your .pt/.ckpt model checkpoint"
    )
    p.add_argument(
        "image",
        type=str,
        help="Path to the input image"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="./",
        help="Base directory for dataset (used to load class names)"
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(384, 384),
        metavar=("H", "W"),
        help="Input image size (height width)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build configs
    data_config = DataConfiguration(
        data_root=args.data_root,
        batch_size=1,
        num_workers=0
    )
    train_config = TrainingConfiguration(
        image_size=tuple(args.image_size)
    )

    # Setup DataModule (to get class names)
    dm = KenyanFood_DataModule(
        batch_size=1,
        num_workers=0,
        image_size=train_config.image_size,
        base_dir=data_config.data_root
    )
    dm.setup()

    # Try to pull class names from the DataModule
    if hasattr(dm, "classes"):
        class_names = dm.classes
    elif hasattr(dm, "val_dataset") and hasattr(dm.val_dataset, "classes"):
        class_names = dm.val_dataset.classes
    else:
        # Fallback to numeric labels
        class_names = [str(i) for i in range(data_config.num_classes)]

    # Load model
    model = load_model_for_TransferLearningandFT(
        dm,
        train_config,
        data_config,
        ckpt_path=args.checkpoint
    )
    model = model.to(device)
    model.eval()

    # Build a transform matching your training preprocessing
    transform = transforms.Compose([
        transforms.Resize(train_config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load & preprocess image
    img = Image.open(args.image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        # if your model returns a dict, extract logits
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        probs = torch.softmax(logits, dim=1)[0]

    # Get top‐1
    confidence, idx = torch.max(probs, dim=0)
    predicted_class = class_names[idx]

    print(f"Prediction: {predicted_class} ({confidence.item() * 100:.2f}%)")


if __name__ == "__main__":
    main()
