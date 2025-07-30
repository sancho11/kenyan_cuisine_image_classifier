"""
train.py

Script to run model training using configurations from common.py.
Expected workflow:
- Parse CLI arguments
- Initialize DataConfiguration and TrainingConfiguration
- Set random seeds and cudnn flags
- Prepare data_module
- Instantiate ModelCheckpoint and TensorBoardLogger
- Load or initialize model
- Run training/validation
"""
import argparse
import random
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from common import (
    DataConfiguration,
    TrainingConfiguration,
    training_validation,
    setup_system
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    # Data configuration
    parser.add_argument("--data-root", type=str, default="./", help="Base directory for dataset files")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--num-classes", type=int, default=13, help="Number of target classes")
    # Training configuration
    parser.add_argument("--seed", type=int, default=21, help="Random seed")
    parser.add_argument("--cudnn-benchmark-enabled", dest="cudnn_benchmark_enabled", action="store_true", help="Enable cudnn benchmark")
    parser.add_argument("--cudnn-benchmark-disabled", dest="cudnn_benchmark_enabled", action="store_false", help="Disable cudnn benchmark")
    parser.set_defaults(cudnn_benchmark_enabled=True)
    parser.add_argument("--cudnn-deterministic", dest="cudnn_deterministic", action="store_true", help="Enable cudnn deterministic")
    parser.add_argument("--cudnn-nondeterministic", dest="cudnn_deterministic", action="store_false", help="Disable cudnn deterministic")
    parser.set_defaults(cudnn_deterministic=True)
    parser.add_argument("--model-name", type=str, default="effnetv2s", help="Model architecture to use")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Pretrained weights spec")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--data-augmentation", dest="data_augmentation", action="store_true", help="Enable data augmentation")
    parser.set_defaults(data_augmentation=False)
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--fine-tune-start", type=int, default=99, help="Epoch to start fine-tuning")
    parser.add_argument("--half-precision", dest="precision", action="store_true", help="Enable half precision training")
    parser.set_defaults(precision=True)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience in epochs")
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(384, 384),
        metavar=("HEIGHT", "WIDTH"), help="Input image size"
    )
    # Logging / checkpoint directories
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path of a pretrained model checkpoint")
    parser.add_argument("--tb-save-dir", type=str, default="tb_logs", help="TensorBoard save directory")
    parser.add_argument("--tb-name", type=str, default="training", help="TensorBoard experiment name")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize configurations
    data_config = DataConfiguration(
        data_root=args.data_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )
    train_config = TrainingConfiguration(
        seed=args.seed,
        cudnn_benchmark_enabled=args.cudnn_benchmark_enabled,
        cudnn_deterministic=args.cudnn_deterministic,
        model_name=args.model_name,
        weights=args.weights,
        epochs=args.epochs,
        data_augmentation=args.data_augmentation,
        learning_rate=args.learning_rate,
        fine_tune_start=args.fine_tune_start,
        precision=args.precision,
        patience=args.patience,
        image_size=tuple(args.image_size),
    )

    # Set random seeds and cudnn flags
    setup_system(train_config)

    # Setup logging
    tb_logger = TensorBoardLogger(
        save_dir=args.tb_save_dir,
        name=args.tb_name,
        log_graph=True,
        version=None,
    )

    # Run training + validation
    model, data_module, model_ckpt = training_validation(
        tb_logger,
        train_config,
        data_config,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
