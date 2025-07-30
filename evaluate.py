"""
evaluate.py

Script to evaluate a trained model checkpoint on the KenyanFood dataset.
It will generate:
  - A grid of sample predictions (PIL image)
  - A grid of sample mispredictions (PIL image)
  - A confusion matrix plot (PIL image)
  - A submission CSV file

Usage:
    python evaluate.py /path/to/your/checkpoint.pt \
        --output_dir ./eval_results \
        --batch_size 16
"""
import os
import argparse

from pytorch_lightning.loggers import TensorBoardLogger

from common import (
    DataConfiguration,
    TrainingConfiguration,
    KenyanFood_DataModule,
    load_model_for_TransferLearningandFT,
    get_sample_prediction,
    get_sample_misprediction,
    get_confusion_matrix,
    generate_submission_file
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint on the KenyanFood dataset."
    )
    p.add_argument("checkpoint",type=str,help="Path to your .pt/.ckpt model checkpoint")
    p.add_argument("--output_dir",type=str,default="./eval_results",help="Where to save eval images and submission CSV")
    p.add_argument("--data_root",type=str,default="./",help="Base directory for your dataset")
    p.add_argument("--batch_size",type=int,default=16,help="Batch size for DataLoader")
    p.add_argument("--num_workers",type=int,default=4,help="Number of workers for DataLoader")
    p.add_argument("--image_size",type=int,nargs=2,default=(384, 384),metavar=("H", "W"),help="Input image size (height width)")
    p.add_argument("--sample_count",type=int,default=32,help="How many samples to draw for the sample‑prediction gallery")
    p.add_argument("--mispred_count",type=int,default=20,help="How many samples to draw for the misprediction gallery")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Build configurations
    data_config = DataConfiguration(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_config = TrainingConfiguration(
        image_size=tuple(args.image_size)
    )

    # 2. Setup data module
    data_module = KenyanFood_DataModule(
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        image_size=train_config.image_size,
        base_dir=data_config.data_root
    )
    data_module.setup()

    # 3. Logger for any TensorBoard logging
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="eval_logs"
    )

    print(f"→ Loading checkpoint from: {args.checkpoint}")
    model = load_model_for_TransferLearningandFT(
        data_module,
        train_config,
        data_config,
        ckpt_path=args.checkpoint
    )

    # 4. Sample predictions
    print("→ Generating sample predictions...")
    sample_preds = get_sample_prediction(
        model,
        data_module,
        tb_logger,
        samples=args.sample_count,
        epoch=train_config.prev_epochs
    )
    sample_path = os.path.join(args.output_dir, "sample_predictions.png")
    sample_preds.save(sample_path)
    print(f"   saved sample predictions to {sample_path}")

    # 5. Sample mispredictions
    print("→ Generating sample mispredictions...")
    mispreds = get_sample_misprediction(
        model,
        data_module,
        tb_logger,
        samples=args.mispred_count,
        source="val",
        epoch=train_config.prev_epochs
    )
    mispred_path = os.path.join(args.output_dir, "sample_mispredictions.png")
    mispreds.save(mispred_path)
    print(f"   saved sample mispredictions to {mispred_path}")

    # 6. Confusion matrix
    print("→ Generating confusion matrix...")
    cm_image = get_confusion_matrix(
        model,
        data_module,
        tb_logger,
        source="val",
        epoch=train_config.prev_epochs
    )
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_image.save(cm_path)
    print(f"   saved confusion matrix to {cm_path}")

    # 7. Submission CSV
    print("→ Writing submission file...")
    submission_path = os.path.join(args.output_dir, "submission.csv")
    generate_submission_file(
        model,
        data_module,
        submission_path
    )
    print(f"   saved submission CSV to {submission_path}")

    print("✅ Evaluation complete.")

if __name__ == "__main__":
    main()