import os, sys
from dataclasses import dataclass
from typing import Tuple, List
import warnings
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger

from torchvision import models, datasets, transforms
from torchvision.transforms.functional import to_pil_image

from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
import seaborn as sns

import io
from PIL import Image
from tqdm.auto import tqdm
import timm 

# ---------------------- Transforms & Data Loading ---------------------- #
def denormalize(tensors):
    """
    Revert image normalization to [0.0, 1.0] range for visualization.

    Args:
        tensors (Tensor): Batch of image tensors of shape [B, C, H, W],
                          normalized by ImageNet mean and std.

    Returns:
        Tensor: Denormalized tensors on CPU clamped to [0, 1].
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    
    tensors = tensors.clone()
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors.cpu(), 0.0, 1.0)

class CSVDataset(Dataset):
    """
    PyTorch Dataset for loading images and labels from a CSV file.

    Args:
        images_dir (str): Directory containing the image files.
        csv_file (str): Path to a CSV with 'id' and optional 'class' columns.
        transform_common (transforms.Compose): Transforms applied to all images.
        transform_aug (transforms.Compose, optional): Augmentation transforms
            applied when data augmentation is enabled.
    """
    def __init__(
        self,
        images_dir: str,
        csv_file: str,
        transform_common: transforms.Compose,
        transform_aug: transforms.Compose = None
    ):
        # Base attributes
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_file, dtype=str)
        self.transform_common = transform_common
        self.transform_aug = transform_aug
        self.data_aug : bool = False

        # Build class-to-index mapping if labels exist
        if 'class' in self.df.columns:
            self.classes = sorted(self.df['class'].unique())
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
            self.labels = self.df['class'].map(self.class2idx).tolist()
        else:
            self.classes = []
            self.class2idx = {}
            self.labels = [None] * len(self.df)

        # Image IDs (filenames without extension)
        self.ids = self.df['id'].tolist()

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.ids)

    def use_data_augmentation(self, on : bool = True):
        """
        Enable or disable augmentation transforms.

        Args:
            on (bool): If True, apply augmentations in __getitem__.
        """
        self.data_aug = on
        

    def __getitem__(self, idx):
        """
        Load an image and its label (if available).

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, label) if labels exist;
                   otherwise (image_tensor, image_id).
        """
        img_id = self.ids[idx]
        # asumimos extensión .jpg; si hay varias, podrías hacer glob
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert('RGB')

        # Apply augmentations if enabled
        if self.transform_aug and self.data_aug:
            img = self.transform_aug(img)
        else:
            img = self.transform_common(img) 

        label = self.labels[idx]
        if label is not None:
            return img, label
            
        return img, img_id

class KenyanFood_DataModule(pl.LightningDataModule):
    """
    LightningDataModule for Kenyan Cuisine image classification.

    Args:
        batch_size (int): Samples per batch.
        num_workers (int): DataLoader worker count.
        image_size (tuple): (height, width) for resizing.
        test_csv (str): Path to test CSV file.
        train_csv (str, optional): Path to train CSV; splits if None.
        base_dir (str): Base directory for dataset files.
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        image_size: tuple = (224, 224),
        test_csv: str = "./dataset/test.csv",
        train_csv: str = None,
        base_dir: str = "./"
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.set_data_augment = False
        self.base_dir = base_dir
        
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.val_csv = None
        self.image_size = (224,224)

        self.mean = None
        self.std  = None
        self.common_transforms=None
        self.aug_transforms=None

        self.train_dataset=None
        self.val_dataset = None
        self.test_dataset =None

    def get_mean_and_std(self):
        """
        Set dataset normalization parameters to ImageNet statistics.
        """
        self.mean = [0.485, 0.456, 0.406]
        self.std =  [0.229, 0.224, 0.225]
    
    def prepare_transforms(self):
        """
        Create common and augmentation transform pipelines.
        """
        min_side = min(self.image_size)
        if self.mean == None or self.std == None:
            self.get_mean_and_std()
        
        preprocess = transforms.Compose(
            [transforms.Resize(int(min_side*1.02)), transforms.CenterCrop(min_side), transforms.ToTensor()]
        )

        self.common_transforms = transforms.Compose(
            [preprocess, transforms.Normalize(self.mean, self.std)]
        )

        self.aug_transforms = transforms.Compose(
            [
                transforms.Resize(int(min_side*1.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=(0, 359), translate=(0.1, 0.2), scale=(0.55, 1.05)),
                transforms.ColorJitter(
                    brightness=0.1, hue=0.15
                ),
                transforms.RandomEqualize(),
                transforms.CenterCrop(min_side),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )

    def prepare_data(self):
        """
        Download/extract data archive and split train/val if needed.
        """
        base_dir = self.base_dir
        print("Preparing KenyanFood dataset")
        filename = os.path.join(base_dir, r"dataset/opencv-pytorch-project-2-classification-round-3.zip")
        root = base_dir

        if os.path.exists(filename):
           print("Dataset zip found.")
        else:
           print("Dataset needs to be downloaded from kaggle first, please download the dataset and place it at: "+str(filename))
        
        with zipfile.ZipFile(filename, "r") as f:

            data_extracted = os.path.join(base_dir, "dataset/images/")
            self.data_root = os.path.join(base_dir, "dataset/")

            if not os.path.isdir(data_extracted):
                # extract the zipfile contents
                extract_dir = os.path.join(base_dir, "dataset/")
                f.extractall(extract_dir)

        #print("Preparation completed.")
        self.images_dir = os.path.join(base_dir, "dataset/images/images/")
        base_train_csv       =  os.path.join(base_dir, "dataset/train.csv")
        val_file        =  os.path.join(base_dir, "dataset/autogen_valset.csv")
        train_file      =  os.path.join(base_dir, "dataset/autogen_trainset.csv")

        if not self.train_csv:
            if not os.path.exists(train_file) or not os.path.exists(val_file):
                #Devide the train_csv in two files with randomly 80% on a train file and the other 20% in a validation file.
                # Read original train CSV 
                df = pd.read_csv(base_train_csv, dtype=str)
                # Shuffle and separate 80/20
                train_df = df.sample(frac=0.8, random_state=42)
                val_df   = df.drop(train_df.index)
                # Guardamos los splits
                train_df.to_csv(train_file, index=False)
                val_df.to_csv(val_file,   index=False)
                print(f"Split created: {len(train_df)} train / {len(val_df)} val")
            
            self.train_csv = train_file

        if not self.val_csv:
            self.val_csv=val_file
        print("Preparation completed.")
        

    def setup(self, stage=None):
        """
        Initialize datasets for fitting and testing stages.

        Args:
            stage (str, optional): One of 'fit', 'test', or None.
        """
        if not self.common_transforms:
            self.prepare_transforms()
        if not self.train_csv:
            self.prepare_data()
        
        # Fit stage: definimos train y val
        if stage in (None, "fit") and self.train_csv:
            self.train_dataset = CSVDataset(
                images_dir=self.images_dir,
                csv_file=self.train_csv,
                transform_common=self.common_transforms,
                transform_aug=self.aug_transforms
            )
            self.val_dataset = CSVDataset(
                images_dir=self.images_dir,
                csv_file=self.val_csv,
                transform_common=self.common_transforms
            )

        # Test stage: definimos test (si se provee)
        if stage in (None,"test") and self.test_csv:
            self.test_dataset = CSVDataset(
                images_dir=self.images_dir,
                csv_file=self.test_csv,
                transform_common=self.common_transforms
            )
        
    def set_data_augmentation(self, on: bool = False):
        """
        Toggle data augmentation for training.

        Args:
            on (bool): Enable augmentation if True.
        """
        self.set_data_augment = on

    def train_dataloader(self):
        """
        Return DataLoader for the training dataset.
        """
        if self.train_dataset is None:
            return None
        
        self.train_dataset.use_data_augmentation(self.set_data_augment)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Return DataLoader for the validation dataset.
        """
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Return DataLoader for the test dataset.
        """
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

# ---------------------- Configuration Dataclasses ---------------------- #
@dataclass(frozen=True)
class DataConfiguration:
    """
    Configuration for dataset and DataLoader parameters.

    Attributes:
        data_root (str): Base directory for dataset files.
        num_workers (int): Number of workers for DataLoader.
        batch_size (int): Batch size for training and evaluation.
        num_classes (int): Number of target classes.
    """
    data_root: str = r"./"
    num_workers: int = 11
    batch_size: int = 16
    num_classes: int = 13

@dataclass
class TrainingConfiguration:
    """
    Configuration for the training process.

    Attributes:
        model_name (str): Identifier of the model architecture to use.
        weights (str): Pretrained weights specification (e.g., "DEFAULT").
        epochs (int): Total number of training epochs.
        prev_epochs (int): Number of epochs already completed (for resuming).
        data_augmentation (bool): Flag to toggle data augmentation.
        learning_rate (float): Initial learning rate for optimizer or scheduler.
        fine_tune_start (int): Epoch after which to unfreeze layers for fine-tuning.
        precision (bool): Bool true or false for enabling half presition.
        patience (int): Early stopping patience in epochs.
        load_from (str): Checkpoint to load from ("last", "best", or file path).
        image_size (tuple): Input image size as (height, width).
    """
    seed: int = 21
    cudnn_benchmark_enabled: bool = True
    cudnn_deterministic: bool = True
    model_name: str = "effnetv2s"
    weights: str = "DEFAULT"
    epochs: int = 20
    prev_epochs: int = 0
    data_augmentation: bool = False
    learning_rate: float = 1e-2
    fine_tune_start: int = 99
    precision: bool = True
    patience: int = 30
    load_from: str = "last"
    image_size: tuple = (384,384)

# ---------------------- System Setup & Persistence ---------------------- #
def setup_system(config: TrainingConfiguration):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
    if config.precision:
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")

# ---------------------- Utility Functions for Evaluation ---------------------- #
def get_confusion_matrix(
    model: torch.nn.Module,
    data_module,
    tb_logger,
    source: str = "val",
    epoch: int = 1
) -> Image.Image:
    """
    Compute and plot the confusion matrix for a given stage, then return it
    as a PIL Image.

    Args:
        model (torch.nn.Module): Trained model for inference.
        data_module (LightningDataModule): Data module with data loaders.
        tb_logger (TensorBoardLogger): Logger for TensorBoard figures.
        source (str): One of 'train' or 'val' to select dataset.
        epoch (int): Current epoch number for logging.

    Returns:
        PIL.Image.Image: The confusion-matrix plot as an image.
    """
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Prepare data loader
    data_module.setup(stage='fit')
    if source == "train":
        data_module.set_data_augmentation(False)
        dataloader = data_module.train_dataloader()
    else:
        dataloader = data_module.val_dataloader()

    # Run inference
    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds  = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().tolist())

    # Build confusion matrix
    class_names = data_module.val_dataset.classes
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)

    # Plot to a new Figure
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()

    # Log to TensorBoard (still logs the figure)
    tb_logger.experiment.add_figure(f"ConfusionMatrix/{source}", fig, global_step=epoch)

    # Convert the Matplotlib figure to a PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Clean up
    plt.close(fig)
    buf.close()

    return img

def get_sample_misprediction(
    model: torch.nn.Module,
    data_module,
    tb_logger,
    samples: int = 32,
    source: str = "val",
    epoch: int = 1
) -> Image.Image:
    """
    Display sample mispredictions from the model.

    Args:
        model (torch.nn.Module): Trained model for inference.
        data_module (LightningDataModule): Provides DataLoaders.
        tb_logger (TensorBoardLogger): Logger for TensorBoard figures.
        samples (int): Maximum number of mispredicted samples to display.
        source (str): One of 'train' or 'val' to select dataset.
        epoch (int): Current epoch number for logging.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    # Setup and select loader
    data_module.setup(source)
    if source == "val":
        loader = data_module.val_dataloader()
    elif source == "train":
        data_module.train_dataset.use_data_augmentation(False)
        loader = data_module.train_dataloader()
    else:
        raise ValueError(f"Invalid Stage: {source!r}")


    mis_imgs, mis_true, mis_pred, mis_prob = [], [], [], []

    # Collect mispredictions
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # Look for wrong predictions
            mis_predicted = preds != targets
            if mis_predicted.any():
                for inp, t, p, pr in zip(inputs[mis_predicted],
                                         targets[mis_predicted],
                                         preds[mis_predicted],
                                         probs[mis_predicted]):
                    mis_imgs.append(inp.cpu())
                    mis_true.append(t.item())
                    mis_pred.append(p.item())
                    mis_prob.append(pr[p].item())
                    if len(mis_imgs) >= samples:
                        break
            if len(mis_imgs) >= samples:
                break

    n = len(mis_imgs)
    if n == 0:
        print("Not mispredictions found.")
        return
    # Plot grid of mispredictions
    cols = 4
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        tensor = mis_imgs[i].unsqueeze(0)
        tensor_image = denormalize(tensor)[0]
        img = transforms.functional.to_pil_image(tensor_image)
        ax.imshow(img)
        true_cls = loader.dataset.classes[mis_true[i]]
        pred_cls = loader.dataset.classes[mis_pred[i]]
        ax.set_title(f"P:{pred_cls} ({mis_prob[i]:.2f})\nT:{true_cls}")
        ax.axis("off")
    plt.tight_layout()

    tb_logger.experiment.add_figure(f"Mispredictions_Samples/{source}", fig, global_step=epoch)
    
    # Convert the Matplotlib figure to a PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Clean up
    plt.close(fig)
    buf.close()

    return img

def get_sample_prediction(
    model,
    data_module,
    tb_logger,
    samples: int = 15,
    epoch: int = 1
) -> Image.Image:
    """
    Display and log a grid of random sample predictions on the validation set.

    Args:
        model (LightningModule): Trained model used for inference.
        data_module (LightningDataModule): Provides the validation DataLoader.
        tb_logger (TensorBoardLogger): Logger for saving figures.
        samples (int): Number of random samples to display (default=15).
        epoch (int): Epoch number for logging (default=1).
    """
    # Ensure reproducibility
    random.seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare model
    model.freeze()
    model.eval()
    model.to(device)

    # Load validation data
    source="val"
    val_loader = data_module.val_dataloader()
    validation_dataset   = val_loader.dataset

    idx_to_class = {v: k for k, v in validation_dataset.class2idx.items()}

    imgs  = []
    preds = []
    probs = []
    trues = []

    for batch_imgs, batch_labels in val_loader:
        with torch.no_grad():
            output = model(batch_imgs.to(device))
        # get probability score using softmax
        prob = F.softmax(output, dim=1)

        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]

        # get the index of the max probability
        pred_index = prob.data.max(dim=1)[1]
        # pass the loaded model
        pred = pred_index.cpu().tolist()
        prob = pred_prob.cpu().tolist()
        true = batch_labels.cpu().tolist()

        imgs.extend([np.asarray(to_pil_image(image)) for image in denormalize(batch_imgs)])

        preds.extend(pred)
        probs.extend(prob)
        trues.extend(true)
        if len(trues)>samples:
            break
        
    # Select random subset
    random_samples = random.sample(list(zip(imgs, preds, probs, trues)), samples)

    # Plot grid
    cols = 4
    rows = math.ceil(samples/cols)
    fig = plt.figure(figsize=(5*cols, 4*rows))
    for idx, (img, p, pr, t) in enumerate(random_samples, 1):
        img = np.array(img).reshape(224, 224, 3)
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title(
            f"Pred: {idx_to_class[p]}, Real: {idx_to_class[t]}, Prob: {pr:.2f}"
        )
        plt.axis("off")
    
    # Log to TensorBoard
    tb_logger.experiment.add_figure(f"Predictions_Samples/{source}", fig, global_step=epoch)

    # Convert the Matplotlib figure to a PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Clean up
    plt.close(fig)
    buf.close()

    return img

# ---------------------- Model, loading and saving ---------------------- #
class TransferLearningWithEfficientNetV2S(pl.LightningModule):
    """
    LightningModule for EfficientNetV2-S transfer learning and fine-tuning.

    This model uses Timm to load a pretrained backbone, freezes early layers,
    and attaches a custom classifier head. Training and validation metrics
    are tracked and logged via TorchMetrics and TensorBoard.
    """
    def __init__(
        self,
        model_name: str = "efficientnetv2_s",
        weights: str = "DEFAULT",
        fine_tune_start: int = 99,
        learning_rate: float = 0.01,
        num_classes: int = 10,
        data_augmentation: bool = False,
        epochs: int = 10,
        class_weights: torch.FloatTensor = None,
    ):
        super().__init__()
        pretrained = True if weights == "DEFAULT" else False
        self.save_hyperparameters()

        # Loss
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

        # Determine pretrained flag and variant
        pretrained = self.hparams.weights == "DEFAULT"
        # Use TIMM variant with available pretrained weights
        variant_name = "tf_efficientnetv2_s_in21ft1k"

        # Load backbone without head
        backbone = timm.create_model(
            variant_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze entire backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze blocks starting from fine_tune_start
        start_idx = max(self.hparams.fine_tune_start - 1, 0)
        if hasattr(backbone, 'blocks'):
            #print(len(backbone.blocks)) There are 6 trainable blocks
            for idx, block in enumerate(backbone.blocks):
                if idx >= start_idx:
                    for param in block.parameters():
                        param.requires_grad = True

        # Classification head
        in_features = backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.hparams.num_classes),
        )

        # Combine backbone and head
        self.model = nn.Sequential(backbone, self.classifier)

        # Metrics
        self.mean_train_loss = MeanMetric()
        self.mean_train_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="micro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="micro")

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)
        
    def training_step(self, batch, *args, **kwargs):
        """
        Training step: compute loss and update metrics on a batch.
        """
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        pred_batch = output.argmax(dim=1)

        # Update metrics
        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_acc(pred_batch, target)

        # Log batch metrics
        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=True)
        self.log("train/batch_acc", self.mean_train_acc, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """
        Log aggregated training metrics and parameter histograms.
        """
        self.log("train/loss", self.mean_train_loss, prog_bar=True, logger=True)
        self.log("train/acc", self.mean_train_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)
        # TensorBoard histograms
        for name, param in self.model.named_parameters():
            self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"grads/{name}", param.grad, self.current_epoch)

    def validation_step(self, batch, *args, **kwargs):
        """
        Validation step: compute loss and update metrics on a batch.
        """
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        pred_batch = output.argmax(dim=1)

        # Update metrics
        self.mean_valid_loss(loss, weight=data.shape[0])
        self.mean_valid_acc(pred_batch, target)

    def on_validation_epoch_end(self):
        """
        Log aggregated validation metrics.
        """
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True, logger=True)
        self.log("valid/acc", self.mean_valid_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)


def load_model_for_TransferLearningandFT(
    data_module,
    train_config,
    data_config,
    ckpt_path: str = None
) -> pl.LightningModule:
    """
    Instantiate or load a Lightning model for transfer learning and fine-tuning.

    This function computes class weights from the training dataset and selects
    the appropriate model class based on the model_name in train_config.

    Args:
        data_module (LightningDataModule): Provides train_dataset.labels for weighting.
        train_config (TrainingConfiguration): Configuration object for training.
        ckpt_path (str, optional): Path to a checkpoint to load. Defaults to None.

    Returns:
        LightningModule: Initialized or loaded model ready for training or resuming.
    """
    class_weights = None

    # Compute class weights if data_module provided
    if data_module:
        data_module.setup()
        train_targets = np.array(data_module.train_dataset.labels, dtype=int)
        
        # Inverted weights based on class frecuency       
        counts = np.bincount(train_targets)  # [n_clase0, n_clase1, ...]
        weights = counts.sum() / (len(counts) * counts)  
        class_weights = torch.tensor(weights, dtype=torch.float)
        
    if not ckpt_path:
        model = TransferLearningWithEfficientNetV2S(
            fine_tune_start=train_config.fine_tune_start,
            num_classes=data_config.num_classes,
            learning_rate=train_config.learning_rate,
            data_augmentation =train_config.data_augmentation,
            epochs = train_config.epochs,
            class_weights=class_weights
        )
    else:
        model = TransferLearningWithEfficientNetV2S.load_from_checkpoint(
            ckpt_path,
            fine_tune_start=train_config.fine_tune_start,
            learning_rate=train_config.learning_rate,
            data_augmentation =train_config.data_augmentation,
            epochs = train_config.epochs,
            class_weights=class_weights
        )
    return model

class ProgressBarLeave(RichProgressBar):
    """
    Customized RichProgressBar that remains visible after completion.

    Args:
        refresh_rate (int): Updates per step (default=1).
        console_kwargs (dict, optional): Passed to underlying Console.
    """
    def __init__(
        self,
        refresh_rate: int = 1,
        console_kwargs: dict = None
    ):
        super().__init__(refresh_rate=refresh_rate, leave=True, console_kwargs=console_kwargs)

# ---------------------- Train and Validation ---------------------- #
def training_validation(
    tensorboard_logger,
    train_config,
    data_config,
    model=None,
    data_module=None,
    checkpoint_callback=None,
    ckpt_path: str = None
) -> tuple:
    """
    Set up and run the training and validation process.

    This function seeds randomness, prepares the data module and model,
    configures the Trainer with callbacks, and executes `.fit()`.

    Args:
        tensorboard_logger (TensorBoardLogger): Logger for TensorBoard.
        train_config (TrainingConfiguration): Training parameters.
        data_config (DataConfiguration): Data-related settings.
        model (LightningModule, optional): Predefined model to train.
        data_module (LightningDataModule, optional): Data loader module.
        checkpoint_callback (ModelCheckpoint, optional): Checkpoint callback.
        ckpt_path (str, optional): Checkpoint path to resume from.

    Returns:
        tuple: (model, data_module, checkpoint_callback) after training.
    """
    # Reproducibility
    pl.seed_everything(21, workers=True)

    # Initialize data module if needed
    if not data_module:
        data_module = KenyanFood_DataModule(
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            image_size=train_config.image_size,
            base_dir=data_config.data_root
        )

    # Initialize model if not provided
    if not model:
        model = load_model_for_TransferLearningandFT(data_module, train_config, data_config)

    # Configure checkpoint callback
    if not checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            monitor='valid/acc',
            mode="max",
            filename='transfer_learning-resnet18-epoch-{epoch:02d}',
            auto_insert_metric_name=True,
            save_last=True,           # saves last one
            save_top_k=1              # saves only the best one plus the last one
            #save_weights_only=True
        )
    # Early stopping and minimum epochs
    total_epochs = train_config.epochs
    
    early_stopping_callback = EarlyStopping(monitor="valid/acc", mode="max", patience=train_config.patience) 
    #min_epochs=int(train_config.min_epochs_percentage*train_config.epochs)
    precision = "32"
    if train_config.precision:
        precision = "16-mixed"
    # Initialize Trainer
    trainer = pl.Trainer(
                accelerator="auto", 
                devices="auto",  
                strategy="auto",
                logger=tensorboard_logger,
                #log_graph=True,
                max_epochs=total_epochs,
                #min_epochs=total_epochs,
                precision = precision,
                callbacks=[
                    early_stopping_callback,                                                         
                    checkpoint_callback,
                    ProgressBarLeave()
                ]  
            )
    
    # Toggle augmentation
    data_module.set_data_augmentation(train_config.data_augmentation)

    # Run training
    if ckpt_path:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data_module)

    # Update prev_epochs for resumed training
    train_config.prev_epochs = model.current_epoch
    return model, data_module, checkpoint_callback


# ---------------------- Submission CSV ---------------------- #
def generate_submission_file(model, data_module, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    all_ids = []
    all_preds = []

    class_names = data_module.val_dataset.classes
    test_loader = data_module.test_dataloader()
    
    for batch in test_loader:
        imgs, paths = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds_idx = torch.argmax(probs, dim=1).cpu().tolist()
    
        # Extract the IDs of the file and map to the predictions
        ids = [os.path.basename(p) for p in paths]
        all_ids.extend(ids)
        all_preds.extend([class_names[i] for i in preds_idx])
    
    # Build and save DataFrame
    df = pd.DataFrame({"ID": all_ids, "CLASS": all_preds})
    df.to_csv(output_csv, index=False)
    print(f"CSV generated at: {output_csv}")
    return df