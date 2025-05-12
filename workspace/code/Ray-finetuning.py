import os
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import VGG16_Weights, MobileNet_V3_Large_Weights, DenseNet121_Weights

# Import Ray and Ray Tune
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# Import Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Import MLflow directly
import mlflow

# Debug prints
print("Script starting...")

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1})
print(f"Ray Scaling Config: {scaling_config}")

# Define dataset path
dataset_path = "/mnt/data/transformed_eye_dataset"
print(f"Using dataset path: {dataset_path}")
print(dataset_path)

# Define advanced augmentation for training data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize larger for crop augmentation
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomVerticalFlip(p=0.1),    # 10% chance to flip vertically (less common)
    transforms.RandomRotation(15),           # Random rotation up to 15 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color jittering
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define minimal transformations for validation and test data
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Just resize to target size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create a custom dataset wrapper to apply different transforms to the same data
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, transform=None, num_augmentations=1):
        self.original_dataset = original_dataset
        self.transform = transform
        self.num_augmentations = num_augmentations
        
    def __len__(self):
        return len(self.original_dataset) * self.num_augmentations
    
    def __getitem__(self, idx):
        # Get original sample index and augmentation index
        original_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        
        # Get the original image and label
        original_image, label = self.original_dataset[original_idx]
        
        # If it's the first augmentation and no transform is specified, return original
        if aug_idx == 0 and self.transform is None:
            return original_image, label
        
        # If it's not the first augmentation or a transform is specified, augment
        if isinstance(original_image, torch.Tensor):
            # Convert tensor to PIL for transforms if needed
            original_image = transforms.ToPILImage()(original_image)
            
        # Apply the transform
        if self.transform:
            augmented_image = self.transform(original_image)
            return augmented_image, label
        else:
            return transforms.ToTensor()(original_image), label

# Define a callback to track metrics
class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'].item())
        if 'train_acc' in metrics:
            self.train_accs.append(metrics['train_acc'].item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'].item())
        if 'val_acc' in metrics:
            self.val_accs.append(metrics['val_acc'].item())

# Custom callback to log metrics to MLflow
class MLflowLoggingCallback(pl.Callback):
    def __init__(self, use_mlflow=True):
        super().__init__()
        self.use_mlflow = use_mlflow
        self.epoch = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.use_mlflow:
            return
        
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            try:
                mlflow.log_metric("train_loss", metrics['train_loss'].item(), step=self.epoch)
            except Exception as e:
                print(f"MLflow logging error (non-critical): {e}")
                
        if 'train_acc' in metrics:
            try:
                mlflow.log_metric("train_acc", metrics['train_acc'].item(), step=self.epoch)
            except Exception as e:
                print(f"MLflow logging error (non-critical): {e}")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.use_mlflow:
            return
            
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics:
            try:
                mlflow.log_metric("val_loss", metrics['val_loss'].item(), step=self.epoch)
            except Exception as e:
                print(f"MLflow logging error (non-critical): {e}")
                
        if 'val_acc' in metrics:
            try:
                mlflow.log_metric("val_acc", metrics['val_acc'].item(), step=self.epoch)
            except Exception as e:
                print(f"MLflow logging error (non-critical): {e}")
        
        self.epoch += 1

# Lightning implementation of the model
class DenseNetLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001, freeze_layers=True, dropout_probability=0.4):
        super().__init__()
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Model definition
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Explicitly move the model to the appropriate device
        self.model = self.model.to(device)

        if freeze_layers:
            # For DenseNet121, model.features is a Sequential of various layers.
            # Unfreeze only the last two modules.
            features = list(self.model.features.children())
            total_blocks = len(features)
            for idx, module in enumerate(features):
                if idx < total_blocks - 2:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
                    
        # Replace classifier - now with configurable dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(self.model.classifier.in_features, num_classes)
        ).to(device)  # Move classifier to device too
        
        # Store parameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # For storing validation predictions
        self.val_labels = []
        self.val_preds = []
        
    def forward(self, x):
        # Ensure input is on the same device as model
        if x.device != next(self.model.parameters()).device:
            x = x.to(next(self.model.parameters()).device)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(1)
        acc = (preds == labels).float().mean()
        
        # Store metrics
        self.train_losses.append(loss.item())
        self.train_accs.append(acc.item())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(1)
        acc = (preds == labels).float().mean()
        
        # Store predictions and labels for confusion matrix
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        
        # Store metrics
        self.val_losses.append(loss.item())
        self.val_accs.append(acc.item())
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Concatenate all predictions and labels
        if len(self.val_preds) > 0 and len(self.val_labels) > 0:
            all_preds = torch.cat(self.val_preds)
            all_labels = torch.cat(self.val_labels)
            
            # Reset for next epoch
            self.val_preds = []
            self.val_labels = []
            
            return all_preds, all_labels
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Safe MLflow logging function
def safe_mlflow_log(func, *args, **kwargs):
    """Safely call MLflow logging functions, handling exceptions"""
    try:
        if mlflow.active_run():
            return func(*args, **kwargs)
    except Exception as e:
        print(f"MLflow logging error (non-critical): {e}")
        return None

# Visualization functions
def plot_accuracy_and_loss(train_losses, val_losses, train_accs, val_accs, use_mlflow=False):
    plt.figure(figsize=(12, 5))
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()
    
    # Log the figure to MLflow
    if use_mlflow:
        safe_mlflow_log(mlflow.log_artifact, 'learning_curves.png')

def plot_confusion_matrix(y_true, y_pred, class_names, use_mlflow=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Log the figure to MLflow
    if use_mlflow:
        safe_mlflow_log(mlflow.log_artifact, 'confusion_matrix.png')

def plot_per_class_accuracy(y_true, y_pred, class_names, use_mlflow=False):
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
    plt.figure(figsize=(14, 8))
    plt.bar(class_names, per_class_accuracy, color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png')
    plt.close()
    
    # Log the figure to MLflow
    if use_mlflow:
        safe_mlflow_log(mlflow.log_artifact, 'per_class_accuracy.png')

def compute_classification_metrics(y_true, y_pred, num_classes, class_names, use_mlflow=False):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Initialize TP, FP, FN
    true_positive = torch.zeros(num_classes)
    false_positive = torch.zeros(num_classes)
    false_negative = torch.zeros(num_classes)

    # Compute TP, FP, FN for each class
    for i in range(num_classes):
        true_positive[i] = ((y_pred == i) & (y_true == i)).sum().item()
        false_positive[i] = ((y_pred == i) & (y_true != i)).sum().item()
        false_negative[i] = ((y_pred != i) & (y_true == i)).sum().item()

    # Compute precision, recall, and F1-score
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Print results
    print("\nClassification Report DenseNetV3:\n")
    print(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("=" * 50)
    for i in range(num_classes):
        print(f"{class_names[i]:<15}{precision[i].item():<12.4f}{recall[i].item():<12.4f}{f1_score[i].item():<12.4f}")

    # Compute overall accuracy
    accuracy = (y_pred == y_true).sum().item() / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")

    # Save metrics to a file
    with open('classification_metrics.txt', 'w') as f:
        f.write("\nClassification Report DenseNetV3:\n\n")
        f.write(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}\n")
        f.write("=" * 50 + "\n")
        for i in range(num_classes):
            f.write(f"{class_names[i]:<15}{precision[i].item():<12.4f}{recall[i].item():<12.4f}{f1_score[i].item():<12.4f}\n")
        f.write(f"\nOverall Accuracy: {accuracy:.4f}\n")
    
    # Log metrics to MLflow
    if use_mlflow:
        for i in range(num_classes):
            safe_mlflow_log(mlflow.log_metric, f"precision_{class_names[i]}", precision[i].item())
            safe_mlflow_log(mlflow.log_metric, f"recall_{class_names[i]}", recall[i].item())
            safe_mlflow_log(mlflow.log_metric, f"f1_score_{class_names[i]}", f1_score[i].item())
        
        safe_mlflow_log(mlflow.log_metric, "overall_accuracy", accuracy)
        safe_mlflow_log(mlflow.log_artifact, 'classification_metrics.txt')
    
    return accuracy

# Visualize some augmented images for verification
def visualize_augmentations(dataset, num_samples=3, augmentations_per_sample=5):
    """Visualize augmentations of random samples to verify augmentation pipeline"""
    # Select random samples
    sample_indices = random.sample(range(len(dataset.original_dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, augmentations_per_sample + 1, figsize=(3*(augmentations_per_sample + 1), 3*num_samples))
    
    for i, sample_idx in enumerate(sample_indices):
        # Get original sample
        original_img, label = dataset.original_dataset[sample_idx]
        
        # Convert to PIL image if it's a tensor
        if isinstance(original_img, torch.Tensor):
            original_img = transforms.ToPILImage()(original_img)
        
        # Display original
        if num_samples > 1:
            ax = axes[i, 0]
        else:
            ax = axes[0]
        
        ax.imshow(original_img)
        ax.set_title(f"Original\n{class_names[label]}")
        ax.axis('off')
        
        # Generate and display augmentations
        for j in range(augmentations_per_sample):
            # Create augmented version
            augmented_img = dataset.transform(original_img)
            
            # Convert tensor to numpy for display
            if isinstance(augmented_img, torch.Tensor):
                augmented_img = augmented_img.permute(1, 2, 0).numpy()
                # Denormalize
                augmented_img = augmented_img * 0.5 + 0.5
                augmented_img = np.clip(augmented_img, 0, 1)
            
            # Display augmented image
            if num_samples > 1:
                ax = axes[i, j + 1]
            else:
                ax = axes[j + 1]
                
            ax.imshow(augmented_img)
            ax.set_title(f"Aug {j+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.close()
    
    # Return the figure path for MLflow logging
    return 'augmentation_examples.png'

# Dummy context manager
class DummyContextManager:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Load and prepare datasets
def prepare_data():
    global train_dataset_original, test_dataset, train_subset, val_subset, augmented_train_dataset, val_dataset, num_classes, class_names
    
    print("Loading datasets...")
    # Load datasets with appropriate transforms
    train_dataset_original = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=None)
    test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=eval_transform)

    # Split the original training set to create validation set
    train_size = int(0.8 * len(train_dataset_original))
    val_size = len(train_dataset_original) - train_size

    # Create random indices for splitting
    indices = list(range(len(train_dataset_original)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train and validation subsets using the indices
    train_subset = torch.utils.data.Subset(train_dataset_original, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset_original, val_indices)

    # Number of augmentations per original image
    num_augmentations = 5  # Each original image will generate 5 augmented versions

    # Apply augmentation to training subset only
    augmented_train_dataset = AugmentedDataset(train_subset, transform=train_transform, num_augmentations=num_augmentations)

    # Apply only normalization transforms to validation set (no augmentation)
    val_dataset = AugmentedDataset(val_subset, transform=eval_transform, num_augmentations=1)

    # Get number of classes
    num_classes = len(train_dataset_original.classes)
    class_names = train_dataset_original.classes
    
    # Print dataset sizes
    print(f"Original training dataset size: {len(train_dataset_original)}")
    print(f"Training subset size (before augmentation): {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Augmented training dataset size: {len(augmented_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {class_names}, Total Classes: {num_classes}")
    
    return train_dataset_original, train_subset, val_subset, augmented_train_dataset, val_dataset, test_dataset, num_classes, class_names

# Ray Tune training function - minimal version to avoid hanging
def tune_train_func(config):
    print(f"Starting Ray Tune training with config: {config}")
    
    # Create data loaders with the batch size from config
    batch_size = config.get("batch_size", 32)
    num_workers = 1  # Reduced for Ray Tune
    
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Create model with hyperparameters from config
    model = DenseNetLightning(
        num_classes=num_classes,
        learning_rate=config.get("learning_rate", 0.001),
        freeze_layers=config.get("freeze_layers", True),
        dropout_probability=config.get("dropout_probability", 0.4)
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,  # Shorter patience for hyperparameter tuning
        verbose=True, 
        mode='min'
    )
    
    # Initialize trainer with fewer epochs for hyperparameter tuning
    trainer = pl.Trainer(
        max_epochs=config.get("epochs", 5),
        callbacks=[early_stop],
        enable_progress_bar=True,
        logger=False,  # Disable logging for cleaner output
        enable_checkpointing=False  # Disable checkpointing for faster tuning
    )
    
    print("Starting training...")
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("Training completed")
    
    # Get validation accuracy
    model.eval()
    val_preds = []
    val_labels = []
    
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())
    
    # Compute validation accuracy
    val_accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Report metrics to Ray Tune
    tune.report(val_accuracy=val_accuracy)

# Function to run hyperparameter tuning with Ray Tune
def run_hyperparameter_tuning(num_samples=8):
    # Define hyperparameter search space
    config = {
        "batch_size": tune.choice([32, 64]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "dropout_probability": tune.uniform(0.2, 0.6),
        "freeze_layers": True,
        "epochs": 5  # Use fewer epochs for hyperparameter tuning
    }
    
    # ASHA scheduler for efficient early stopping
    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2
    )
    
    # Run hyperparameter tuning
    print("Starting hyperparameter tuning with Ray Tune...")
    
    # Use a simplified version without TorchTrainer to avoid potential hanging
    tuner = tune.Tuner(
        tune.with_resources(
            tune_train_func,
            resources={"cpu": 2, "gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            metric="val_accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples
        ),
        param_space=config
    )
    
    results = tuner.fit()
    
    # Get best hyperparameters
    best_trial = results.get_best_result("val_accuracy", "max")
    best_config = best_trial.config
    
    print(f"\nBest hyperparameters found:")
    print(f"Batch size: {best_config['batch_size']}")
    print(f"Learning rate: {best_config['learning_rate']:.6f}")
    print(f"Dropout probability: {best_config['dropout_probability']:.4f}")
    print(f"Best validation accuracy: {best_trial.metrics['val_accuracy']:.4f}")
    
    return best_config

# Train the final model with MLflow tracking (using the original working code)
def train_lightning_model(config=None):
    # Training parameters
    experiment_name = "DenseNet121_Eye_Classification_Tuned"
    epochs = 20
    learning_rate = 0.001
    dropout_probability = 0.4
    batch_size = 32
    
    # Override with tuned parameters if available
    if config:
        learning_rate = config.get("learning_rate", learning_rate)
        dropout_probability = config.get("dropout_probability", dropout_probability)
        batch_size = config.get("batch_size", batch_size)
    
    # Set up MLflow - use direct MLflow API without PyTorch Lightning MLflow logger
    mlflow_uris = ["http://172.18.0.5:8000", "http://mlflow:8000"]
    use_mlflow = False
    
    # Try to connect to MLflow
    print("\nSetting up MLflow connection...")
    for uri in mlflow_uris:
        try:
            print(f"Trying to connect to MLflow at {uri}")
            mlflow.set_tracking_uri(uri)
            # Check if experiment exists or create it
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                print(f"Found existing experiment '{experiment_name}' with ID {experiment.experiment_id}")
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment '{experiment_name}' with ID {experiment_id}")
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            print(f"Successfully set up MLflow with experiment '{experiment_name}'")
            use_mlflow = True
            break
        except Exception as e:
            print(f"Error connecting to MLflow at {uri}: {e}")
    
    if not use_mlflow:
        print("Could not connect to MLflow. Training will continue without MLflow logging.")
    
    # Create data loaders
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Visualize some augmentations before training
    print("Generating augmentation examples...")
    aug_examples_path = visualize_augmentations(augmented_train_dataset, num_samples=3, augmentations_per_sample=5)
    
    # Create a MLflow run if MLflow is available
    if use_mlflow:
        try:
            active_run = mlflow.start_run(run_name=f"densenet121_tuned_run_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"Started MLflow run with ID: {active_run.info.run_id}")
            
            # Log parameters
            mlflow.log_params({
                "model_type": "DenseNet121",
                "freeze_layers": True,
                "learning_rate": learning_rate,
                "dropout_probability": dropout_probability,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "Adam",
                "num_classes": num_classes,
                "num_augmentations": num_augmentations,
                "original_train_size": len(train_dataset_original),
                "augmented_train_size": len(augmented_train_dataset),
                "validation_size": len(val_dataset),
                "test_size": len(test_dataset)
            })
            
            # Log augmentation examples
            mlflow.log_artifact(aug_examples_path)
            
            print("Logged basic parameters and augmentation examples to MLflow")
        except Exception as e:
            print(f"Error starting MLflow run: {e}")
            use_mlflow = False
