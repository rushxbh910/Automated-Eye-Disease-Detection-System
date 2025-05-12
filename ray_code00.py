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
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import VGG16_Weights, MobileNet_V3_Large_Weights, DenseNet121_Weights

# Import Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Define dataset path
dataset_path = "/mnt/data/transformed_eye_dataset"
print(f"Using dataset path: {dataset_path}")
print(dataset_path)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)

# Split training set to create validation set
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to count images per class
def count_images_per_class(dataset):
    # Check if this is a subset from random_split
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # For a dataset that's been through random_split
        class_counts = {cls: 0 for cls in dataset.dataset.classes}
        for idx in dataset.indices:
            _, label = dataset.dataset.samples[idx]
            class_counts[dataset.dataset.classes[label]] += 1
    else:
        # For a direct ImageFolder dataset
        class_counts = {cls: 0 for cls in dataset.classes}
        for _, label in dataset.samples:
            class_counts[dataset.classes[label]] += 1
    
    return class_counts

# Get number of classes
num_classes = len(train_dataset.dataset.classes)
print(f"Classes: {train_dataset.dataset.classes}, Total Classes: {num_classes}")
class_names = train_dataset.dataset.classes

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

# Lightning implementation of the model
class DenseNetLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001, freeze_layers=True):
        super().__init__()
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Model definition
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

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
                    
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.classifier.in_features, num_classes)
        )
        
        # Store parameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # For storing validation predictions
        self.val_labels = []
        self.val_preds = []
        
    def forward(self, x):
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
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
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
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
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

# Visualization functions
def plot_accuracy_and_loss(train_losses, val_losses, train_accs, val_accs):
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

def plot_confusion_matrix(y_true, y_pred, class_names):
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

def plot_per_class_accuracy(y_true, y_pred, class_names):
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

def compute_classification_metrics(y_true, y_pred, num_classes, class_names):
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

# Lightning Trainer with callbacks
def train_lightning_model():
    # Training parameters
    epochs = 20
    learning_rate = 0.001
    
    # Create model instance
    model = DenseNetLightning(
        num_classes=num_classes,
        learning_rate=learning_rate,
        freeze_layers=True
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True, 
        mode='min'
    )
    
    # Metrics tracking callback
    metrics_callback = MetricsCallback()
    
    # Initialize trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[early_stop, metrics_callback],
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    
    # Train the model
    print("\nTraining DenseNet121 (unfreeze last 2 blocks) with Adam optimizer...\n")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model in .pth format
    torch.save(model.state_dict(), '/mnt/data/densenet121_model.pth')
    print(f"Model saved as densenet121_model.pth")
    
    return model, trainer, metrics_callback

# Main execution
if __name__ == "__main__":
    # Train the Lightning model
    model, trainer, metrics_callback = train_lightning_model()
    
    # Get logs for plotting - using model's stored metrics
    train_losses = model.train_losses
    val_losses = model.val_losses
    train_accs = model.train_accs
    val_accs = model.val_accs
    
    # Visualize training results
    plot_accuracy_and_loss(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate on validation set
    model.eval()
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    # Plot confusion matrix and per-class accuracy
    plot_confusion_matrix(all_val_labels, all_val_preds, class_names)
    plot_per_class_accuracy(all_val_labels, all_val_preds, class_names)
    compute_classification_metrics(all_val_labels, all_val_preds, num_classes, class_names)