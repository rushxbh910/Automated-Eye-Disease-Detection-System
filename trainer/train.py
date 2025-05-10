import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn, optim
import torchvision.models as models
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

# === CONFIG ===
DATA_DIR = "transformed_eye_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
EPOCHS = 5
BATCH_SIZE = 32
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "densenet121_eye.pth")

# === TRANSFORM ===
transform = transforms.ToTensor()

# === LOADERS ===
train_loader = DataLoader(ImageFolder(TRAIN_DIR, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageFolder(VAL_DIR, transform=transform), batch_size=BATCH_SIZE, shuffle=False)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_loader.dataset.classes)
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier.in_features, num_classes)
)
model = model.to(device)

# === TRAINING SETUP ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# === MLFLOW SETUP ===
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "EyeDiseaseDetection")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION", os.path.join(os.getcwd(), "mlruns"))

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()
if not client.get_experiment_by_name(experiment_name):
    client.create_experiment(name=experiment_name, artifact_location=artifact_location)
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="DenseNet121_Training"):
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("model", "DenseNet121")
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average="macro")
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step(val_loss)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }, step=epoch)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
              f"F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    # === SAVE MODEL ===
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.pytorch.log_model(model, "model")
    print(f"✅ Model saved to {MODEL_PATH} and logged to MLflow.")