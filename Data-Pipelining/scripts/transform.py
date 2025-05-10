import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
RAW_DATA_DIR = "/data/raw_eye_dataset"
OUTPUT_DIR = "/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define image transformations
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(RAW_DATA_DIR, transform=base_transform)
labels = [label for _, label in dataset.samples]

# Split into train/test (80/20 stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

# Further split test set into 3 equal holdout subsets (stratified)
holdout_labels = [labels[i] for i in test_idx]
sss_holdout = StratifiedShuffleSplit(n_splits=3, test_size=1/3, random_state=42)
holdout_splits = list(sss_holdout.split(np.zeros(len(holdout_labels)), holdout_labels))

holdouts = []
used_indices = set()
for i, (train_h, test_h) in enumerate(holdout_splits):
    # Avoid overlaps
    new_indices = [j for j in test_h if j not in used_indices]
    used_indices.update(new_indices)
    subset_indices = [test_idx[j] for j in new_indices]
    holdouts.append(Subset(dataset, subset_indices))

# Save function
def save_subset(subset, output_path):
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    for i, (img, label) in enumerate(loader):
        class_dir = os.path.join(output_path, dataset.classes[label.item()])
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"img_{i:05d}.png")
        save_image(img, save_path)

# Save all splits
save_subset(train_set, os.path.join(OUTPUT_DIR, "train"))
save_subset(test_set, os.path.join(OUTPUT_DIR, "test"))
for i, holdout in enumerate(holdouts):
    save_subset(holdout, os.path.join(OUTPUT_DIR, f"holdout_{i+1}"))

print("✅ Data transformed and saved into:")
print(f"- {OUTPUT_DIR}/train")
print(f"- {OUTPUT_DIR}/test")
print(f"- {OUTPUT_DIR}/holdout_1, holdout_2, holdout_3")
