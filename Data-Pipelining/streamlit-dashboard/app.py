import streamlit as st
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

# Set the base path where your transformed dataset is mounted
DATA_PATH = "/data"

# Sidebar controls
st.sidebar.title("Dataset Viewer")
split = st.sidebar.selectbox("Select dataset", sorted(os.listdir(DATA_PATH)))
max_images = 5
num_images = st.sidebar.slider("Images per class", 1, max_images, 3)

# Title
st.title("Eye Disease Dataset Dashboard")

# === Class Distribution Bar Chart ===
split_path = os.path.join(DATA_PATH, split)
classes = sorted(os.listdir(split_path))
class_counts = {cls: len(os.listdir(os.path.join(split_path, cls))) for cls in classes}

st.subheader("Image Count per Class")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(class_counts.keys(), class_counts.values(), color='skyblue')
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
ax.set_title(f"Class Distribution in '{split}' Split")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# === Sample Image Display ===
st.subheader("Sample Images")

for cls in classes:
    cls_path = os.path.join(split_path, cls)
    image_files = os.listdir(cls_path)
    if not image_files:
        continue

    st.markdown(f"### {cls}")

    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    cols = st.columns(min(len(selected_images), 5))  # Show up to 5 images in a row

    for col, img_name in zip(cols, selected_images):
        img_path = os.path.join(cls_path, img_name)
        image = Image.open(img_path)
        image.thumbnail((224, 224))
        col.image(image, caption=img_name, use_container_width=True)
