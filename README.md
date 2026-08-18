# 🧠 ML Model Training with Google Colab

> 🚀 Train your **custom image dataset** using Google Colab GPU and save the trained model directly to Google Drive.

---

## 📌 Overview

This guide explains how to train an ML image-classification model using your **own dataset**.

The workflow is:

```text
        📂 Custom Dataset
               │
               ▼
        ☁️ Google Drive
               │
               ▼
        🔗 Google Colab
               │
               ▼
        🧹 Preprocessing
               │
               ▼
        ✂️ Train / Validation Split
               │
               ▼
        🧠 ResNet18 Training
               │
               ▼
        📊 Model Evaluation
               │
               ▼
        💾 Trained Model
               │
               ▼
        🚀 Ready for Deployment
```

---

# ⚡ 1. Open Google Colab

Create a new notebook in **Google Colab**.

### Enable GPU

Go to:

```text
Runtime → Change runtime type → T4 GPU → Save
```

Then verify that the GPU is available:

```python
!nvidia-smi
```

If the GPU is enabled, Colab will display information about the available NVIDIA GPU.

---

# 📂 2. Prepare Your Dataset

Upload your dataset to **Google Drive**.

The dataset should follow this structure:

```text
📁 MyDrive
└── 📁 dataset
    ├── 📁 happy
    │   ├── 🖼️ image1.jpg
    │   ├── 🖼️ image2.jpg
    │   └── ...
    │
    ├── 📁 neutral
    │   ├── 🖼️ image1.jpg
    │   ├── 🖼️ image2.jpg
    │   └── ...
    │
    └── 📁 tired
        ├── 🖼️ image1.jpg
        ├── 🖼️ image2.jpg
        └── ...
```

### 💡 Important

**Each folder represents one class.**

For example:

| Folder    | Class      |
| --------- | ---------- |
| `happy`   | 😊 Happy   |
| `neutral` | 😐 Neutral |
| `tired`   | 😴 Tired   |

You can replace these classes with your own classes.

For example:

```text
dataset/
├── cat/
├── dog/
└── horse/
```

or:

```text
dataset/
├── healthy/
├── diseased/
└── unknown/
```

---

# 🔗 3. Connect Google Drive

Run this cell in Colab:

```python
from google.colab import drive

drive.mount('/content/drive')
```

Google Colab will ask for permission.

After authorization, your Google Drive will be available inside:

```text
/content/drive/MyDrive/
```

---

# 📍 4. Set Dataset Path

Set the location of your dataset:

```python
DATASET_PATH = "/content/drive/MyDrive/dataset"
```

Check whether the dataset is detected correctly:

```python
import os

print(os.listdir(DATASET_PATH))
```

Expected output:

```text
['happy', 'neutral', 'tired']
```

If you see your class folders, your dataset is ready. ✅

---

# 📦 5. Install Required Libraries

Run:

```python
!pip install torch torchvision scikit-learn matplotlib
```

These libraries are used for:

| Library        | Purpose                     |
| -------------- | --------------------------- |
| `PyTorch`      | 🧠 Model training           |
| `Torchvision`  | 🖼️ Image datasets & models |
| `Scikit-learn` | 📊 Evaluation               |
| `Matplotlib`   | 📈 Visualization            |

---

# 🖼️ 6. Load the Dataset

Use `ImageFolder` to automatically detect the class folders.

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))
```

Example output:

```text
Classes: ['happy', 'neutral', 'tired']
Total images: 3000
```

---

# ✂️ 7. Split the Dataset

The dataset needs to be divided into training and validation data.

```python
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
```

### 📊 Dataset Split

```text
         📂 Dataset
             │
       ┌─────┴─────┐
       ▼           ▼
   🟢 80%       🔵 20%
   Training     Validation
```

---

# 🚚 8. Create DataLoaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
```

### Why DataLoader?

It loads images in batches instead of loading the entire dataset into memory at once.

```text
Dataset
   ↓
Batch 1 → 32 images
Batch 2 → 32 images
Batch 3 → 32 images
   ↓
Model
```

---

# 🧠 9. Create the Model

We use **ResNet18** with transfer learning.

Instead of training an entire neural network from scratch, we start with a model that has already learned useful image features.

```python
import torch
import torch.nn as nn
from torchvision import models

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = models.resnet18(weights="DEFAULT")

num_classes = len(dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

print("Using device:", device)
```

Expected:

```text
Using device: cuda
```

if GPU is enabled.

---

# ⚙️ 10. Define Loss & Optimizer

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### Components

```text
Loss Function
     ↓
Measures how wrong the prediction is

Optimizer
     ↓
Updates the model weights

Learning Rate
     ↓
Controls how quickly the model learns
```

---

# 🚀 11. Train the Model

Set the number of training epochs:

```python
epochs = 10
```

Then train:

```python
for epoch in range(epochs):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Loss: {loss.item():.4f}"
    )
```

Example:

```text
Epoch 1/10 | Loss: 0.8421
Epoch 2/10 | Loss: 0.5214
Epoch 3/10 | Loss: 0.3187
...
Epoch 10/10 | Loss: 0.1023
```

As training progresses, the loss should generally decrease.

---

# 📊 12. Evaluate the Model

After training, evaluate the model using the validation dataset.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Validation Accuracy: {accuracy:.2f}%")
```

Example:

```text
Validation Accuracy: 91.35%
```

> ⚠️ Accuracy depends entirely on your dataset and should not be treated as guaranteed.

---

# 💾 13. Save the Trained Model

Save the trained model directly to Google Drive:

```python
MODEL_PATH = "/content/drive/MyDrive/best_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("✅ Model saved:", MODEL_PATH)
```

Your trained model will now be available in:

```text
📁 MyDrive
└── best_model.pth
```

---

# 🏷️ 14. Save Class Names

The model also needs to know which output number corresponds to which class.

```python
import json

CLASS_PATH = "/content/drive/MyDrive/class_names.json"

with open(CLASS_PATH, "w") as f:
    json.dump(dataset.classes, f)

print(dataset.classes)
```

Example:

```json
[
    "happy",
    "neutral",
    "tired"
]
```

Now you have:

```text
📁 MyDrive
├── 🧠 best_model.pth
└── 🏷️ class_names.json
```

---

# 🔄 15. Train a New Dataset

Want to train another dataset?

Simply replace the dataset:

```text
📁 dataset
├── class_1
├── class_2
└── class_3
```

Then change:

```python
DATASET_PATH = "/content/drive/MyDrive/dataset"
```

Run the Colab cells again.

The new model will be trained using your new dataset.

---

# 🧪 16. Improve the Model

If the model does not perform well, try:

```text
📈 More training images
      ↓
🧹 Better data quality
      ↓
🔄 Data augmentation
      ↓
⚙️ Tune learning rate
      ↓
🔢 Increase epochs
      ↓
🧠 Try another pretrained model
```

Most importantly:

> **Better and more diverse data usually matters more than simply increasing the number of epochs.**

---

# ✅ Final Workflow

```text
┌───────────────────────────┐
│      📂 YOUR DATASET      │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│     ☁️ GOOGLE DRIVE       │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│     🔗 GOOGLE COLAB       │
│          + GPU            │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│    🖼️ PREPROCESS DATA     │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│     ✂️ SPLIT DATASET      │
│       80% / 20%           │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│      🧠 TRAIN MODEL       │
│        ResNet18           │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│       📊 EVALUATE         │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│      💾 SAVE MODEL        │
│    best_model.pth         │
└───────────────────────────┘
```

---

## 🎯 Quick Start

If your dataset is already prepared, the basic process is:

```text
1️⃣ Upload dataset to Google Drive

2️⃣ Open Google Colab

3️⃣ Enable T4 GPU

4️⃣ Mount Google Drive

5️⃣ Set DATASET_PATH

6️⃣ Install libraries

7️⃣ Load dataset

8️⃣ Split dataset

9️⃣ Train ResNet18

🔟 Evaluate model

1️⃣1️⃣ Save best_model.pth

1️⃣2️⃣ Save class_names.json
```

### 🏆 Result

```text
Your Dataset
     ↓
Google Colab GPU
     ↓
Trained ML Model
     ↓
best_model.pth
     ↓
Ready to use in your application 🚀
```
