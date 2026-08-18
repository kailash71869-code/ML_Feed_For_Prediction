# How to Train the Dataset

## 1. Add Your Dataset

Place your dataset inside the `dataset` folder.

For image classification, organize it like this:

```text
dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── class_3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each folder name represents one class.

Example:

```text
dataset/
├── happy/
├── neutral/
└── tired/
```

---

## 2. Install Requirements

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Start Training

After adding your dataset, run:

```bash
python train.py
```

The training script will:

1. Load the images from `dataset/`
2. Detect the classes automatically
3. Preprocess the images
4. Split the data into training and validation sets
5. Train the model
6. Evaluate the model
7. Save the trained model

---

## 4. Training Output

After successful training, the trained model will be saved inside:

```text
models/
```

Example:

```text
models/
└── best_model.pth
```

The class names should also be saved:

```text
models/
└── class_names.json
```

---

## 5. Adding More Data

To improve the model, add more images to the appropriate class folder.

Example:

```text
dataset/
├── happy/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── new_image1.jpg
│   └── new_image2.jpg
│
├── neutral/
└── tired/
```

Then train again:

```bash
python train.py
```

The model will be retrained using the updated dataset.

---

## 6. Important Dataset Rules

* Use correctly labelled images.
* Avoid duplicate images.
* Remove corrupted or extremely blurry images.
* Keep a reasonable number of images for every class.
* Use different people, poses, lighting conditions, and backgrounds where applicable.
* Do not put test images into the training dataset.

---

## 7. Simple Workflow

```text
Add Dataset
     ↓
dataset/class_name/
     ↓
Run train.py
     ↓
Model Training
     ↓
Model Evaluation
     ↓
best_model.pth
```

**To train your own dataset, simply place your images inside the correct class folders and run `python train.py`.**
