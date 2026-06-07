[README_1.md](https://github.com/user-attachments/files/28685836/README_1.md)
# Counterfeit Medicine Detection Using CNN

A deep learning project that uses Convolutional Neural Networks (CNNs) to classify medicine packaging images as **Real** or **Fake** based on visual features such as print quality, foil texture, and colour consistency.

---

## Project Overview

Counterfeit medicines are a global public health threat. This project applies CNN-based binary image classification to automatically detect fake medicines from packaging images — a fast, scalable alternative to laboratory chemical analysis.

Two models are trained and compared:
- **Custom CNN** — a 3-block convolutional network built from scratch
- **MobileNetV2** — a transfer learning model pre-trained on ImageNet

---

## Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Fake vs Real Medicine Datasets Images](https://www.kaggle.com/datasets/surajkumarjha1/fake-vs-real-medicine-datasets-images) |
| Classes | 2 (Fake, Real) |
| Image Size | Resized to 224 × 224 pixels |
| Splits | Train / Validation / Test |
| Format | RGB JPEG/PNG in class-labelled subdirectories |

---

## Repository Structure

```
counterfeit-medicine-cnn/
├── notebooks/
│   └── counterfeit_medicine_cnn.ipynb   # Main Kaggle notebook
├── figures/
│   ├── sample_medicine_images.png
│   ├── dataset_split.png
│   ├── dataset_split_distribution.png
│   ├── augmented_examples.png
│   ├── fake_vs_real_comparison.png
│   ├── rq1_accuracy_curve.png
│   └── roc_curve.png
├── README.md
└── requirements.txt
```

---

## Model Architecture

### Custom CNN

```
Input (224×224×3)
→ Conv2D(32) → MaxPool
→ Conv2D(64) → MaxPool
→ Conv2D(128) → MaxPool
→ Flatten → Dense(128) → Dropout(0.5)
→ Dense(1, Sigmoid)
```

Compiled with: `Adam` | `binary_crossentropy` | 10 epochs

### Transfer Learning — MobileNetV2

- MobileNetV2 base (ImageNet weights, frozen)
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dense(1, Sigmoid)

Compiled with: `Adam` | `binary_crossentropy` | 5 epochs

---

## Results

| Model | Epochs | Best Val Accuracy | ROC-AUC |
|---|---|---|---|
| Custom CNN | 10 | ~96.91% | 0.990 |
| MobileNetV2 | 5 | — | — |

> MobileNetV2 test accuracy is printed at runtime via `mobilenet_model.evaluate(test_generator)`.

---

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately can a custom CNN classify real vs. counterfeit medicines? |
| RQ2 | What visual differences does the CNN rely on for classification? |
| RQ3 | How does data augmentation improve CNN robustness? |
| RQ4 | How does model performance evolve across training epochs? |
| RQ5 | How well are genuine and counterfeit images separable (ROC-AUC)? |

---

## How to Run

### Option 1 — Kaggle Notebook (Recommended)

1. Open the notebook on Kaggle
2. Enable GPU accelerator under **Settings → Accelerator → GPU**
3. Add your Kaggle API token (`kaggle.json`) to the Kaggle secrets or upload it manually
4. Run all cells top to bottom

### Option 2 — Google Colab

1. Upload `counterfeit_medicine_cnn.ipynb` to Colab
2. Mount Google Drive or upload the dataset manually
3. Update `train_path`, `valid_path`, and test path variables to match your directory
4. Run all cells

### Option 3 — Local Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/counterfeit-medicine-cnn.git
cd counterfeit-medicine-cnn

# Install dependencies
pip install -r requirements.txt

# Download dataset via Kaggle API
kaggle datasets download -d surajkumarjha1/fake-vs-real-medicine-datasets-images
unzip fake-vs-real-medicine-datasets-images.zip -d dataset/

# Launch Jupyter
jupyter notebook notebooks/counterfeit_medicine_cnn.ipynb
```

---

## Requirements

```
tensorflow>=2.10
numpy
matplotlib
scikit-learn
kaggle
graphviz
Pillow
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Output Figures

All figures are saved automatically as high-resolution PNG files (300 DPI) during notebook execution:

- `sample_medicine_images.png` — representative images from both classes
- `dataset_split.png` — bar chart of train/val/test image counts
- `augmented_examples.png` — examples of augmented training images
- `fake_vs_real_comparison.png` — side-by-side Fake vs. Real samples
- `rq1_accuracy_curve.png` — training and validation accuracy per epoch
- `roc_curve.png` — ROC curve with AUC score

---

## Course Information

| Field | Details |
|---|---|
| Course | Machine Learning |
| Phase | Phase 2 — Proposal and Code Implementation |
| Student | MAHESH GOWDA SONNEGOWDA|
| University | University of Europe for Applied Sciences (UE Potsdam) |
| Programme | MSc Data Science |

---

## License

This project is submitted for academic purposes. The dataset is publicly available on Kaggle under its original licence. All code in this repository is original work by the student.
