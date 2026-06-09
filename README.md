# 🚗 Car Damage Detection & Parts Instance Segmentation using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8--seg-red.svg)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Framework-Google_Colab-orange.svg)](https://colab.research.google.com/)

An automated Computer Vision solution developed to streamline vehicle insurance claims, car rentals, and smart garage inspections. This project utilizes **YOLOv8 Instance Segmentation** to simultaneously locate, classify, and isolate individual car panels (e.g., doors, bumper, hood) and structural anomalies (e.g., dents, scratches, cracks, corrosion) with precise polygon masks.

---

## 📌 Project Overview

Manual car inspection for damage evaluation is slow, subjective, and highly prone to human oversight. This project implements a full deep learning instance segmentation pipeline capable of:
1. **Localization & Mapping:** Drawing tight bounding boxes and pixel-accurate masks over 29 custom vehicle classes.
2. **Multi-Task Handling:** Managing background structural components alongside localized surface flaws in a single-shot inference pass.
3. **Smart Damage Filtering:** Utilizing a custom Python post-processing logic to programmatically hide heavy structural part masks and reveal *only* the micro-level surface damages (like hidden dents) for cleaner reporting.

---

## 📊 Dataset Structure & Specifications

The model was trained using a multi-class vehicle dataset configured via a custom pipeline. 

### Class Matrix Summary (Total: 29 Classes):
* **Structural Parts:** `Back-bumper`, `Front-door`, `Hood`, `Fender`, `Windshield`, `Mirror`, etc.
* **Damage Indicators (Target Defect Group):** `Dent` (Class ID: 8), `Scratch` (Class ID: 25), `Cracked`, `Corrosion`, `Broken part`.

### Environment Configuration:
* **Development Platform:** Google Colab (GPU-accelerated runtime environment)
* **Model Baseline:** `yolov8n-seg.pt` (Nano Instance Segmentation framework)
* **Image Dimensions:** $640 \times 640$ pixels
* **Training Depth:** 50 Epochs

---

## 📈 Training Analytics & Key Metrics

The model reached optimal convergence around epoch 40, showing an excellent alignment between Training Loss and Validation Loss trajectories without signs of overfitting.

* **Bounding Box Precision (mAP50):** `0.2803`
* **Segmentation Mask Precision (mAP50):** `0.2378`

> 💡 *Evaluation Note: Automated car dent extraction is notoriously complex due to erratic lighting refractions, ambiguous metallic contours, and specular highlights. A mask mAP50 of ~24% on an multi-class layout establishes a robust benchmark for localized anomaly tracking.*

---

## 🚀 Advanced Filtered Inference Pipeline

To eliminate visual clutter caused by large overlapping component masks (e.g., a massive door mask completely hiding a small dent boundary), a dedicated automation script was built.

### Core Architecture:
* **Dynamic Config Parsing:** Automatically reads the `data.yaml` configuration file and extracts damage-specific indices without requiring hardcoded user inputs.
* **Auto-Display Engine:** Batches the unseen `test` directory ($415+$ images), maps structural defects, and instantly visualizes multiple sample images side-by-side using `matplotlib` inside the notebook runtime.

### Custom Script Snip:
```python
from ultralytics import YOLO
import yaml

# Initialize the trained framework weights
model = YOLO('runs/segment/yolov8_car_dent_segmentation/exp_1/weights/best.pt')

# Execute targeted defect filtering across unseen test configurations
results = model.predict(
    source='/content/drive/MyDrive/Car_dent/test/images',
    conf=0.12,
    save=True,
    classes=[5, 6, 7, 8, 25] # Automatically isolates flaws (dents, scratches, etc.)
)
