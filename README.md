# Gesture Recognition System

🚀 **GestureRecognitionSystem** is a deep learning project focused on **human behavior modeling**:

- **Hand Gesture Recognition (HGR)** from video.

This work combines **signal processing** and **state-of-the-art neural networks** (CNNs, ViViTs) to create systems capable of more natural and multimodal **human-computer interaction**.

---

## ✨ Features

- 📹 **Hand Gesture Recognition**
    - Based on the **IPN Hand dataset** (13 gesture classes for screen control).
    - Uses **MediaPipe** for hand landmark extraction.
    - Models: **CNN** and **Video Vision Transformer (ViViT)**.
    - Achieved **88.1% accuracy** with ViViT + landmarks, a result competitive with state-of-the-art while being **6× smaller**.

---

## 📊 Results Summary

| Model                       | Score          | Params |
| --------------------------- | -------------- | ------ |
| MediaPipe Landmarks + ViViT | **88.1% acc.** | 3.66M  |
| MediaPipe Landmarks + CNN   | 85.2% acc.     | 22.87M |

- **Hand Gesture Recognition (HGR):** Achieved **88.1% accuracy**, comparable to state-of-the-art methods while using a model **6× smaller**.

> Example confusion matrices and figures can be found in the [documentation](./docs).

---

## 📂 Datasets

- **[IPN Hand](https://github.com/GibranBenitez/IPN-Hand)** – 4,218 gesture instances across 13 classes for screen interaction.

---

## 🛠 Tools & Frameworks

- **Python**, **PyTorch**, **Keras**
- **MediaPipe** (hand landmark extraction)
- **Weights & Biases (W&B)** for experiment tracking

---

## 📥 Installation

```bash
git clone https://github.com/jossebv/GestureRecognitionSystem.git
cd GestureRecognitionSystem
pip install -r requirements.txt
```

---

## 🔮 Future Work

- Data augmentation for ViViT to improve generalization.
- More efficient architectures for deployment on resource-constrained devices.
