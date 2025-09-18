# facemask-detectionn
# 😷 Face Mask Detection using CNN & Deep Learning

## 📌 Project Overview
This project is a **Face Mask Detection System** built using **Convolutional Neural Networks (CNN)** and Deep Learning.  
It classifies whether a person in an image/video frame is **wearing a mask** or **not wearing a mask**.

---

## 📊 Dataset
- **Dataset Used:** [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- **Classes:** 
  - `with_mask`
  - `without_mask`
- **Data Preprocessing:**
  - Images resized to (224x224)
  - Normalized pixel values (0-1)
  - Applied Data Augmentation (flip, rotation, zoom)

---

## 🧠 Model Architecture
- **Model Type:** Convolutional Neural Network (CNN)
- **Layers:**
  - Conv2D + ReLU
  - MaxPooling
  - Dropout (to prevent overfitting)
  - Dense layers
  - Softmax output (2 classes)
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  

---

## 🏋️ Training
- **Epochs:** 20
- **Batch Size:** 32
- **Callbacks:**
  - EarlyStopping (monitor validation loss)
  - ReduceLROnPlateau (adjust learning rate dynamically)

---

## 📈 Results
| Metric       | Value |
|-------------|-------|
| Training Accuracy | 98% |
| Validation Accuracy | 96% |

- Confusion matrix shows balanced performance on both classes.

---

## 🎥 Real-Time Detection
- Uses **OpenCV** to capture webcam video.
- Performs face detection using Haar cascades or DNN face detector.
- Classifies each detected face as:
  - ✅ `Mask`
  - ❌ `No Mask`
- Draws colored bounding boxes:
  - Green → Mask
  - Red → No Mask

---

## 🖥️ How to Run

### 1️⃣ Install Requirements
```bash
pip install tensorflow opencv-python numpy matplotlib
)
