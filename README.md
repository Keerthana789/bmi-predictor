# Facial BMI Predictor

A deep learning web app that predicts a person's Body Mass Index (BMI) from facial images. Built using VGG-Face with Flask for deployment, this project supports both webcam input and image uploads.

---

## 🚀 Features

- Predict BMI from a **live webcam** or **uploaded image**
- Classifies into BMI categories: **Underweight**, **Normal**, **Overweight**, **Obese**
- Real-time prediction using a pretrained deep learning model
- Clean, responsive web interface

---

## 🧠 Why VGG-Face?

I evaluated **ResNet50V2**, **VGG16**, and **VGG-Face** on the same BMI dataset.  
**VGG-Face** outperformed the others due to its domain-specific training on facial images, capturing visual cues like facial adiposity more effectively.

### ✅ Final Test Evaluation (VGG-Face):
- **RMSE**: 7.57  
- **MAE**: 5.63  

---

## 🧪 Model Training Summary

The model was trained using a cleaned dataset of facial images and BMI labels, with a consistent pipeline across all base architectures:

- Images resized to 224×224 and normalized
- Data split: 80% training, 20% validation
- Used transfer learning with frozen base layers
- Dense regression head: [Flatten → Dense(1024) → Dropout → Dense(512) → ... → Output]
- Loss: Mean Squared Error (MSE), Optimizer: Adam

Three models were evaluated:
- **ResNet50V2**
- **VGG16**
- **VGG-Face**

The training notebook (`BMI_Prediction.ipynb`) contains detailed logs, plots, and final metrics.

---


## 💾 Download the Model

📦 Download the trained VGG-Face BMI model from Google Drive and place it in the project root:

👉 [Download Best model vggface_model.h5 from Google Drive](https://drive.google.com/file/d/1Oq-_e-g_Vs9SEHktS2kyr-imOWLnArSz/view?usp=sharing)

> ⚠️ Not included in this repo due to GitHub file size limits.

---

## ▶️ Run the App

```bash
pip install -r requirements.txt
python app.py


