# Diabetic Retinopathy Early Detection System

## Project Overview

This project offers a deep learning-based web application for the detection of Diabetic Retinopathy from retinal images. The application utilizes various state-of-the-art architectures of convolutional neural networks to enhance the accuracy of predictions. The application helps in early diagnosis by determining the level of severity.

---

## Models Used

The following deep learning models have been implemented and tested:

* EfficientNetB0
* VGG16
* ResNet
* Xception
* DenseNet
* Inception

The most accurate model is used for prediction.

---

## Features

* Image processing using OpenCV (CLAHE and edge detection)
* Multi-model testing and comparison
* Web application development using Flask
* Severity level classification:

  * Mild
  * Moderate
  * Severe
  * Proliferative Diabetic Retinopathy
  * No Diabetic Retinopathy
* Efficient and easy-to-use interface

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Flask
* HTML, CSS

---

## Project Structure

diabetic_retinopathy_application/
│── static/              # Static files (CSS, images)
│── templates/           # HTML templates
│── app.py               # Main Flask application
│── model_predict.py     # Prediction logic
│── config.py            # Configuration settings
│── notebooks/           # Model training and experimentation
│── README.md

---

## Methodology

1. The user uploads a retinal image through the web interface. 2. The image is preprocessed using techniques such as CLAHE and edge detection. 3. Multiple deep learning models are evaluated. 4. The best-performing model predicts the severity level. 5. The result is displayed to the user. --- ## Model Evaluation The models are evaluated using the following metrics: - Accuracy - Precision - Recall - F1 Score --- ## Note Model weight files (.h5) are not included in this repository due to GitHub size limitations. --- ## How to Run the Project git clone https://github.com/bunty5017144/Diabetic_Retinopathy_Early_Detection.git cd diabetic_retinopathy_application pip install -r requirements.txt python app.py --- ## Future Enhancements - Deployment on cloud platforms - Mobile-responsive interface - Integration of ensemble learning techniques - Performance optimization and real-time analytics --- ## Author Hareen Chowdary Bangalore Dayananda Sagar Academy of Technology and Management --- ## License This project is intended for academic and research purposes. ::: --- #  What to do now 1. Create `README.md` 2. Paste this content 3. Run: ```bash git add README.md git commit -m "Added README" git push ``` --- If you want, I can next help you: - add a model comparison table - improve this for ATS/resume impact - prepare how to explain this in interviews Just tell me. ::contentReference[oaicite:0]{index=0}
