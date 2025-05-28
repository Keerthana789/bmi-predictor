import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, Response, redirect, url_for

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array


app = Flask(__name__)

# Load trained model (change path as needed)
model = load_model('vggface_model.h5')


# --- BMI Category Helper ---
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# --- Predict BMI from image frame ---
def predict_bmi(frame):
    img = Image.fromarray(frame)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction[0][0]


# --- Webcam Streaming ---
def generate_frames():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            predicted_bmi = predict_bmi(frame)
            category = get_bmi_category(predicted_bmi)

            cv2.putText(frame, f'BMI: {predicted_bmi:.2f} ({category})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()


# --- Homepage ---
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>BMI Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                margin: 0;
                padding: 0;
            }

            h1 {
                margin-top: 30px;
            }

            .container {
                display: flex;
                justify-content: space-around;
                align-items: flex-start;
                margin-top: 40px;
                width: 80%;
            }

            .box {
                flex: 1;
                margin: 0 20px;
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }

            iframe, img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }

            form {
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1>BMI Prediction</h1>
        <div class="container">
            <div class="box">
                <h2>Live Webcam</h2>
                <img src="/video_feed">
            </div>
            <div class="box">
                <h2>Upload Image</h2>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*"><br><br>
                    <input type="submit" value="Predict BMI">
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

# --- Webcam Feed Route ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Image Upload and Prediction ---
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return "No file selected."

        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_bmi = prediction[0][0]
        category = get_bmi_category(predicted_bmi)

        return f'''
            <h2>Predicted BMI: {predicted_bmi:.2f}</h2>
            <h3>Category: {category}</h3>
            <a href="/">Back to Home</a>
        '''

    return '''
        <h1>Upload Image for BMI Prediction</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <input type="submit" value="Predict BMI">
        </form>
    '''


# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
