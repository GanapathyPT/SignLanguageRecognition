import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response

app = Flask(__name__)
model = load_model('./modal/ASL.h5')
labels = [
"A",
"B",
"C",    
"D",
"del",
"E",
"F",
"G",
"H",
"I",
"J",
"K",
"L",
"M",
"N",
"nothing",
"O",
"P",
"Q",
"R",
"S",
"space",
"T",
"U",
"V",
"W",
"X",
"Y",
"Z",
]


def generate_frames():
    camera = cv2.VideoCapture(-1)
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize the frame to the input shape of the model (64x64x3)
        resized = cv2.resize(frame, (224, 224))
        
        # Normalize the pixel values
        normalized = resized / 255.0
        
        # Reshape the image to (1, 64, 64, 3) to match the input shape of the model
        reshaped = normalized.reshape((1, 224, 224, 3))
        
        # Pass the image through the model and get the predicted class probabilities
        probs = model.predict(reshaped)[0]
        
        # Get the class label with the highest probability
        label = labels[np.argmax(probs)]
        
        # Display the label on the frame
        cv2.putText(frame, label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        rear, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)