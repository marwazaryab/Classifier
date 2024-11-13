from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# intialize flash
app = Flask(__name__)

# load trained model which is the CIFHAR10 model
model = load_model("cifar10_classifier_model.h5")

# Define class names for the CIFAR-10 data sheet
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
 
# route for homepage (Render HTML form to upload an image to match)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Convert FileStorage to a byte stream 
    img = image.load_img(io.BytesIO(file.read()), target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize the image
    
    # perdiction through aray of images
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    predicted_class = class_names[class_idx]
    
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
