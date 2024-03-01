from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model("visualizations\model.h5")

# Define the classes
classes = ['Parasitized', 'Uninfected']  

# Function to preprocess the image
def preprocess_image(image_path):

    img = image.load_img(image_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict(image_path):
    img = preprocess_image(image_path)
    result = model.predict(img)
    prediction = classes[np.argmax(result)]
    return prediction

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and display prediction
@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = image_file.filename
            print("image_path=", image_path )
            image_file.save(image_path)
            prediction = predict(image_path)
            return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
