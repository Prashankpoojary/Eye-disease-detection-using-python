from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model(r"C:\Users\Prashank Poojary\Documents\mini project\application\mini.keras")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']

    # Read the uploaded image file
    img = image_file.read()

    # Convert the image to a numpy array
    img = image.img_to_array(image.load_img(io.BytesIO(img), target_size=(200, 200)))

    # Expand the dimensions to match the expected input shape
    img = np.expand_dims(img, axis=0)

    # Predict using the model
    prediction = model.predict(img)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        prediction_text = "cataract"
    elif predicted_class == 1:
        prediction_text = "diabetic_retinopathy"
    elif predicted_class == 2:
        prediction_text = "glaucoma"
    elif predicted_class == 3:
        prediction_text = "normal"
    else:
        prediction_text = "Error"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
