from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
from tensorflow import keras
import tensorflow as tf
import base64
import cv2

app = Flask(__name__)

# Load the model
loaded_model = keras.models.load_model('gan_model')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def preprocess(image):
    # Load the low-light image
    image = cv2.imread('low_light_image.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale (if necessary)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter to detect edges
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return image
    
def preprocess_image(original_image):
    
    resized_image = original_image.resize((600, 400))
    # Convert the image to array
    image_array = keras.preprocessing.image.img_to_array(resized_image)
    # Normalize the image
    image_array = image_array.astype("float32") / 255.0
    # Expand dimensions to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def infer(image_array):
   
    output_array = loaded_model(image_array)
    output_image = tf.cast((output_array[0] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image
    

# Function to enhance a single image
def enhance_single_image(image):
    image_array = preprocess_image(image)
    # Perform inference
    enhanced_image = infer(image_array)
    return enhanced_image

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    # Check if request contains file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if file is an image
    if file and allowed_file(file.filename):
        # Read image
        img = Image.open(io.BytesIO(file.read()))

        # Enhance image
        enhanced_image = enhance_single_image(img)

        # Convert enhanced image to bytes
        img_byte_array = io.BytesIO()
        enhanced_image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Convert bytes to base64 string
        enhanced_base64_string = base64.b64encode(img_byte_array).decode('utf-8')

        return jsonify({'enhanced_image': enhanced_base64_string})

    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
