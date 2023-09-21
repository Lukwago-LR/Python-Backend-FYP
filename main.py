import os
import numpy as np
import tensorflow as tf
import tensorflow_hub
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# Define the batch size, 32 is a good start
BATCH_SIZE = 32

# Define image size
IMG_SIZE = 224

app = Flask(__name__)
CORS(app)


@app.route("/")
@cross_origin()
def home():
    return jsonify("Welcome")


# Find the unique label values
unique_labels = ['Bengin', 'Malignant', 'Normal']


# Turn prediction probabilities into their respective labels
def get_prediction_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_labels[np.argmax(prediction_probabilities)]


# Create a function to load a trained model
def load_model(model_path):
    """
     Loads a saved model
    """

    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer": tensorflow_hub.KerasLayer}
                                       )
    return model


# Create a function for preprocessing images
def process_image(image_path):
    """
  Takes am image file path and turns the image into a tensor
  """
    # Read in an image file
    image = tf.io.read_file(image_path)

    # Turn the jpg image into numerical tensor with 3 colour channels (Red, Green,Blue)
    image = tf.image.decode_jpeg(image, channels=3)

    # convert the colour channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to our desired value (244,244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def get_image_label(image_path, label):
    """
  Takes image path and associated label
  Processes the image and returns a tuple of (image, label).
  """
    image = process_image(image_path)
    return image, label


# Create a function to turn data into batches
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
  Creates batches of data out of the image (X) and label (y) pairs
  Shuffles the data if it's training data but doesn't shuffle if it's validation data
  Also accepts test data as input (no labels):
  """
    # if the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # only filepaths (no labels)
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch
        # if data is a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  # filepaths
                                                   tf.constant(y)))  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch
    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  # filepaths
                                                   tf.constant(y)))  # labels
        # Shuffling path_names and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # Create (image,label) tuples (This also turns the image paths into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
        return data_batch


def classify_image(image_path_list):
    # Turn the custom image into batch datasets
    custom_data = create_data_batches(image_path_list, test_data=True)
    print(custom_data)

    # Loading the full model
    loaded_full_model = load_model("models/20230130-17341675100040-full-image-set-mobienetv2-Adam.h5")

    # Make predictions on the custom data
    custom_predictions = loaded_full_model.predict(custom_data)

    # Get custom image prediction labels
    custom_prediction_labels = [get_prediction_label(custom_predictions[i]) for i in range(len(custom_predictions))]

    # return the predicted label_list
    return custom_prediction_labels


@app.route("/predict", methods=["GET"])
def post_new_image():
    # let now access the image for our machine learning processing
    root_image_path = "uploads/"
    image_filenames = [root_image_path + file_name for file_name in os.listdir(root_image_path)]

    # Classification of the image
    predicted_label_list = classify_image(image_filenames)

    return jsonify(response={"success": predicted_label_list})


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file.save('uploads/' + file.filename)
        return "File saved successfully"


if __name__ == '__main__':
    app.run(debug=True)
