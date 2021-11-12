# General libs
import pathlib
import logging
import os
import datetime
import confuse
import pickle
import requests
from flask import Flask, request, jsonify, Response
from azure.storage.blob import BlobServiceClient
# Image and texts processing libs
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Additional libs
import numpy as np
from io import BytesIO
from PIL import Image

SIZE = 224
THRESHOLD = 0.8
MAX_LENGTH = 7

root_dir = os.getcwd()
cv_model_path = os.path.join(root_dir, "cv_model")
nlp_model_path = os.path.join(root_dir, "nlp_model")

config_file_path = os.path.join(root_dir, "config_default.yaml")
tokenizer_path = os.path.join(root_dir, "tokenizer.pickle")

with open(os.path.join(root_dir, "tokenizer.pickle"), 'rb') as handle:
    tokenizer = pickle.load(handle)

config = confuse.Configuration('producttypeclassification', __name__)
config.set_file(config_file_path)

cv_product_types_list = config['product_type_list']['cv_product_list'].get()
nlp_product_types_list = config['product_type_list']['nlp_product_list'].get()

# Create folder name and file name for log file in Blob storage
current_time = str(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z"))
hour = current_time.split("T")[1][:2]
folder_name = current_time.split("T")[0]
log_file_name = (hour + ".log").replace(":", "")
directory_name = "PythonLogs"
sub_directory_name = "ProductTypeClassification"
blob_name = directory_name + "/" + sub_directory_name + "/" + folder_name + "/" + log_file_name

# Set up file and smtp loggers
logger_tofile = logging.getLogger(__name__)
logger_tofile.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger_tofile.addHandler(file_handler)


def logs_to_blob():
    azure_storage_connections_str = config['AzureStorage']['azure_storage_connections_str'].get()
    container_name = config['AzureStorage']['container_name'].get()

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connections_str)

    upload_file_path = log_file_name
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Upload the created file
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.shutdown()
    pathlib.Path(upload_file_path).unlink()


cv_model = tf.keras.models.load_model(cv_model_path, compile=False)
nlp_model = tf.keras.models.load_model(nlp_model_path, compile=False)


def predict_image_from_url(image_url):
    product_type = ""
    try:
        response = requests.get(image_url, timeout=1)
        image = Image.open(BytesIO(response.content))
        image = np.array(image)
        tensor = tf.convert_to_tensor(image)
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        tensor = tf.image.resize(tensor, (SIZE, SIZE))
        tensor = tf.expand_dims(tensor, 0)
        predicted_vector = cv_model.predict(tensor)
        # probability = np.amax(predicted_vector)
        index = np.argmax(predicted_vector)
        product_type = str((cv_product_types_list[index]))
    except:
        pass
    # logs_to_blob()
    return product_type


def predict_product_from_txt(product_name):
    sequence_list = []
    try:
        nlp_label = ""
        sequence_list.append(product_name)
        seq = tokenizer.texts_to_sequences(sequence_list)
        padded = pad_sequences(seq, maxlen=MAX_LENGTH)
        pred = nlp_model.predict(padded)
        nlp_label = nlp_product_types_list[np.argmax(pred[0]) - 1]
        # probability = np.amax(pred)
    except:
        pass
    return nlp_label


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    output_json = {}
    if request.authorization and request.authorization.username == 'User' and request.authorization.password == 'Password':
        if request.method == "POST":
            try:
                input_json = request.get_json()
            except ValueError:
                pass
        nlp_label = predict_product_from_txt(input_json["product_name"])
        cv_label = predict_image_from_url(input_json["image_url"])
        if nlp_label == cv_label:
            output_json["product_type"] = cv_label
            return jsonify(output_json)
        else:
            return Response(status=204)


if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')
