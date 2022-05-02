#from inference_app import evaluate
import io
import json
import numpy as np

#from torchvision import models
#import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from flask_cors import CORS
import base64


app = Flask(__name__)
CORS(app)

ckpt_path='./models/hrnetw32.pth'
config_path='baseline.hrnetw32'
img_path="/home/bozcomlekci/Downloads/img"

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       # cls_map = evaluate(ckpt_path, config_path, False, img_path)
        #file = request.files['file']
        img = Image.open("./examples/example.png")
        # gets the image and reads in binary
        image_binary = request.files['image'].read()
        # converts the  binary to a base 64 string
        image_string = base64.b64encode(image_binary)
        #img_bytes = cls_map.read()
        #json_data = json.dumps(np.array(img).tolist())
        return image_string

@app.route('/predict', methods=['GET'])
def view():
    #cls_map = evaluate(ckpt_path, config_path, False, img_path)
    #file = request.files['file']
    incoming_img = request.json
    print("printed", incoming_img)
    img = Image.open("./examples/example.png")
    json_data = json.dumps(np.array(img).tolist())
    return jsonify({'img_id': 'map', 'img': json_data})

