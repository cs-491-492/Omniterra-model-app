from inference_app import evaluate
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
        image_binary = request.files['image'].read()
        img = Image.open(io.BytesIO(image_binary))
        cls_map = evaluate(ckpt_path, config_path, False, img=img)
        #file = request.files['file']
        # gets the image and reads in binary
        # converts the  binary to a base 64 string
        buffered = io.BytesIO()
        cls_map.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        #json_data = json.dumps(np.array(img).tolist())
        return img_str

@app.route('/predict', methods=['GET'])
def view():
    #cls_map = evaluate(ckpt_path, config_path, False, img_path)
    #file = request.files['file']
    incoming_img = request.json
    print("printed", incoming_img)
    img = Image.open("./examples/example.jpeg")
    json_data = json.dumps(np.array(img).tolist())
    return jsonify({'img_id': 'map', 'img': json_data})

