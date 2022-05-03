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
        cls_map, ratio_dict = evaluate(ckpt_path, config_path, False, img=img)
        #file = request.files['file']
        # gets the image and reads in binary
        # converts the  binary to a base 64 string
        buffered = io.BytesIO()
        cls_map.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        #json_data = json.dumps(np.array(img).tolist())
        return jsonify({'img_id': 'map', 'img': img_str, 'ratio_dict': ratio_dict})

@app.route('/predict', methods=['GET'])
def view():
    cls_map, ratio_dict = evaluate(ckpt_path, config_path, False, img_path)
    #file = request.files['file']
    buffered = io.BytesIO()
    cls_map.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({'img_id': 'map', 'img': img_str, 'ratio_dict': ratio_dict})

