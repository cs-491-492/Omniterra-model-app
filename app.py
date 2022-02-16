from inference_app import evaluate
import io
import json
import numpy as np

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

ckpt_path='./models/hrnetw32.pth'
config_path='baseline.hrnetw32'
img_path="/home/bozcomlekci/Downloads/img"

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cls_map = evaluate(ckpt_path, config_path, False, img_path)
        #file = request.files['file']
        img_bytes = cls_map.read()
        return jsonify({'img_id': 'map', 'img': cls_map})

@app.route('/predict', methods=['GET'])
def view():
    cls_map = evaluate(ckpt_path, config_path, False, img_path)
    #file = request.files['file']
    json_data = json.dumps(np.array(cls_map).tolist())
    return jsonify({'img_id': 'map', 'img': json_data})



