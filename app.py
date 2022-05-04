from inference_app import evaluate
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

from flask_cors import CORS
import base64

import pymongo
mongoclient = pymongo.MongoClient('mongodb://localhost:27017')
#mongoclient.drop_database('test')
#mongoclient.drop_database('segmentation')
db = mongoclient['segmentation']

app = Flask(__name__)
CORS(app)

ckpt_path='./models/hrnetw32.pth'
config_path='baseline.hrnetw32'
img_path="/home/bozcomlekci/Downloads/img"

@app.route('/')
def hello():
    if request.method == 'GET':
        cls_map, ratio_dict = evaluate(ckpt_path, config_path, False, img_path)
        buffered = io.BytesIO()
        cls_map.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({'img_id': 'map', 'img': img_str, 'ratio_dict': ratio_dict})



@app.route('/retrieve_collection', methods=['POST, GET'])
def retrieve_cols():
    if request.method == 'POST':
        cname = request.files['cname']
        cursor = db[cname].find({})
        return cursor
    if request.method == 'GET':
        cursor = db['test'].find({})
        return cursor

@app.route('/list_collections', methods=['POST, GET'])
def list_cols():
    if request.method == 'POST':
        clist = db.list_collection_names()
        return jsonify(clist)
    if request.method == 'GET':
        clist = db.list_collection_names()
        return jsonify(clist)

@app.route('/predict', methods=['POST'])
def predict():
    image_binary = request.files['image'].read()
    img = Image.open(io.BytesIO(image_binary))
    cls_map, ratio_dict = evaluate(ckpt_path, config_path, False, img=img)
    buffered = io.BytesIO()
    cls_map.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(ratio_dict)
    data = jsonify({'img': img_str, 'ratio_dict': ratio_dict})
    cname = request.form['cname']
    if cname != "":
        clist = db.list_collection_names()
        if cname not in clist:
            db.add_collection(cname)
        db[cname].insert_one({'img': img_str, 'ratio_dict': ratio_dict})
    return data


