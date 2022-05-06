from inference_app import evaluate
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from bson import json_util
from flask_cors import CORS
import base64 
from utils.image_utils import add_transparent_layer
from utils.data_utils import ratio_to_category, get_images

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

@app.route('/retrieve_collection', methods=['POST'])
def retrieve_cols():
    cname = request.form['cname']
    cursor = db[cname].find({})  
    collection = [item for item in cursor] 
    keys = list(collection[0].keys())
    newKeys = keys[1:]
    new_collection = []
    for item in collection:
        new_collection.append({x:item[x] for x in newKeys})
    plot_array = ratio_to_category(new_collection)
    imgs = get_images(new_collection)
    gallery_dict = {'imgs':imgs, 'plot_array':plot_array}
    print(gallery_dict)
    return jsonify(gallery_dict)
    
    return jsonify(new_collection)
   
#  if request.method == 'GET':
       # cursor = db['test'].find({})
        #return cursor
@app.route('/list_collections', methods=['GET'])
def list_cols():
    clist = db.list_collection_names()
    return jsonify(clist)

#     if request.method == 'POST':
#        clist = db.list_collection_names()
#        return jsonify(clist)

@app.route('/predict', methods=['POST'])
def predict():
    image_binary = request.files['image'].read()
    img = Image.open(io.BytesIO(image_binary))
    cls_map, ratio_list = evaluate(ckpt_path, config_path, False, img=img)
    #ratio_list = [{'x': 'Water', 'y': 1}]
    buffered = io.BytesIO()
    img_painted = add_transparent_layer(img, cls_map)
    img_painted.save(buffered, format="PNG")
    #img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data = jsonify({'img': img_str, 'ratio_dict': ratio_list})
    cname = request.form['cname']
    if cname != "":
        clist = db.list_collection_names()
        if cname not in clist:
            db.create_collection(cname)
        db[cname].insert_one({'img': img_str, 'ratio_dict': ratio_list})
    return data


