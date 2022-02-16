from inference import evaluate

from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    #
    return 'Hello World!'

#ckpt_path='./models/hrnetw32.pth'
#config_path='baseline.hrnetw32'
#img_path="/home/bozcomlekci/Downloads/img"
#evaluate(ckpt_path, config_path, False, img_path)

