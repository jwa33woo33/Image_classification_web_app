import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

import torch
from torchvision.utils import save_image

from inference import evaluation
from model import LeNet

from keras.models import model_from_json
import json

file_extensions = set(['png', 'jpg', 'jpeg', 'gif'])

save_img_path = './static/images/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

app.secret_key = "mother fucker"
app.config['UPLOAD_FOLDER'] = save_img_path
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



def img_files(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in file_extensions
	
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/mnist/upload')
def mnist_upload():
    return render_template('upload.html')

@app.route('/mnist/pad')
def mnist_pad():
    return render_template('pad.html')

@app.route('/mnist/upload/pred', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('/mnist/upload'))
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(url_for('/mnist/upload'))

    if file and img_files(file.filename):
        filename = secure_filename(file.filename)
        image_source = request.files['file'].stream

        #image_path = './static/images/' + filename
        weight_path = './weight/mnist.pth'

        model = LeNet().to(device)
        image, preds = evaluation(image_source, weight_path, model)

        image = image[0]
        save_image(image, './static/images/' + filename)

        preds = int(preds.cpu())

        print('upload_image filename: ' + filename)

        flash(f'Prediction: {preds}')
        #return render_template('upload.html', filename=filename)        
        return url_for('mnist/upload/success', filename = filename)
    else:
        flash('Image extension must be -> png, jpg, jpeg, gif')
        return redirect(request.url)

def pad_image():
        #pad output:
        model = LeNet().to(device)
        weight_path = './weight/mnist.pth'
        array_raw = json.loads(request.data)['array']
        array_raw_np = np.array(array_raw)
        array_processed = []
        for i in range(0,280, 10):
            for j in range(0, 280, 10):
                array_processed.append(int(np.average(array_raw[i:i+10, j:j+10])))
        pixels = np.array(array_processed).flatten().reshape((1,28,28,1))
        
        image, pred = evaluation(pixels, weight_path, model)
        json_respond = json.dumps({"digit": "{}".format(pred)})
        return json_respond

    

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='./images/' + filename), code=301)



@app.route('/asdf') # 이걸 추가하면 http://localhost:9000/asdf 에 test1이 프린트됨 ㅇㅇ
def test1():
    return 'test1'

if __name__ == '__main__':
	app.run(host='localhost', port=9000, debug = True)
