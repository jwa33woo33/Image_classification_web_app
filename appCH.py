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
def mnist_view():
    return render_template('upload2.html')

@app.route('/mnist/pad')
def mnist_pad():
    return render_template('pad.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('upload2.html')
        fname = secure_filename(f.filename)
        os.makedirs('static', exist_ok = True)
        f.save(os.path.join('static', fname))
        im = Image.open(f, 'r')
        pix_val = list(im.getdata())
        pix_val = torch.Tensor(pix_val)
        pix_val = pix_val.view(1,1,28,28)

        weight_path = './weight/mnist.pth'
        model = LeNet().to(device)
        image, preds = evaluation(pix_val, weight_path, model)
        preds = int(preds)
        #print(fname)
        data = {'num': preds,'filename': fname}
        print(data)
        print(type(data))
        #num = preds
        #filename = fname
        print(data['num'], data['filename'])
        #print(type(num), type(filename))
        return render_template('upload2.html', num= data['num'], filename = data['filename'])
        #return data


if __name__ == '__main__':
	app.run(host='localhost', port=9000, debug = True)
