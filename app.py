import os
import glob
import json
import base64
import io

from PIL import Image
import numpy as np

import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import torch
from torchvision.utils import save_image

from inference import evaluation, mnist_pad_data_preprocess
from model import LeNet

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

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
def main_page():
	return render_template('main_page.html')


@app.route('/mnist/upload')
def mnist_view():
    return render_template('upload_mnist.html')


@app.route('/mnist/pad')
def mnist_pad():
    return render_template('draw_mnist.html')


@app.route('/mnist/prediction', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        #if f is empty string, nothing will happen.
        if fname =='':
            return redirect(url_for('mnist_view'))
        os.makedirs('static', exist_ok = True)
        f.save(os.path.join('static', fname))
        img = Image.open(f, 'r')

        weight_path = './weight/mnist.pth'
        model = LeNet().to(device)
        img, preds = evaluation(img, weight_path, model)
        preds = int(preds)
        data = {'num': preds,'filename': fname}
    return render_template('upload_mnist.html', num= data['num'], filename = data['filename'])


@app.route('/mnist/draw_prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Access the image
        draw = request.form['url']
        # Removing the useless part of the url.
        draw = draw[init_Base64:]
        # Decode to bytes array
        draw_decoded = base64.b64decode(draw)
        # Conver bytes array to PIL Image
        imageStream = io.BytesIO(draw_decoded)
        img = Image.open(imageStream)
        # Data Preprocessing for Evaluation
        img_final = mnist_pad_data_preprocess(img)
        weight_path = './weight/mnist.pth'
        model = LeNet().to(device)
        img, preds = evaluation(img_final, weight_path, model)
        preds = int(preds)

    return render_template('draw_mnist.html', prediction=preds)

for f_ext in file_extensions:  # Delete file after display
    for img_file in glob.glob(f'./static/*.{f_ext}'):
        os.remove(img_file)

if __name__ == '__main__':
	app.run(host='localhost', port=9000, debug = True)

