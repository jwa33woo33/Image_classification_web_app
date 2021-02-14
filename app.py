import os
import glob
import json
import base64
import io

from PIL import Image
import numpy as np
import cv2

import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import torch
from torchvision.utils import save_image

from inference import mnist_evaluation, quickdraw_evaluation
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

@app.route('/quickdraw')
def quickdraw():
    return render_template('draw_quickdraw.html')

@app.route('/mnist/prediction', methods=['POST'])
def mnist_upload_image():
    if request.method == 'POST':
        mnist_f = request.files['file']
        mnist_fname = secure_filename(mnist_f.filename)
        #if f is empty string, nothing will happen.
        if mnist_fname =='':
            return redirect(url_for('mnist_view'))
        os.makedirs('static', exist_ok = True)
        mnist_f.save(os.path.join('static', mnist_fname))

        mnist_img = Image.open(mnist_f, 'r')

        mnist_img_display = mnist_img.resize((256, 256)) # To display large size image
        mnist_display = 'display_' + mnist_fname
        mnist_img_display.save(os.path.join('static', mnist_display))

        weight_path = './weight/mnist.pth'
        mnist_model = LeNet().to(device)
        mnist_img, mnist_preds = mnist_evaluation(mnist_img, weight_path, mnist_model)
        mnist_preds = int(mnist_preds)
    return render_template('upload_mnist.html', num=mnist_preds, filename=mnist_display)


@app.route('/mnist/draw_prediction', methods=['POST'])
def mnist_predict():
    if request.method == 'POST':

        mnist_draw = request.form['url']
        mnist_draw = mnist_draw[init_Base64:]
        mnist_draw_decoded = base64.b64decode(mnist_draw)

        # Fix later(to PIL version)
        # Conver bytes array to PIL Image  
        # imageStream = io.BytesIO(draw_decoded)
        # img = Image.open(imageStream)

        mnist_img = np.asarray(bytearray(mnist_draw_decoded), dtype="uint8")
        mnist_img = cv2.imdecode(mnist_img, cv2.IMREAD_GRAYSCALE)
        mnist_img = cv2.resize(mnist_img, (28,28), interpolation = cv2.INTER_AREA)
        mnist_img = Image.fromarray(mnist_img)

        weight_path = './weight/mnist.pth'
        mnist_model = LeNet().to(device)
        mnist_img, mnist_pred = mnist_evaluation(mnist_img, weight_path, mnist_model)

        mnist_pred = int(mnist_pred)
    return render_template('draw_mnist.html', prediction=mnist_pred)

@app.route('/quickdraw/prediction', methods=['POST'])
def quickdraw_predict():
    quickdraw_animal_map = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion', 'lobster', 'mermaid', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda', 'penguin', 'pig', 'rabbit', 'raccoon', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'teddy-bear', 'tiger', 'whale', 'zebra']

    if request.method == 'POST':
        quick_draw = request.form['url']
        quick_draw = quick_draw[init_Base64:]
        quick_draw_decoded = base64.b64decode(quick_draw)

        # Fix later(to PIL version)
        # Conver bytes array to PIL Image
        # imageStream = io.BytesIO(draw_decoded)
        # img = Image.open(imageStream)

        quick_img = np.asarray(bytearray(quick_draw_decoded), dtype="uint8")
        quick_img = cv2.imdecode(quick_img, cv2.IMREAD_GRAYSCALE)
        quick_img = cv2.resize(quick_img, (28,28), interpolation = cv2.INTER_AREA)
        print(quick_img)
        quick_img = Image.fromarray(quick_img)

        weight_path = './weight/quickdraw_90_animal.pth'
        quick_model = LeNet(num_classes=44).to(device)
        quick_img, quick_pred = quickdraw_evaluation(quick_img, weight_path, quick_model)

        quick_pred = int(quick_pred)
        print(quick_pred)
        quick_label = quickdraw_animal_map[quick_pred]
    return render_template('draw_quickdraw.html', prediction=quick_label)

for f_ext in file_extensions:  # Delete file after display
    for img_file in glob.glob(f'./static/*.{f_ext}'):
        os.remove(img_file)

if __name__ == '__main__':
	app.run(host='localhost', port=9000, debug = True)

