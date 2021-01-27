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
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	if file and img_files(file.filename):
		filename = secure_filename(file.filename)
		image_source = request.files['file'].stream

		# image_path = './static/images/' + filename
		weight_path = './weight/mnist.pth'

		model = LeNet().to(device)
		image, preds = evaluation(image_source, weight_path, model)

		image = image[0]
		save_image(image, './static/images/' + filename)

		preds = int(preds.cpu())

		print('upload_image filename: ' + filename)

		flash(f'Prediction: {preds}')
		return render_template('upload.html', filename=filename)
	else:
		flash('Image extension must be -> png, jpg, jpeg, gif')
		return redirect(request.url)



    

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='./images/' + filename), code=301)



@app.route('/asdf') # 이걸 추가하면 http://localhost:9000/asdf 에 test1이 프린트됨 ㅇㅇ
def test1():
    return 'test1'

if __name__ == '__main__':
	app.run(host='localhost', port=9000, debug = True)