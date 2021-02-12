import sys
import glob

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    ])

def mnist_evaluation(inputs, weight_path, model):

    inputs = transform(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    # inputs = inputs.view(1,1,28,28)
    # inputs = inputs[:,-1,:,:]
    print(inputs.size)  
    model.load_state_dict(torch.load(weight_path, map_location=device)) #CPU Comparable. 
    model.eval()
    with torch.no_grad():
        
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)


    return inputs, preds

def quickdraw_evaluation(inputs, weight_path, model):
    
    inputs = transform(inputs)
    inputs = torch.unsqeeze(inputs, dim =0)
    #Evaluation function here
    return inputs, preds


def mnist_remove_transparency(img, bg_colour=(255,255,255)):

    #Only process if image has transparency
    if img.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        #Need to convert to RGBA if LA format due to a bug in PIL
        alpha  = img.convert('RGBA').split()[-1]

        bg = Image.new("RGBA",img.size, bg_colour+(255,))
        bg.paste(img, mask=alpha)
        return bg
    return img


def mnist_pad_data_preprocess(img):

    if img:
        img = mnist_remove_transparency(img).convert('L')
        img = np.array(img)
        img_resize =[]
        for i in range(0, 280, 10):
            for j in range(0, 280, 10):
                img_resize.append(int(np.average(img[i:i+10, j:j+10])))
        img_resize = np.array(img_resize).flatten().reshape((28,28))
        img_pil = Image.fromarray(np.uint8(img_resize), 'L')
        return img_pil
    return img

def class_dict_extraction(path, fileformat):
    #Look for all files with specific format and put it in as dictionary
    files = sorted(glob.glob(path + '/*.' +fileformat))

    #Add files in dictionary
    dic = class_dictionary()
    for key, value in enumerate(files):       
        #Get rid of the path and print label only
        label = value[len(path)+1 : -len(fileformat)-1]
        dic.add(key, label)
    return dic

class class_dictionary(dict):
    #create dictionary class
    #init function
    def __init__(self):
        self = dict()
    #Function to add key and value
    def add(self, key, value):
        self[key] = value


