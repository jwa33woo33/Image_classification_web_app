import sys

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    ])

def evaluation(inputs, weight_path, model):

    inputs = transform(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    # inputs = inputs.view(1,1,28,28)
    # inputs = inputs[:,-1,:,:]
    print(inputs.size)    
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)


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

