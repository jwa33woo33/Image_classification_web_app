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
    
    model.load_state_dict(torch.load(weight_path))

    # print('\n\n\n\n\n', inputs.shape, 'ya!!!!!!!!!!!!!!!!!!!!!!\n\n\n')

    model.eval()
    with torch.no_grad():
        
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)


    return inputs, preds


