import sys

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

def evaluation(img_source, weight_path, model):

    inputs = Image.open(img_source)

    inputs = transform(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)

    model.load_state_dict(torch.load(weight_path))

    with torch.no_grad():
        model.eval()    # set the model to evaluation mode (dropout=False)

        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)
        # correct_prediction = torch.argmax(prediction, 1) == labels
        # accuracy = correct_prediction.float().mean()
        # print('Accuracy:', accuracy.item())

    return inputs, preds


