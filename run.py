
#import libraries
import os
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO
import base64
import io
import time
from collections import OrderedDict
import json


from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import plotly
import chart_studio.plotly as py 
import plotly.graph_objs as go
from simple_cnn import SimpleCNN


from flask import Flask
from flask import render_template, request


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import image_utils
from my_image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


label_dict = {
    0: 'apple',
    1: 'banana',
    2: 'car',
    3: 'cat',
    4: 'dog',
    5: 'house',
    6: 'tree',
    7: 'bicycle',
    8: 'fish',
    9: 'chair'
}


def load_model(filepath = 'checkpoint_classifier.pth'):
    
    print("Loading model from {} \n".format(filepath))

    checkpoint = torch.load(filepath, map_location='cpu')
    model = SimpleCNN(len(label_dict))  # Use your CNN class
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_prediction(model, input_np):
    """
    Preprocess and predict using the trained CNN.
    The input_np should be 0–1, we normalize it to -1–1 like in training.
    """
    if input_np.ndim == 2:
        input_np = input_np[np.newaxis, :, :]

    input_np = (input_np - 0.5) / 0.5

    input_tensor = torch.from_numpy(input_np).unsqueeze(0).float()  # (1,1,28,28)

    with torch.no_grad():
        logits = model(input_tensor)
        ps = F.softmax(logits, dim=1).cpu().numpy()

    label = int(np.argmax(ps))
    label_name = label_dict[label]
    return label,label_name,ps


def view_classify(img, preds):

    preds = preds.squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.axis('off')
    ax2.set_aspect(0.1)
    # Reverse both data and tick labels
    
    ax2.barh(np.arange(10), preds)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels([label_dict[i] for i in range(10)], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    ts = time.time()
    plt.savefig('prediction' + str(ts) + '.png')

app = Flask(__name__)


model = load_model()
model.eval() 


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/go/<dataURL>')
def pred(dataURL):

    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')
    image_b64_str = dataURL
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    ts = time.time()
    img = img.convert("RGBA")
    image_cropped = crop_image(img)
    image_normalized = normalize_image(image_cropped) 


    img_rgb = convert_to_rgb(image_normalized)

    image_np = convert_to_np(img_rgb)

    label, label_num, preds = get_prediction(model, image_np)
    print("This is a {}".format(label_num))

    view_classify(image_np, preds)

    #plotly visualization
    graphs = [
        {
            'data': [
                go.Bar(
                        x=preds.ravel().tolist()[::-1],
                        y=list(label_dict.values())[::-1],
                        orientation = 'h')
            ],

            'layout': {
                'title': 'Class Probabilities',
                'yaxis': {
                    'title': "Classes"
                },
                'xaxis': {
                    'title': "Probability",
                }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render the hook.html
    return render_template(
        'hook.html',
        result = label_num, 
        ids=ids, 
        graphJSON=graphJSON, 
        dataURL = dataURL 
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
