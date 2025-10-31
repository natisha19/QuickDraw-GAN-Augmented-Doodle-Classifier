#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May  10 2019

@author: Aleksandra Deis

Script which runs flask web application for Qick Draw

"""
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

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


# import Plotly
import plotly
import chart_studio.plotly as py # import plotly.plotly as py
import plotly.graph_objs as go
from simple_cnn import SimpleCNN

# import Flask
from flask import Flask
from flask import render_template, request

# import image processing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import image_utils
from my_image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Dictionary with label codes
# label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
#               5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword'}
# New class list (in the order your model was trained)
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


def load_model(filepath = 'C:\Users\Natisha\Downloads\checkpoint_classifier.pth'):
    """
    Function loads the model from checkpoint.

    INPUT:
        filepath - path for the saved model

    OUTPUT:
        model - loaded pytorch model
    """

    print("Loading model from {} \n".format(filepath))

    checkpoint = torch.load(filepath, map_location='cpu')
    model = SimpleCNN(len(label_dict))  # Use your CNN class
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_prediction(model, input):
    """
    Function to get prediction (label of class with the greatest probability).

    INPUT:
        model - pytorch model
        input - (numpy) input vector

    OUTPUT:
        label - predicted class label
        label_name - name of predicted class
    """
    # Convert input to tensor
    input = torch.from_numpy(input).float()
    input = input.unsqueeze(0)  # (1, 1, 28, 28) if shape is (1, 28, 28)
    with torch.no_grad():
        logits = model(input)
        ps = F.softmax(logits, dim=1)
    preds = ps.numpy()
    label = np.argmax(preds)
    label_name = label_dict[label]
    return label, label_name, preds

def view_classify(img, preds):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (numpy) image file
        preds - (numpy) predicted probabilities for each class
    """
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

# load model
model = load_model()
model.eval() # set to evaluation

# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    # render web page
    return render_template('index.html')

@app.route('/go/<dataURL>')
def pred(dataURL):
    """
    Render prediction result.
    """

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    # get the base64 string
    image_b64_str = dataURL
    # convert string to bytes
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    # open Image with PIL
    img = Image.open(image_data)

    # save original image as png (for debugging)
    ts = time.time()
    #img.save('image' + str(ts) + '.png', 'PNG')

    # convert image to RGBA
    img = img.convert("RGBA")

    # preprocess the image for the model
    image_cropped = crop_image(img) # crop the image and resize to 28x28
    image_normalized = normalize_image(image_cropped) # normalize color after crop

    # convert image from RGBA to RGB
    img_rgb = convert_to_rgb(image_normalized)

    # convert image to numpy
    image_np = convert_to_np(img_rgb)

    # apply model and print prediction
    label, label_num, preds = get_prediction(model, image_np)
    print("This is a {}".format(label_num))

    # save classification results as a diagram
    view_classify(image_np, preds)

    # create plotly visualization
    graphs = [
        #plot with probabilities for each class of images
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

    # render the hook.html passing prediction resuls
    return render_template(
        'hook.html',
        result = label_num, # predicted class label
        ids=ids, # plotly graph ids
        graphJSON=graphJSON, # json plotly graphs
        dataURL = dataURL # image to display with result
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
