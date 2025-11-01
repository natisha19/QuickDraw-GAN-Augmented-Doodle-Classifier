# QuickDraw-GAN-Augmented-Doodle-Classifier
We train a 10‑class QuickDraw CNN, optionally augment with a tiny DCGAN, and serve real‑time predictions via a Flask app. The project includes a compact PyTorch CNN, preprocessing utilities, a Colab‑ready end‑to‑end training script, and a Flask endpoint that returns predictions with a Plotly visualization.

Overview
This repository builds a grayscale 28×28 CNN classifier over ten selected QuickDraw categories and can train a tiny DCGAN to synthesize sketches for data augmentation before retraining the classifier. The Flask app loads the saved PyTorch checkpoint, applies consistent preprocessing, and returns class probabilities from softmax for the drawn input.

Features
SimpleCNN: three Conv‑BN‑ReLU stages with pooling and an MLP head for 10 logits, designed for 28×28 grayscale inputs.
Colab script: downloads class bitmaps, prepares loaders, trains the CNN, trains a tiny DCGAN, generates synthetic images, and retrains with augmentation, saving checkpoints.
Flask app: loads checkpoint_classifier.pth once on startup, preprocesses incoming canvas images, and renders a probability chart with Plotly.
Image utilities: crop to drawing content, intensity normalization, RGBA→RGB conversion, PIL↔NumPy round‑trip, and simple visualization helpers.
Repository structure
my_image_utils.py : image preprocessing, normalization, RGBA handling, and helper visualizations used by the app and exploration flows.
simple_cnn.py : SimpleCNN architecture used for training and inference in the web app.
gansmall.ipynb : Colab‑ready end‑to‑end: dataset download, CNN training, DCGAN training, synthetic generation, augmentation retrain, and checkpoint export.
run.py : Flask server: loads checkpoint, preprocesses inputs, computes predictions, and renders a Plotly chart.
Requirements
Use Python 3.9+ with PyTorch, torchvision, numpy, Pillow, Flask, matplotlib, plotly, scikit‑learn, and wget for dataset downloads in the training script .The Flask app expects checkpoint_classifier.pth and uses the same normalization pipeline as the training transforms for consistent results.

Installation
Install dependencies and run the app locally after producing a trained checkpoint.
