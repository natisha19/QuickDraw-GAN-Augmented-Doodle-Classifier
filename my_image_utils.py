import numpy as np
import pandas as pd
import os
from os import path
import pickle
import random
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from PIL import ImageOps

def load_data():
    print("Loading data \n")
    if not(path.exists('xtrain_doodle.pickle')):
        print("Loading data from the web \n")
        categories = ['apple','banana', 'car', 'cat', 'dog','house','tree','bicycle','fish','chair']
        URL_DATA = {}
        for category in categories:
            URL_DATA[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category +'.npy'
        classes_dict = {}
        for key, value in URL_DATA.items():
            response = requests.get(value)
            classes_dict[key] = np.load(BytesIO(response.content))
        for i, (key, value) in enumerate(classes_dict.items()):
            value = value.astype('float32')/255.
            if i == 0:
                classes_dict[key] = np.c_[value, np.zeros(len(value))]
            else:
                classes_dict[key] = np.c_[value,i*np.ones(len(value))]
        label_dict = {
            0: 'apple', 1: 'banana', 2: 'car', 3: 'cat', 4: 'dog',
            5: 'house', 6: 'tree', 7: 'bicycle', 8: 'fish', 9: 'chair'
        }
        lst = []
        for key, value in classes_dict.items():
            lst.append(value[:3000])
        doodles = np.concatenate(lst)
        y = doodles[:,-1].astype('float32')
        X = doodles[:,:784]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    else:
        print("Loading data from pickle files \n")
        file = open("xtrain_doodle.pickle",'rb')
        X_train = pickle.load(file)
        file.close()
        file = open("xtest_doodle.pickle",'rb')
        X_test = pickle.load(file)
        file.close()
        file = open("ytrain_doodle.pickle",'rb')
        y_train = pickle.load(file)
        file.close()
        file = open("ytest_doodle.pickle",'rb')
        y_test = pickle.load(file)
        file.close()
    return X_train, y_train, X_test, y_test

def view_image(img, filename = 'image'):
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.reshape(28, 28).squeeze())
    ax.axis('off')
    plt.savefig(filename + '.png')

def convert_to_PIL(img):
    img_r = img.reshape(28,28)
    pil_img = Image.new('RGB', (28, 28), 'white')
    pixels = pil_img.load()
    for i in range(0, 28):
        for j in range(0, 28):
            if img_r[i, j] > 0:
                pixels[j, i] = (255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255))
    return pil_img

def rotate_image(src_im, angle = 45, size = (28,28)):
    dst_im = Image.new("RGBA", size, "white")
    src_im = src_im.convert('RGBA')
    rot = src_im.rotate(angle)
    dst_im.paste(rot, (0, 0), rot)
    return dst_im

def flip_image(src_im):
    # Flips the image horizontally
    dst_im = src_im.transpose(Image.FLIP_LEFT_RIGHT)
    return dst_im

def convert_to_np(pil_img):
    pil_img = pil_img.convert('RGB')
    img = np.zeros((28, 28))
    pixels = pil_img.load()
    for i in range(0, 28):
        for j in range(0, 28):
            img[i, j] = 1 - pixels[j, i][0] / 255
    return img

def add_flipped_and_rotated_images(X_train, y_train):
    print("Adding flipped and rotated images to the training set. \n")
    X_train_new = X_train.copy()
    y_train_new = y_train.copy().reshape(y_train.shape[0], 1)
    for i in range(0, X_train.shape[0]):
        img = X_train[i]
        pil_img = convert_to_PIL(img)
        angle = random.randint(5, 10)
        rotated = convert_to_np(rotate_image(pil_img, angle))
        flipped = convert_to_np(flip_image(pil_img))
        X_train_new = np.append(X_train_new, rotated.reshape(1, 784), axis = 0)
        X_train_new = np.append(X_train_new, flipped.reshape(1, 784), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)
        if i % 100 == 0:
            print(f"Processed {i} files out of {X_train.shape[0]}.")
    return X_train_new, y_train_new

def view_label_heatmap(X_train, y_train, label, label_name):
    label_filter = y_train == label
    X = pd.DataFrame(X_train)
    X_train_labeled = X[label_filter]
    X_mean = np.sum(X, axis = 0).values
    fig, ax = plt.subplots(figsize=(6,9))
    ax.set_title(label_name)
    ax.imshow(X_mean.reshape(28, 28).squeeze())
    ax.axis('off')
    plt.savefig(label_name + '.png')

def view_images_grid(X_train, y_train, label, label_name):
    # Plots a grid of sample images with the specified label
    indices = np.where(y_train == label)
    X = pd.DataFrame(X_train)
    for label_num in range(0,50):
        plt.subplot(5,10, label_num+1)
        image = X.iloc[indices[0][label_num]].as_matrix().reshape(28,28)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(label_name)
        plt.savefig(label_name + '_grid.png')

def plot_image(image, label_name):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    ax.set_title(label_name)
    dims = (fig.canvas.get_width_height()[0] * 2, fig.canvas.get_width_height()[1] * 2)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(dims[::-1] + (3,))
    return image

def create_animated_images(X_train, y_train, label, label_name):
    # Creates and saves an animated gif of sample images from a class
    indices = np.where(y_train == label)
    X = pd.DataFrame(X_train)
    images = []
    for label_num in range(0,50):
        image = X.iloc[indices[0][label_num]].as_matrix().reshape(28,28)
        images.append(image)
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./'+ label_name + '.gif', [plot_image(i, label_name) for i in images], fps=1)

def crop_image(image):
    # Crops white spaces from image and centers the content, resizing to (28,28)
    cropped_image = image
    width, height = cropped_image.size
    pixels = cropped_image.load()
    image_strokes_rows = []
    image_strokes_cols = []
    for i in range(0, width):
        for j in range(0, height):
            if (pixels[i,j][3] > 0):
                image_strokes_cols.append(i)
                image_strokes_rows.append(j)
    if (len(image_strokes_rows)) > 0:
        row_min = np.array(image_strokes_rows).min()
        row_max = np.array(image_strokes_rows).max()
        col_min = np.array(image_strokes_cols).min()
        col_max = np.array(image_strokes_cols).max()
        margin = min(row_min, height - row_max, col_min, width - col_max)
        border = (col_min, row_min, width - col_max, height - row_max)
        cropped_image = ImageOps.crop(cropped_image, border)
    width_cropped, height_cropped = cropped_image.size
    dst_im = Image.new("RGBA", (max(width_cropped, height_cropped), max(width_cropped, height_cropped)), "white")
    offset = ((max(width_cropped, height_cropped) - width_cropped) // 2, (max(width_cropped, height_cropped) - height_cropped) // 2)
    dst_im.paste(cropped_image, offset, cropped_image)
    dst_im.thumbnail((28,28), Image.Resampling.LANCZOS)
    return dst_im

def normalize(arr):
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def normalize_image(image):
    # Performs normalization to get ranges in [0,255] for each channel
    arr = np.array(image)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    return new_img

def alpha_composite(front, back):
    # Alpha composites two RGBA images together
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result

def alpha_composite_with_color(image, color=(255, 255, 255)):
    # Alpha composite an RGBA image onto a solid RGB background
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)

def convert_to_rgb(image):
    image_rgb = alpha_composite_with_color(image)
    image_rgb.convert('RGB')
    return image_rgb
