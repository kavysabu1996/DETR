import tensorflow as tf
from os import path
import urllib
from urllib import request
import numpy as np
import cv2

def read_image(input_image):
    if isinstance(input_image, str):  
        if input_image.startswith('http'):  
            resp = urllib.request.urlopen(input_image)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:  
            image = cv2.imread(input_image)
    elif isinstance(input_image, np.ndarray): 
        image = input_image
    else:
        print("Error: Invalid input image format.")
    return image

def resize(image, min_side=800.0, max_side=1333.0):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cur_min_side = tf.minimum(w, h)
    cur_max_side = tf.maximum(w, h)

    scale = tf.minimum(max_side / cur_max_side,
                       min_side / cur_min_side)
    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image

def build_mask(image):
    return tf.zeros(tf.shape(image)[:2], dtype=tf.bool)

def cxcywh2xyxy(boxes):
    cx, cy, w, h = [boxes[..., i] for i in range(4)]

    xmin, ymin = cx - w*0.5, cy - h*0.5
    xmax, ymax = cx + w*0.5, cy + h*0.5

    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes


def absolute2relative(boxes, img_size):
    width, height = img_size
    scale = tf.constant([width, height, width, height], dtype=tf.float32)
    boxes *= scale
    return boxes

def preprocess_image(image):
    image = resize(image, min_side=800.0, max_side=1333.0)
    
    channel_avg = tf.constant([0.485, 0.456, 0.406])
    channel_std = tf.constant([0.229, 0.224, 0.225])
    image = (image / 255.0 - channel_avg) / channel_std

    return image, build_mask(image)
