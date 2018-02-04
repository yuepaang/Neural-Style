#!/usr/bin/env python2
# -*- coding: utf-8 -*- 


import tensorflow as tf
import numpy as np
import skimage.transform as st
import imageio
from sys import stderr
from PIL import Image
import pickle


try:
    reduce
except NameError:
    from functools import reduce


# Preprocessing function
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def image_preprocessing(img):
    pad_size = abs(img.shape[0]-img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    img = st.resize(img, (64 - 4*2, 64 - 4*2), mode='constant')
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)
    img = img.flatten()
    img = (img - 128) / 128
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


# Loading the content image, style image
content = imageio.imread('content.jpg')
# preprocessing
grey = rgb2gray(content)
content_pre = (grey/256).astype(np.float32)
print "content image loaded"

# Loading the style image
style = imageio.imread('style.jpg')
grey = rgb2gray(style)
style_pre = (grey/256).astype(np.float32)
print "style image loaded"


num_char = 200
content_weight_blend = 1.0
content_weight = 5e0
style_weight = 5e2
tv_weight = 10
learning_rate = 1e-2
beta1 = 0.999
beta2 = 0.9999
epsilon = 1e-08
iterations = 1000
print_iterations = 100
checkpoint_iterations = 2
preserve_colors = True

# define the layers and layer weights
content_layer = ['conv1_1','conv2_1','conv3_1']
style_layer = ['relu1_1','relu2_1','relu3_1']
content_layer_weights = [0.8, 0.2, 0]

# Restore trained network
with open("weights","rb") as w:
    weights = pickle.load(w)
print "Network loaded"


def feature_extract(x,layer):
    conv1_1 = tf.nn.bias_add(tf.nn.conv2d(x, tf.constant(weights['w_c1']), strides=[1, 1, 1, 1], padding='SAME'),tf.constant(weights['b_c1']))
    relu1_1 = tf.nn.relu(conv1_1)
    conv1_2 = tf.nn.max_pool(relu1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2_1 = tf.nn.bias_add(tf.nn.conv2d(conv1_2, tf.constant(weights['w_c2']), strides=[1, 1, 1, 1], padding='SAME'), tf.constant(weights['b_c2']))
    relu2_1 = tf.nn.relu(conv2_1)
    conv2_2 = tf.nn.max_pool(relu2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv3_1 = tf.nn.bias_add(tf.nn.conv2d(conv2_2, tf.constant(weights['w_c3']), strides=[1, 1, 1, 1], padding='SAME'), tf.constant(weights['b_c3']))
    relu3_1 = tf.nn.relu(conv3_1)
    conv3_2 = tf.nn.max_pool(relu3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv4_1 = tf.nn.bias_add(tf.nn.conv2d(conv3_2, tf.constant(weights['w_c4']), strides=[1, 1, 1, 1], padding='SAME'), tf.constant(weights['b_c4']))
    relu4_1 = tf.nn.relu(conv4_1)
    conv4_2 = tf.nn.max_pool(relu4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv5_1 = tf.nn.bias_add(tf.nn.conv2d(conv4_2, tf.constant(weights['w_c5']), strides=[1, 1, 1, 1], padding='SAME'), tf.constant(weights['b_c5']))
    relu5_1 = tf.nn.relu(conv5_1)
    conv5_2 = tf.nn.max_pool(relu5_1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

    layer_out = [conv1_1,relu1_1,conv1_2,conv2_1,relu2_1,conv2_2,conv3_1,relu3_1,conv3_2,conv4_1,relu4_1,conv4_2,conv5_1,relu5_1,conv5_2]

    layer_names = ['conv1_1','relu1_1','conv1_2','conv2_1','relu2_1','conv2_2','conv3_1','relu3_1','conv3_2','conv4_1','relu4_1','conv4_2','conv5_1','relu5_1','conv5_2']
    
    out_dict = dict(zip(layer_names,layer_out))
    return out_dict[layer]


# Create image variable
image = tf.Variable(tf.truncated_normal([1,len(content_pre),len(content_pre[0]),1], stddev = 0.1))
image_width = len(content_pre[0])
image_height = len(content_pre)

sess2 = tf.Session()

# Content_forward
#imsave("content_processed.jpg",content_pre)

    
content_pre = np.reshape(content_pre,[1,len(content_pre),len(content_pre[0]),1])
content_feature = {}
for layer in content_layer:
    content_feature[layer] = sess2.run(feature_extract(x=content_pre, layer=layer))
#content_dict =  dict(zip(content_layer, content_feature))
print "content feature computed"


# Style_forward
#imsave("style_processed.jpg",grey)
style_pre = np.reshape(style_pre,[1,len(style_pre), len(style_pre[0]),1])

style_feature = {}
for layer in style_layer:
    style_feature[layer] = sess2.run(feature_extract(x=style_pre, layer=layer))
#style_dict =  dict(zip(content_layer, style_feature))

print "style feature computed"

# Image_forward
image_content_feature = {}
for layer in content_layer:
    image_content_feature[layer] = feature_extract(x=image,layer = layer)
image_style_feature = {}
for layer in style_layer:
    image_style_feature[layer] = feature_extract(x=image,layer = layer)


#image_content_dict = dict(zip(content_layer, image_content_feature))
#image_style_dict = dict(zip(style_layer, image_style_feature))

print "image feature computed"


###compute loss###
#content loss


content_loss = 0
content_losses = []
for i, layer in enumerate(content_layer):
    content_losses.append(content_layer_weights[i] * content_weight * \
    (2 * tf.nn.l2_loss(image_content_feature[layer] - content_feature[layer]) / content_feature[layer].size))
    
content_loss += reduce(tf.add, content_losses)

print "content loss computed" 

#style loss
style_loss = 0

layer_weight = 1.0
style_layer_weight_exp = 2.0
style_layer_weights = []
for layer in style_layer:
    style_layer_weights.append(layer_weight)
    layer_weight *= style_layer_weight_exp

#normalize style layer weights
layer_weights_sum = np.sum(style_layer_weights)
style_layer_weights /= layer_weights_sum
style_blend_weights = 1.0
#compute loss
style_losses = []
for i, layer in enumerate(style_layer):
    layer_feature = image_style_feature[layer]
    _, height, width, number = map(lambda i: i.value, layer_feature.get_shape())
    size = height * width * number
    feats = tf.reshape(layer_feature,[-1,number])
    gram = tf.matmul(tf.transpose(feats),feats)
    #gram = layer_feature
    style_gram = style_feature[layer]
    style_gram = np.reshape(style_gram,[-1,style_gram.shape[3]])
    style_gram = np.matmul(style_gram.T, style_gram)
    style_losses.append(style_layer_weights[i] * 2 * tf.nn.l2_loss(gram - style_gram)/ (style_gram.size*size))

style_loss += style_weight * style_blend_weights * reduce(tf.add, style_losses)

#total variation denoising
tv_y_size = _tensor_size(image[:,1:,:,:])
tv_x_size = _tensor_size(image[:,:,1:,:])
tv_loss = tv_weight * 2* ((tf.nn.l2_loss(image[:,1:,:,:] - image[:,:image_height-1,:,:]) / tv_y_size) +\
(tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:image_width - 1,:])/tv_x_size))

#overall_loss
loss = content_loss + style_loss + tv_loss

print "overall loss computed"

#optimizer
train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)
sess2.run(tf.global_variables_initializer())
print "optimizer defined"

#train
for iteration in range(iterations):
    sess2.run(train_step)
    if iteration%100 == 0:
        print sess2.run(loss)
        print sess2.run(content_loss)
        print sess2.run(style_loss)
        print sess2.run(tv_loss)
img_out = sess2.run(image)
img_out = 256 * np.reshape(img_out,[image_height,image_width])
imsave("output1.jpg", img_out)

#save style features
# for layer in style_layer:
#     feature = style_feature[layer]
#     _, h, w ,d = feature.shape
#     feature = np.reshape(feature,[h,w,d])
#     feature = np.sum(feature,axis = 2)*256
#     print(feature.shape)
#     imsave("style_"+layer+".jpg",feature)

# for layer in content_layer:
#     feature = content_feature[layer]
#     _, h, w, d = feature.shape
#     feature = np.reshape(feature,[h,w,d])
#     feature = np.sum(feature,axis = 2)
#     print feature
#     f1 = open("style_"+layer, "wb")
#     pickle.dump(feature,f1)
#     f1.close()    
# imsave("content_"+layer+".jpg",feature)

sess2.close()