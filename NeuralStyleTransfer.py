#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:49:29 2019

@author: bajpaiarjun0333
"""

from nst_utils import *
import os
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf

#loading the main model
#model=load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
#print(model)

#content_image=scipy.misc.imread("images/louvre.jpg")
#imshow(content_image)

#function to find out the content cost
def compute_content_cost(a_C,a_G):
    #retriving the dimension of the activation function
    m,n_H,n_W,n_C=a_G.get_shape.as_list()
    
    a_C_unrolled=tf.reshape(a_C,[n_H*n_W,n_C])
    a_G_unrolled=tf.reshape(a_G,[n_H*n_W,n_C])
    
    J_content=(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))))/(4*n_H*n_W*n_C)
    
    return J_content


#reading the style imahe
#style_image=scipy.misc.imread("images/monet_800600.jpg")
#imshow(style_image)

#The style matrix
def gram_matrix(A):
    GA=tf.matmul(A,tf.transpose(A))
    return GA

#computing the single layer style cost 
def compute_layer_style_cost(a_S,a_G):
    m,n_H,n_W,n_C=a_G.get_shape().as_list()
    a_S=tf.reshape(a_S,[n_H*n_W,n_C])
    a_G=tf.reshape(a_G,[n_H*n_W,n_C])
    
    GS=gram_matrix(tf.transpose(a_S))
    GG=gram_matrix(tf.transpose(a_G))
    
    J_style_layer=tf.reduce_sum((GS-GG)**2)/(4*(n_C**2)*((n_H*n_W)**2))
    return J_style_layer

STYLE_LAYERS=[('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]

#overall style cost
def compute_style_cost(model,STYLE_LAYERS):
    J_style=0
    for layer_name,coeff in STYLE_LAYERS:
        out=model[layer_name]
        a_S=sess.run(out)
        
        a_G=out
        
        J_style_layer=compute_layer_style_cost(a_S,a_G)
        J_style+=coeff*J_style_layer
        
    return J_style

#defining the total cost
def total_cost(J_content,J_style,alpha=10,beta=40):
    J=alpha*J_content+beta*J_style
    return J

tf.reset_default_graph()
sess=tf.InteractiveSession()
content_image=scipy.misc.imread("images/louvre_small.jpg")
style_image=scipy.misc.imread("images/monet.jpg")
content_image=reshape_and_normalize_image(content_image)
style_image=reshape_and_normalize_image(style_image)

generated_image=generate_noise_image(content_image)
imshow(generated_image[0])

model=load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))
out=model['conv4_2']
a_C=sess.run(out)
a_G=out
J_content=compute_content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))
J_style=compute_style_cost(model,STYLE_LAYERS)

J=total_cost(J_content,J_style,10,40)

optimizer=tf.train.AdamOptimizer(2.0)
train_step=optimizer.minimize(J)

#defining the model
def model_nn(sess,input_image,num_iterations=200):
    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        _=sess.run(train_step)
        
        generated_image=sess.run(model['input'])
        if i%20==0:
            Jt,Jc,Js=sess.run([J,J_content,J_style])
            print("Iteration "+str(i)+":")
            print("Total cost : "+str(Jt))
            print("Content cost : "+str(Jc))
            print("Style cost : "+str(Js))
            save_image("output/"+str(i)+".png",generated_image)
            
    save_image('output/generated_image.jpg',generated_image)
    return generated_image

model_nn(sess,generated_image)



