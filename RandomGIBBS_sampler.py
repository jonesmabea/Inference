#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:55:05 2017

@author: Jones
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

def add_gaussian_noise(im,prop,varSigma):
    """
    Adds gaussian noise to image
    args: image im, proportion of pixels to be altered prop, sigma value of gaussian varSigma
    returns Noisy image im2 
    
    
    """
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]
    return im2
def add_saltnpeppar_noise(im,prop):
    """
    Adds salt and pepper noise to image
    args: Image im, Number of pixel to be altered prop
    returns: Noisy image im2
    
    """
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2


def neighbours(i,j,M,N,size=4):
    """
    Function that finds surrounding pixel values. 
    Args: horizontal position i, vertical position j, horizontal limit M, Vertical limit N, size of surrounding pizels size
    returns list of surrounding pixels.
    """
    if size == 4:
        if (i == 0 and j == 0):
            n = [(0, 1), (1, 0)]
        elif i == 0 and j == N - 1:
            n = [(0, N - 2), (1, N - 1)]
        elif i == M - 1 and j == 0:
            n = [(M - 1, 1), (M - 2, 0)]
        elif i == M - 1 and j == N - 1:
            n = [(M - 1, N - 2), (M - 2, N - 1)]
        elif i == 0:
            n = [(0, j - 1), (0, j + 1), (1, j)]
        elif i == M - 1:
            n = [(M - 1, j - 1), (M - 1, j + 1), (M - 2, j)]
        elif j == 0:
            n = [(i - 1, 0), (i + 1, 0), (i, 1)]
        elif j == N - 1:
            n = [(i - 1, N - 1), (i + 1, N - 1), (i, N - 2)]
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        return n
    if size == 8:
        #print('Not yet implemented\n')
        # top left
        if (i == 0 and j == 0):
            n = [(0, 1), (1, 0), (1, 1)]
        # bottom left
        elif i == 0 and j == N - 1:
            n = [(0, N - 2), (1, N - 1), (1, N - 2)]
        # top right
        elif i == M - 1 and j == 0:
            n = [(M - 1, 1), (M - 2, 0), (M - 2, 1)]
        # bottom right
        elif i == M - 1 and j == N - 1:
            n = [(M - 1, N - 2), (M - 2, N - 1), (M - 2, N - 2)]
        # left
        elif i == 0:
            n = [(0, j - 1), (0, j + 1), (1, j), (1, j - 1), (1, j + 1)]
        # right
        elif i == M - 1:
            n = [(M - 1, j - 1), (M - 1, j + 1), (M - 2, j), (M - 2, j + 1), (M - 2, j - 1)]
        # top
        elif j == 0:
            n = [(i - 1, 0), (i + 1, 0), (i, 1), (i - 1, 1), (i + 1, 1)]
        # bottom
        elif j == N - 1:
            n = [(i - 1, N - 1), (i + 1, N - 1), (i, N - 2), (i + 1, N - 2), (i - 1, N - 2)]
        # middle 
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
        return n

    

def likelihood(x,y):
    """
    Likelihood function for image pixel value and latent value
    args: latent value x, image pixel value y
    returns likelihood
    
    """
    return (np.square(2*(y-0.5)+x))


def prior(x,nb,x_i):
    """
    Prior function to return number of common neighbours
    args: neighbouring pixel values neighbours, distribution value u 
    returns total of neighbouring distribution values m
    
    
    """
    m = 0
    for el in nb:
        i,j = el
        
        m += x[i][j]*x_i
    
    return m
    
def normalise(im, x):
    
    """
    Function to return normalised image values to either 0,1 for Python3+ 
    
    Args: Image im, Uninitialised latent variable x
    returns: initialised latent variable x, mean field distribution u
    
    """
  
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if im[row][col] <= 0:
                im[row][col] = 0
                x[row][col] = -1
            else:
                x[row][col] = 1
    
    return im, x


def GIBBS(im,x,size):
    EPOCH = 50
    m,n = im.shape
    x_new = x
    np.random.seed(42)
    for epoch in range(EPOCH):
        rows = np.random.permutation(m)
        cols = np.random.permutation(n)
        for i in rows:
            for j in cols:
                nb = neighbours(i,j,m,n,size)
                y = im[i,j]
                p = np.divide(prior(x,nb,1)*likelihood(1,y),(prior(x,nb,1)*likelihood(1,y) + prior(x,nb,-1)*likelihood(-1,y)))
                t = np.random.uniform(0,1)
                if p > t:
                    x_new[i][j] = 1
                else:
                    x_new[i][j] = -1
                     
    return x_new
   # proportion of pixels to alter



  
im = imread('./pics/pug.jpg')
im = im/255

print("Beginning Gibbs Sampler with Random Pixels")
for p in (0.1,0.3,0.5):
    for size in(4,8):
        prop = p
        varSigma = p
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.set_xlabel("Original Image")
        ax.imshow(im,cmap='gray')
        
        
        x = np.zeros(im.shape)
        im2, x= normalise(im,x)
        im2 = GIBBS(im2,x,size)
        ax1 = fig.add_subplot(222)
        ax1.set_xlabel("Original Remake")
        ax1.imshow(im2,cmap='gray')
        
        ##############################################
        fig = plt.figure()
        im2 = add_gaussian_noise(im,prop,varSigma)
        ax2 = fig.add_subplot(221)
        ax2.set_xlabel("Gaussian Noise")
        ax2.imshow(im2,cmap='gray')
        
        
        x = np.zeros(im.shape)
        im2, x= normalise(im2,x)
        im2 = GIBBS(im2,x,size)
        ax4 = fig.add_subplot(222)
        ax4.set_xlabel("Gaussian Remake")
        ax4.imshow(im2,cmap='gray')
        
        ###############################################
        fig = plt.figure()
        im3 = add_saltnpeppar_noise(im,prop)
        ax3 = fig.add_subplot(221)
        ax3.set_xlabel("Salt and Pepper Noise")
        ax3.imshow(im3,cmap='gray')
        
        
        x = np.zeros(im.shape)
        im3, x= normalise(im3,x)
        im3 = GIBBS(im3,x,size)
        ax5 = fig.add_subplot(222)
        ax5.set_xlabel("Salt and Pepper Remake")
        ax5.imshow(im3,cmap='gray')
        
        plt.show()
        print(f"size: {size}")
        print("#"*20)
    print(f"Prop:{prop}  Sig:{varSigma}")
    print("###################################################")