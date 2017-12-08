#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:38:15 2017

@author: Jones Agwata

This is an Image segmentation program

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.neighbors import KNeighborsClassifier

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



def likelihood(cl,y):
    """
    Likelihood function for image pixel value and knn
    args: classifier object cl, image pixel value y
    returns classifier prediction
    
    """
    return cl.predict(y.reshape(-1,3))


def normalise(im, x):
    """
    Function to return normalised image values to either 0,1 for Python3+ 
    
    Args: Image im, Uninitialised latent variable x
    returns: initialised latent variable x, mean field distribution u
    
    """
    u = np.zeros(im.shape)
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if im[row][col] <= 0:
                im[row][col] = 0
                x[row][col] = -1
            else:
                x[row][col] = 1

    return im, x, u

def im_plot(x, im):
    """
    Function to decode latent variables and replace 
    background pixels to green(0,255,0)
    
    Args: Latent Variable c, Image im
    """
    m,n = x.shape[0],im.shape[1]
    for i in range(m):
        for j in range(n):
            if x[i][j] == -1:
                im[i][j]=[0,255,0]
                
    return im
                

def IMSEG(im,x,cl):
    """
    Main function to segment image
    args: Image im, latent variable x, classifier object cl
    returns: encoded latent variable x
    """
    m,n = im.shape[0],im.shape[1]
    for i in range(m):
        for j in range(n):
            fl = likelihood(cl,im[i][j])
            #print(fl, bl)
            if fl == 1:
                x[i][j] = 1
            else:
                x[i][j]= -1
                    
                
                     
    return x
   # proportion of pixels to alter



# import the necessary packages


def pix(im):
    """
    Function to return list of of rgb pixel values
    Args: Image im
    """
    m,n = im.shape[:2]
    p_list = []
    for i in range(m):
        for j in range(n):
          p_list.append([im[i][j][0],im[i][j][1],im[i][j][2]])  


    return np.array(p_list)

#Read original image 
im = imread('./pics/pug2-r.ppm')

#Read foreground and background mask 
f_m = imread('./pics/pug2-r-fmask.ppm')
b_m = imread('./pics/pug2-r-bmask.ppm')


#Get all foreground and background pixels and concatenate them in to one list.
X1 = pix(f_m)
X2 = pix(b_m)
X = np.concatenate((X1,X2))

#Create Target variable denoting foreground as 1 and Background as 0 in same order as pixel list
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])
Y=np.concatenate((Y1,Y2))

#Create KNN classifier and fit the values.
f_neigh = KNeighborsClassifier(n_neighbors=20)
f_neigh = f_neigh.fit(X,Y)


#Plot original image
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(im,cmap='gray')


x = np.zeros(im.shape[:2])
x_n = IMSEG(im,x,f_neigh)
print("done")
#plot Segmented image
im = im_plot(x_n,im)
ax3 = fig.add_subplot(122)
ax3.imshow(im,cmap='gray')

