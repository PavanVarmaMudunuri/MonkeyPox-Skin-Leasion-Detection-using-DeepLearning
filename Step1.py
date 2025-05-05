import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os,fnmatch
from skimage.data import coins
from skimage.morphology import label, remove_small_objects
from skimage.measure import regionprops, find_contours
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
from scipy import stats
#from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from skimage.measure import compare_ssim
import warnings
#from Step1 import *
warnings.filterwarnings("ignore")
###############################
###############################
def pre_process(fname):                                                    ''' This is the main function that takes an image filename (fname) and performs preprocessing.
                                                                                cv2.threshold is used to create binary images using different thresholding methods.
                                                                                Otsu's thresholding is automatically determining the best threshold. 
                                                                                Gaussian blur helps smooth the image before applying Otsu.'''
        filename=fname;
        img = cv2.imread(filename,0)
        ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        images = [img, 0, th1,
                  img, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image','Histogram','Global Thresholding',
                  'Original Noisy Image','Histogram',"Otsu's Thresholding",
                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
        for i in range(3):                                                                                            '''Shows comparisons between global and Otsu’s thresholding with and without noise reduction.'''
                plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
                plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
                plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
                plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
                plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
                plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        plt.show()
#################################                                                                     

#################################                                                                                       
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)                        '''Applies and visualizes all basic OpenCV thresholding types: BINARY, INV, TRUNC, TOZERO, and TOZERO_INV.'''
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        for i in range(0,6):
                plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
                plt.title(titles[i])
                plt.xticks([]),plt.yticks([])
        plt.show()
###############################
        I = img
        height, width = np.shape(I)                                        '''Calculates the mean pixel value to be used as an initial threshold.

                                                                        Then separates the image into two groups based on this threshold for further analysis.'''
        print("Size of Image: ", height, " X ", width)
        plt.imshow(I, cmap=plt.get_cmap('gray'))
        plt.title("Original Image (Step 1)")
        plt.show()
        f = open("img.txt", "w")
        sum = 0
        for px in range(0, height):
                for py in range(0, width):
                        sum = sum + I[px][py] #sum of all pixels
                        f.write (str(I[px][py]) + " ")
                        f.write("\n")
        no_of_pixels=height*width#no of pixels
        Threshold =sum/no_of_pixels # initial Threshold
        print("Threshold : ", Threshold)
        sum1=0
        sum2=0
        count1=0
        count2=0
        for i in np.arange(height):
                for j in np.arange(width):
                        a = I.item(i,j)
                        if a > Threshold:
                                sum1 =sum1 + a
                                count1 =count1 + 1
                        else:
                                sum2 =sum2 + a
                                count2 =count2 + 1
        A1 =sum1/count1
        A2 =sum2/count2
        print("Average G1 :",A1 )
        print("Average G2 :",A2 )

        Threshold =(1/2)*(A1+A2)
        c= -2

        Threshold2=Threshold
        while c > 1:#checking for T0 Condition          '''Uses iterative method to converge to a better threshold value by recalculating class means.'''
                sum1=0
                sum2=0
                count1=0
                count2=0
                Threshold2=Threshold
                for i in np.arange(height):                         '''Applies the computed final threshold and modifies the original image to a binary form.'''
                        for j in np.arange(width):
                                a = I.item(i,j)
                                if a > Threshold:
                                        sum1 =sum1 + a
                                        count1 =count1 + 1
                                else:
                                        sum2 =sum2 + a
                                        count2 =count2 + 1
        A1 =sum1/count1
        A2 =sum2/count2
        Threshold =(1/2)*(A1+A2)
        c = Threshold2-Threshold
        print('Threshold new:',Threshold)
        for i in np.arange(height):
                for j in np.arange(width):
                        a = I.item(i,j)
                        if a > Threshold: #Final Threshold
                                b=255#initializing value G
                                I.itemset((i,j) ,b)# G(i,j) = 255 , if I(i,j) > T
                        else:
                                b=0 #G(i,j)= 0, if I(i,j)<=T
                                I.itemset((i,j) ,b)
        plt.imshow(I, cmap=plt.get_cmap('gray')) # final image plot
        plt.title("Resulting Image ::")
        plt.show()
        Icopy = I.astype('uint8')
        mean = 1.0 # some constant
        std = 1.0# some constant (standard deviation)
        noisy_img = Icopy + np.random.normal(mean, std, Icopy.shape)                  '''Adds Gaussian noise to the image to simulate real-world imaging noise.'''
        I2 = np.clip(noisy_img, 0, 255)
        plt.imshow(I2, cmap=plt.get_cmap('gray'))
        plt.title(" Noise Addition ::")
        plt.show()
        img_bw = 255*I2.astype('uint8')
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)   '''Uses morphological operations (MORPH_CLOSE and MORPH_OPEN) to remove small noise and artifacts.'''
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        out = I2 * mask
        plt.imshow(out, cmap=plt.get_cmap('gray'))
        plt.title("Removal of Noise ")
        plt.show()
        ## median Filter ###
        out2 =cv2.medianBlur(I, 3)                                        '''Applies a median filter to smooth the image and reduce noise while preserving edges.'''
        plt.imshow(out2, cmap=plt.get_cmap('gray'))
        plt.title("Removal of Noise median Filter ")
        plt.show()
########################################
########################################
########################################
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#_,contours,hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        contours,hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)    '''Finds and labels contours in the image.

                                                                                                Uses regionprops to get statistics about labeled regions.
                                                                                                Counts objects that are larger than a calculated threshold area.'''

#V1 = img.astype(np.int32)
        V2 = np.zeros_like(img).astype(np.uint8)
        a1 = 21 * (584 / 565) * 2
        label_img = label(img, connectivity = 2)
        props = regionprops(label_img)
        count = 0
        V2 = remove_small_objects(label_img, min_size = a1)
        for prop in props:
                if prop['Area'] > a1:
                        count += 1
        print(count)
##########################################
##########################################
##########################################

#import os
#path ="DataSet/validation/"
#filelist = []
#for root, dirs, files in os.walk(path):
#	for file in files:
#		filelist.append(file)

#for name in filelist:
#    print(name)
 #   pre_process(path+name)
