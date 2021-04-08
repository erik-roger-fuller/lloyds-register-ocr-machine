import cv2
import numpy as np
from cv2 import medianBlur

"""Image processsing"""
def upscale(image, scale_factor):
    scale_percent = scale_factor * 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)#INTER_CUBIC
    return resized

def remove_noise(image, nsize):
    #if image is not None:
    #    print("remove noise exists!")
    image = cv2.medianBlur(image, nsize) #input_image=image, kernel_size=nsize)
    return image

def thresholding(image):# thresholding
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def dilate(image):# dilation
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=2)

def erode(image):# erosion
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(image, kernel, iterations=2)

def opening(image):# opening - erosion followed by dilation
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# canny edge detection
def canny(image):
    return cv2.Canny(image, 50, 150, apertureSize=3)

def apply_filter(image):
    """Define a 5X5 kernel and apply the filter to gray scale image
    Args: image: np.array  Returnsfiltered: np.array"""
    kernel = np.ones((35, 35), np.float32) / 15
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered

def brightness_contrast_adj(image, alpha, beta):
    #alpha = 1.7 # Simple contrast control
    #beta = (-155)    # Simple brightness control
    adj = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

def top_crop(img):
    y , x = img.shape[0] , img.shape[1]
    roi = img[200:(y-100), 75:(x-75)]
    #print(roi.shape)
    return roi

def rectrangle_crop(img, top_left, bottom_right):
    left, top = top_left[0], top_left[1]
    right, bottom = bottom_right[0], bottom_right[1]
    roi = img[top:bottom , left:right]
    print("rectangle:  ",top ,bottom , left, right)
    return roi

def img_processing(img):
    img = top_crop(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #filtered = apply_filter(grey)
    adj = brightness_contrast_adj(image=grey, alpha=1.7 , beta=-155)
    thresh = thresholding(adj)
    # Apply edge detection method on the image
    edges = canny(thresh)
    return edges, grey

def preprocess_for_ocr(image, scale_factor, boundaries):
    image = brightness_contrast_adj(image=image, alpha=1.8, beta=-200)
    #image = remove_noise(image)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]#thresholding(image)
    image = dilate(image)
    #image = erode(image)
    #image = opening(image)
    #image = canny(image)
    return image