import cv2
import pytesseract
import numpy as np

"""Image processsing"""
def upscale(image, scale_factor):
    scale_percent = scale_factor * 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)#INTER_CUBIC
    return resized

def remove_noise(image):
    return cv2.medianBlur(image, 7)
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# dilation
def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=2)
# erosion
def erode(image):
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(image, kernel, iterations=2)
# opening - erosion followed by dilation
def opening(image):
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
    print(top ,bottom , left, right)
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

def preprocess_for_ocr(image, scale_factor):
    image = upscale(image, scale_factor)
    image = brightness_contrast_adj(image=image, alpha=1.8, beta=-200)
    #image = remove_noise(image)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]#thresholding(image)
    image = dilate(image)
    #image = erode(image)
    #image = opening(image)
    #image = canny(image)
    return image

def houghp_boxfind(edges, adj ):
    minLineLength = 1000
    maxLineGap = 10
    # This returns an array of r and theta values
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength,maxLineGap)
    distr = range(0, len(lines))

    xcenters, ycenters, ycenters_prelim = [], [], []
    for i in distr:
        for x1, y1, x2, y2 in lines[i]:
            xdiff, ydiff = (x1-x2), (y1-y2)
            #print(xdiff, ydiff)
            if xdiff == 0: #finding vertical ines
                avgx, avgy = int((x1+x2)/2), int((y1+y2)/2)
                xcenters.append(avgx)
                ycenters_prelim.append(avgy)
                #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.circle(img, (avgx, avgy), radius=1, color=(0, 0, 255), thickness=-2)

    yi_q_rhigh = np.percentile(ycenters_prelim, 85)
    yi_q_rlow = np.percentile(ycenters_prelim, 15)
    #print("y qartiles", yi_q_rhigh, yi_q_rlow)

    for y in ycenters_prelim:
        if yi_q_rhigh > y > yi_q_rlow:
            ycenters.append(y)
    ycenter = np.mean(ycenters)
    xcenter = np.mean(xcenters)
    rightedge = int(np.percentile(xcenters, 95)) + 10
    print(rightedge)
    #print(xcenter)
    #print(ycenter) int(xcenter-1400)

    top_left = (int(rightedge-2300), int(ycenter-2000))
    bottom_right = (rightedge, int(ycenter+1900))

    #cv2.rectangle(img=adj, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=5)
    #cv2.imwrite('houghlines5.jpg',img)
    #print("box is done!")
    crop = rectrangle_crop(adj, top_left, bottom_right)
    #cv2.imwrite('houghlines6.jpg',crop)
    print("crop is done!")
    return crop


def horizontal_hijinks(crop):
    horizontal = np.copy(crop)
    horizontal = horizontal[0:horizontal.shape[0] , 400:int(horizontal.shape[1]-100)]

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 10

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 15))
    wholeline = cv2.getStructuringElement(cv2.MORPH_RECT, (cols, 12))

    # Apply morphology operations
    horizontal = cv2.medianBlur(horizontal, 39)
    horizontal = cv2.threshold(horizontal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=2)

    horizontal = cv2.erode(horizontal, wholeline, iterations=2)
    horizontal = cv2.dilate(horizontal, wholeline, iterations=1)

    #cv2.imwrite(f'horizontal_{num}.jpg',horizontal)
    print("Horizont is done!")
    return horizontal


def get_line_regions(horizontal, crop, num):
    regions = np.array(np.mean(horizontal, axis=1))
    y_indices = []
    for i, region in enumerate(regions):
        #print(region)
        if region > 240:
            y_indices.append(True)
            #print(i)
        else:
            y_indices.append(False)
    regions = np.where(np.roll(y_indices,1)!= y_indices)[0]
    print(len(regions)/2)

    singles, single_lines = [], []
    region_img = np.copy(crop)
    for region in regions[0::2]:
        #print(region)
        cv2.line(region_img, (0, region), (crop.shape[1], region), (0, 255, 0), 2)
        singles.append(region)
    #cv2.imwrite(f'horizontal_regions{num}.jpg', region_img)
    print("lines done!")

    for single in singles:
        line = crop[int(single - 25):int(single + 150), 0:crop.shape[1]]
        #(crop, (0, region), (crop.shape[1], region), (0, 255, 0), 2)
        single_lines.append(line)

    return single_lines

"""
def vertical_colsplit(line):
    vertical = np.copy(line)
    #
    # Specify size on line axis
    rows = vertical.shape[0]

    # Create structure element for extracting certical lines through morphology operations
    wholevert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, rows))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 50))


    # Apply morphology operations
    vertical = cv2.medianBlur(vertical, 39)
    vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    vertical = cv2.dilate(vertical, verticalStructure, iterations=2)
    vertical = cv2.erode(vertical, wholevert, iterations=2)
    vertical = cv2.dilate(vertical, wholevert, iterations=1)
    cv2.imshow("single_lines", vertical)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""