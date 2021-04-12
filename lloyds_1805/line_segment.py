import cv2
import numpy as np
from image_editing import *

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
    #print(rightedge)

    top_left = (int(rightedge-2300) , int(ycenter-1900)) #
    bottom_right = (rightedge, int(ycenter+1900))
    print(top_left, bottom_right)
    #cv2.rectangle(img=adj, pt1=top_left, pt2=bottom_right, color=(0, 0, 255), thickness=5)
    #cv2.imwrite('houghlines5.jpg',img)
    #print("box is done!")
    crop = rectrangle_crop(adj, top_left, bottom_right)
    print("crop:", crop.shape)
    #cv2.imwrite('houghlines6.jpg',crop)
    #print("crop is done!")
    return crop

def horizontal_hijinks(crop):
    horizontal = crop
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    #cv2.imread(horizontal)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 15))
    wholeline = cv2.getStructuringElement(cv2.MORPH_RECT, (cols, 12))

    # Apply morphology operations
    horizontal = remove_noise(horizontal, 39)
    horizontal = cv2.threshold(horizontal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=2)
    horizontal = cv2.erode(horizontal, wholeline, iterations=2)
    horizontal = cv2.dilate(horizontal, wholeline, iterations=1)
    #cv2.imwrite(f'horizontal_{num}.jpg',horizontal)
    print("Horizont is done!")
    return horizontal

def get_line_regions(horizontal, vertical, crop, num):
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

    singles, single_lines, single_lines_boxes = [], [], []
    #region_img = np.copy(crop)
    for region in regions[0::2]:
        #print(region)
        #cv2.line(region_img, (0, region), (crop.shape[1], region), (0, 255, 0), 2)
        singles.append(region)
    #cv2.imwrite(f'horizontal_regions{num}.jpg', region_img)
    print("lines done!")

    for single in singles:
        yup, ydown = 25, 150
        line = crop[int(single - yup):int(single + ydown), 0:crop.shape[1]]
        line_vert = vertical[int(single - yup):int(single + ydown), 0:crop.shape[1]]
        line_horiz = horizontal[int(single - yup):int(single + ydown), 0:crop.shape[1]]

        t_lo = (0, 0)
        b_ro = (line_horiz.shape[1], line_horiz.shape[0])
        cv2.rectangle(img=line_horiz, pt1=t_lo, pt2=b_ro, color=(255, 255, 255), thickness=-5)
        boundaries = cv2.bitwise_and(line_horiz, line_vert)

        line = cv2.copyMakeBorder(line, 5, 5, 5, 5, cv2.BORDER_CONSTANT )
        boundaries = cv2.copyMakeBorder(boundaries, 5, 5, 5, 5, cv2.BORDER_CONSTANT)

        #boundariestest = cv2.bitwise_and(line, line_vert)
        #boundariestest = cv2.bitwise_xor(cv2.bitwise_and(line, boundaries), boundaries)
        #single_lines_boxes = boundaries[int(single - yup):int(single + ydown), 0:crop.shape[1]]
        #(crop, (0, region), (crop.shape[1], region), (0, 255, 0), 2)

        single_lines.append(line)
        single_lines_boxes.append(boundaries)
    return single_lines, single_lines_boxes


def vertical_colsplit(line, horizontal):
    vertical = line
    rows = vertical.shape[0]
    #rows = rows // 150
    # Create structure element for extracting certical lines through morphology operations
    wholevert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    lastpass = cv2.getStructuringElement(cv2.MORPH_RECT, (3, rows))

    # Apply morphology operations
    vertical = cv2.dilate(vertical, verticalStructure, iterations=3)
    vertical = cv2.erode(vertical, wholevert, iterations=3)
    vertical = cv2.dilate(vertical, wholevert, iterations=5)
    vertical = cv2.dilate(vertical, lastpass, iterations=2)
    vertical = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return vertical

def get_contour_precedence(contour):
    origin = cv2.boundingRect(contour)
    x_loc = origin[0]
    return x_loc

def ocr_boundings(line, line_boxes):
    contours, hierarchy = cv2.findContours(line_boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ocr_boxes = []
    contours.sort(key=lambda x: get_contour_precedence(x))
    for cnt in contours:
        #cv2.drawContours(line, [cnt], 0, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(cnt)
        #print(w)
        if w < 100:
            del cnt
            #print(x, y, w, h)
            pass
        else:
            cropped_contour = line[y:y + h, x:x + w]
            ocr_boxes.append(cropped_contour)
        #print("text boxes: " , len(ocr_boxes), " contours:  ", len(contours))
    return ocr_boxes

"""
        cv2.imshow("single_lines", cropped_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


t_lo = (0, int(single - yup))
b_ro = (crop.shape[1], int(single + ydown))
cv2.rectangle(img=line_horiz, pt1=t_lo, pt2=b_ro, color=(0, 0, 0), thickness=-5)
t_l = (0, int(single - yup + 3))
b_r = (crop.shape[1], int(single + ydown - 3))
cv2.rectangle(img=line_horiz, pt1=t_l, pt2=b_r, color=(255, 255, 250), thickness=-5)"""


