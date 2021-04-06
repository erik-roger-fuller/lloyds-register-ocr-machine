import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from numpy import random
from line_segment import *
from text_processing import *

def display_singles_near_you(single_lines):
    for single_line in single_lines:
        cv2.imshow("single_lines", single_line)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#text box size is : 2288 wide by 3588 wide
#259 x 359?? or 13:18 if rounded up to 2600 x 3600
#from img_proc_func import cv2_img_pipe, bounding_box_crop, single_lines

num = random.randint(50, 500)
num = "{:03d}".format(num)


num = 470
filename = f"HECROSS1805/ROS1805Ship_jp2/ROS1805Ship_0{num}.jp2"
#filename, num = f"HECROSS1805/ROS1805Ship_jp2/ROS1805Ship_0258.jp2" , 258
#num = 258
print("num = ", num)

img = cv2.imread(filename)
edges, grey = img_processing(img)
crop = houghp_boxfind(edges, grey)
horizontal = horizontal_hijinks(crop)

#newcrop = preprocess_for_ocr(crop)
#cv2.imwrite("ocr.jpg", newcrop)
#print("edit 2 completed!")
#cv2.waitKey(0)
#cv2.destroyAllWindows()

single_lines = get_line_regions(horizontal=horizontal, crop=crop, num=num)

names = []
#display_singles_near_you(single_lines)
for line in single_lines:
    name = line_slice(line)
    names.append(name)


print(names)

"""cv2.imshow("single_lines", list_no)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""


"""cv2.imshow("single_lines", ship_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
"""custom_config = r'--oem 3 --psm 6'
    # #  # outputbase digits' #for nums only
    # custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6' #lowercase alpha only
    text = pytesseract.image_to_string(line, config=custom_config)
    text = text.replace("\n", "")
    print(i, text)
    #line[top:bottom , left:right]
    """

