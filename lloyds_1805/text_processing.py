import cv2
import pytesseract
from pytesseract import Output
import pytesseract
import re
import nltk
from line_segment import *

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

def line_slice(line):
    # image = upscale(image, scale_factor)
    line = preprocess_for_ocr(line, 5)
    #list_no = line[0:750, 0:850]
    #hip_name = line[0:450, 900:12500]
    #ship_ln2 = line[375:875, 950:12500]

    #list_no = intread(list_no)
    #ship_name = textread(ship_name)
    #name2 = textread(ship_ln2)
    #mtest = pytesseract.image_to_data(ship_name, lang='eng', config=r"-c classify_font_name=Times_New_Roman_Bold.ttf" , )
    #print("test:  " , mtest)
    """  
    cv2.imshow("single_lines", list_no)
    cv2.waitKey(0)
    cv2.imshow("single_lines", ship_name)
    cv2.waitKey(0)
    cv2.imshow("single_lines", ship_ln2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    img = preprocess_for_ocr(img, 4)
    img = preprocess_for_ocr(img, 5)
    if np.average(ship_ln2[50:200, 100:-100]) < 250:
    """
    name = list_no + " | " + ship_name + "\n\t" + name2
    name = ship_classify(name)
    print(name)
    #print(f"ln: ...{intread(list_no)}...{name}")
    return name

def ship_classify(name):
    name = name.replace("Bg", "Brig")
    return name

"""text analysis"""
def textread(roi):
    custom_config = r"--psm 13 -c classify_font_name=Times_New_Roman_Bold.ttf" #tessedit_char_whitelist=Aa BbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz  tessedit_char_blacklist=[]{}=+$-~`;:!"
    #"
    #--oem 3
    text = pytesseract.image_to_string(roi, lang='eng', config=custom_config)
    text = text.replace("\n", "").replace("[", " | ").replace("]", " | ").replace("\x0c", "")
    text = text.replace("§", "S").replace("\\V", "W").replace("!", " | ")
    text = text.replace("{", " | ").replace("}", " | ").replace("\\", " | ")
    text = text.replace(")", " | ").replace("(", " | ").replace("/", " | ")
    text = text.replace("`", " | ").replace("'", " | ").replace('"', " | ")
    #re.sub(r'\b[j]', '', text)
    re.sub(r"[V1]\b", 'A 1', text)
    #text = p.findall(text)
    return text

def intread(roi):
    custom_config = r'--oem 3 --psm 13 -c classify_font_name=Times_New_Roman_Bold.ttf outputbase digits'
    text = pytesseract.image_to_string(roi, config=custom_config)
    text = text.replace("\n", "").replace("\x0c", "")
    #print("debug: ", text)
    """    p = re.compile(r'\d+')
    text = p.findall(text)"""
    return text