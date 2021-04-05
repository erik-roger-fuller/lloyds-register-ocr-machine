import cv2
import pytesseract
from pytesseract import Output
import pytesseract
import re
import nltk
from line_segment import *

def line_slice(line):
    #list_no = line[0:150, 0:210]
    #list_no = preprocess_for_ocr(list_no, 4)

    ship_name = line[0:110, 0:2400]
    ship_name = preprocess_for_ocr(ship_name, 3)

    ship_ln2 = line[100:165, 190:2400]
    ship_ln2 = preprocess_for_ocr(ship_ln2, 4)

    """    cv2.imshow("single_lines", list_no)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("single_lines", ship_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow("single_lines", ship_ln2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if np.average(ship_ln2[50:200, 100:-100]) < 250:
            
    
    """
    name = textread(ship_name)

    name2 = textread(ship_ln2)
    name = name + "\n\t" + name2

    name = ship_classify(name)
    print(name)
    #print(f"ln: ...{intread(list_no)}...{name}")
    return name

def ship_classify(name):
    name = name.replace("Bg", "Brig")
    return name

"""text analysis"""
def textread(roi):
    custom_config = r"--psm 8 -c classify_font_name=Times_New_Roman_Bold.ttf" #tessedit_char_whitelist=Aa BbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz  tessedit_char_blacklist=[]{}=+$-~`;:!"
    #"
    #--oem 3
    text = pytesseract.image_to_string(roi, lang='eng', config=custom_config)
    text = text.replace("\n", "").replace("[", " | ").replace("]", " | ").replace("\x0c", "")
    text = text.replace("§", "S").replace("\\V", "W").replace("!", " | ")
    text = text.replace("{", " | ").replace("}", " | ").replace("\\", " | ")
    text = text.replace(")", " | ").replace(")", " | ").replace("/", " | ")
    #re.sub(r'\b[j]', '', text)
    re.sub(r'[V1]\b', 'A 1', text)
    #text = p.findall(text)
    return text

def intread(roi):
    custom_config = r'--oem 3 --psm 13 -c classify_font_name=Times_New_Roman_Bold.ttf outputbase digits'
    text = pytesseract.image_to_string(roi, config=custom_config)
    text = text.replace("\n", "")
    #print("debug: ", text)
    p = re.compile(r'\d+')
    text = p.findall(text)
    #text = text[0]
    return text