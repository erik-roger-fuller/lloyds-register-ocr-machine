import cv2
import pytesseract
from pytesseract import Output
import pytesseract
import re
import nltk
from line_segment import *

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

def ocr_boxes_in_place(ocr_boxes):
    ocr_dict = {"list_no" : ocr_boxes[0] , "ship_name" : ocr_boxes[1] , "master" : ocr_boxes[2] ,
                "tonnage" : ocr_boxes[3] , "home_port" : ocr_boxes[4] , "prob_len" : ocr_boxes[5] ,
                "prob_underwriter" : ocr_boxes[6] , "prob_beam" : ocr_boxes[7] , "route" : ocr_boxes[8] ,
                "condition" : ocr_boxes[9] }
    return ocr_dict

def read_dict_boxes(ocr_dict, line , i, num):
    list_no = intread(ocr_dict["list_no"])
    ship_name = textread(ocr_dict["ship_name"])
    ship_name = ship_classify(ship_name)

    master = textread(ocr_dict["master"])
    tonnage = intread(ocr_dict["tonnage"])
    home_port = textread(ocr_dict["home_port"])
    prob_len = intread(ocr_dict["prob_len"])
    prob_underwriter = textread(ocr_dict["prob_underwriter"])
    prob_beam = intread(ocr_dict["prob_beam"])
    route = textread(ocr_dict["route"])
    route = location_classify(route)

    condition = textread(ocr_dict["condition"])
    condition = year_sub(condition)

    #page_number = num
    #year = 1805
    print(f"page_pos [{i}]:\t  {list_no} | {ship_name} | {master} | {tonnage}  | {home_port} | "
          f"{prob_len} | {prob_underwriter} | {prob_beam} | {route} | {condition}")

    ocr_dict = { "page_no": num, "page_pos" : i , "list_no" : list_no , "ship_name" : ship_name , "master" : master ,
                "tonnage" : tonnage , "home_port" : home_port , "prob_len" : prob_len ,
                "prob_underwriter" : prob_underwriter , "prob_beam" : prob_beam , "route" : route ,
                "condition" : route,  }

    return ocr_dict



def line_slice(line, line_boxes, i , num):
    line, line_boxes = preprocess_for_ocr(line, 5, line_boxes)
    ocr_boxes = ocr_boundings(line, line_boxes)
    if len(ocr_boxes) != 10:
        raise TypeError
    ocr_dict = ocr_boxes_in_place(ocr_boxes)
    ocr_dict = read_dict_boxes(ocr_dict, line, i, num)
    return ocr_dict

    #name = list_no + " | " + ship_name + "\n\t" + name2
    #name = ship_classify(name)
    #print(name)
    #print(f"ln: ...{intread(list_no)}...{name}")
    #return name

def ship_classify(name):
    name = name.replace("Bg ", "Brig ").replace("Bqe","Barque ").replace("Cr ","Cutter ")
    name = name.replace("Dr ","Dogger ").replace("G ","Galliot ").replace("K ","Ketch ")
    name = name.replace("Lr ","Lugger ").replace("S ","Ship ").replace("H ","Hoy ")
    name = name.replace("Sk ","Smack ").replace("Sp ","Sloop ").replace("Sr ","Schooner ")
    name = name.replace("St ", "Schoot ").replace("Sw ", "Snow ").replace("Yt", "Yacht ")

    name = name.replace(" s ","sheathed").replace("s&d"," sheathed and doubled ").replace("sC","sheathed with Copper")
    name = name.replace("s.C.I.B", "sheathed with Copper and Iron Bolts").replace("s.W.&C", "sheathed with Copper over Boards")
    name = name.replace("C.rp", "Copper repaired").replace("C.lm","Coppered to light-water¬ mar").replace(" Ch ", " chinamed ")

    return name

def location_classify(name):
    name = re.sub(r'^[B][e]{1}([A-Z])', r"Belfast \1", name)
    name = re.sub(r'^[B][r]{1}([A-Z])', r"Bristol \1", name)
    name = re.sub(r'^[C][o]{1}([A-Z])', r"Cork \1", name)
    name = re.sub(r'^[D][u]{1}([A-Z])', r"Dublin \1", name)
    name = re.sub(r'^[E][x]{1}([A-Z])', r"Exeter \1", name)
    name = re.sub(r'^[C][s]{1}([A-Z])', r"Cowes \1", name)

    name = re.sub(r'^[L][h]{1}([A-Z])', r"Leith \1", name)
    name = re.sub(r'^[L][i]{1}([A-Z])', r"Liverpool \1", name)
    name = re.sub(r'^[D][a]{1}([A-Z])', r"Dartmouth \1", name)
    name = re.sub(r'^[F][a]{1}([A-Z])', r"Falmouth \1", name)
    name = re.sub(r'^[G][r]{1}([A-Z])', r"Grenock \1", name)
    name = re.sub(r'^[H][l]{1}([A-Z])', r"Hull \1", name)

    name = re.sub(r'^[L][a]{1}([A-Z])', r"Lancaster \1", name)
    name = re.sub(r'^[L][o]{1}([A-Z])', r"London \1", name)

    name = re.sub(r'^[L][y]{1}([A-Z])', r"Lynn \1", name)
    name = re.sub(r'^[N][c]{1}([A-Z])', r"Newcastle \1", name)
    name = re.sub(r'^[P][o]{1}([A-Z])', r"Poole \1", name)
    name = re.sub(r'^[P][l]{1}([A-Z])', r"Plymouth \1", name)
    name = re.sub(r'^[S][d]{1}([A-Z])', r"Sunderland \1", name)
    name = re.sub(r'^[Y][a]{1}([A-Z])', r"Yarmouth \1", name)

    name = re.sub(r'^[S][h]{1}([A-Z])', r"Shields \1", name)
    name = re.sub(r'^[P][h]{1}([A-Z])', r"Portsmouth \1", name)
    name = re.sub(r'^[T][n]{1}([A-Z])', r"Teignmouth \1", name)
    name = re.sub(r'^[T][p]{1}([A-Z])', r"Topsham \1", name)
    name = re.sub(r'^[W][a]{1}([A-Z])', r"Waterford \1", name)
    name = re.sub(r'^[W][n]{1}([A-Z])', r"Whitehaven \1", name)
    name = re.sub(r'^[W][o]{1}([A-Z])', r"Workington \1", name)
    name = re.sub(r'^[W][y]{1}([A-Z])', r"Whitby \1", name)

    return name

def year_sub(condition):
    condition = re.sub(r'[V][1]\b', 'A 1', condition)
    condition = re.sub(r'^(\d\d)', r'18\1', condition)
    return condition

"""text analysis"""
def textread(roi):
    custom_config = "--psm 12 -c classify_font_name=Times_New_Roman_Bold.ttf tessedit_char_whitelist=Aa BbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz  tessedit_char_blacklist=[]{}=+$-~`;:!"

    text = pytesseract.image_to_string(roi, lang='eng', config=custom_config)
    text = text.replace("\n", " ").replace("[", "").replace("]", "").replace("\x0c", " ")
    text = text.replace("§", "S").replace("\\V", "W").replace("!", "")
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = text.replace(")", "").replace("(", "").replace("/", "")
    text = text.replace("`", "").replace("'", "").replace('"', "")
    #re.sub(r'\b[j]', '', text)

    #text = p.findall(text)
    return text

def intread(roi):
    custom_config = ' --psm 8 -c classify_font_name=Times_New_Roman_Bold.ttf outputbase digits'
    text = pytesseract.image_to_string(roi, config=custom_config)
    text = text.replace("\n", "").replace("\x0c", "")
    #print("debug: ", text)
    """    p = re.compile(r'\d+')
    text = p.findall(text)"""
    return text





"""for contour in contours:
print(contour)
#)
#coords = cv2.boundingRect(contour)
#print(coords)
#text_image = cv2.rectangle(line, (x, y), (x + w, y + h), (0, 255, 0), 2)
#boxes.append([x, y, w, h])
#roi = ((x, x + h), (y, y + w))
roi = line[coords]
#print(roi)
#rois.append(roi)
cv2.imshow("single_lines", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()"""



        #roi = bitnot[x:x + h, y:y + w]


        #print(x, y, w, h)
    #cv2.rectangle(img=line, pt1=, pt2=bottom_right, color=(0, 0, 255), thickness=5)



"""         if (w > 500 and h > 500):
            text_img = cv2.rectangle(line, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])
            
    return boxes"""

"""  
    # lineboundaries =
    # list_no = line[0:750, 0:850]
    # hip_name = line[0:450, 900:12500]
    # ship_ln2 = line[375:875, 950:12500]

    # list_no = intread(list_no)
    # ship_name = textread(ship_name)
    # name2 = textread(ship_ln2)
    # mtest = pytesseract.image_to_data(ship_name, lang='eng', config=r"-c classify_font_name=Times_New_Roman_Bold.ttf" , )
    # print("test:  " , mtest)
 
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