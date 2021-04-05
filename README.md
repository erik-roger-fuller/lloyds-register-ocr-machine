# lloyds-register-ocr-machine
a program to OCR and then sort the corpus of LLoy'ds registers currently being digitzed by the Lloyd's register foundation. The eventual goal of this is to create dataset with which detailed historical reserach can be performed. 

# Lloyd's Register OCR machine
for those who are unfamiliar, Lloyd's Register is a printed artifact of the birth of modern markets in insurance, shipping and financial speculation. Its roots go back to the the 1600's, but in its present form, since 1765 until the present day it is a list of ships and owners with data such as insurable condition and normal trade route. In addition, Lloyd's itself (no relation to the bank) has set about digitizing all manuscripts in their collection, in line with their new status as a nonprofit. This all couples with the absolutely dreadful automated OCR of these Registers on the internet database, where, as articles in the public domain, they have been uploaded. 

https://hec.lrfoundation.org.uk/archive-library/documents
## intentions: 
this project has two intentions: for me to teach myself the mechanics of text recognition and to provide for the academic community a robust well scanned dataset that encompasses every Lloyd's Register currently in the public domain. These Registers are ripe for a tailored OCR in that they 
* display data in near identical formatting for nearly 100 years
* are nearly all in the public domain
* have immense use to historians already
to that end, i am starting with the 1805 Lloyd's list as a suitably equidistant version that follows the formatting quite closely. 

##  current status 
so far, the project is at the point of successfully segmenting the pages into OCR-able lines- 
i have developed a relatively robust algorithm that segments the page using Hough lines algorithm as well as the simple weighted average and statistical z score in order to find the most likely contender for each line of large font text. 
this solves the problem of the lists: they feature very close together text in a tightly compacted grid pattern which very much complicates OCR features. this combined with the small undertext and the arcane system of abbreviation is shaping up to be a deceptively difficult project. It has certainly been one that fires all few my "cylinders" : mathematical, visual, linguistic and last but not least, historical 

### current to-dos are: 
* configure tesseract OCR engine 
* create custom dictionary to ensure correct reading of terms- this includes the translation of abbreviations (such as *BG* into brig etc)
* create segmentation algorithm to correctly judge the location of text boxes for certain values (IE ship type or tonnage require alphabetical vs numeral int OCR)
* create algorithm that responds to the unique typesetting concentration of Lloyd's list- to save having to lay out so many little lead letters, the lists of this era will use the "---" em dash to signify a repetition of the above value. this is used for both numbers and alphanumeric sequences. while this needs only a simple python flag it would be very vulnerable to data loss int he event of an incomplete OCR. this is good segment to the next problem
* increase the robustness of the detection algorithm and do so in a manner that will send *[rejected]* lines that cannot meet a certain criterion of accuracy to a database. these lines can then be correctly identified with the help of volunteers who can lend their much more skilled but slow eyes to the effort. 
* vis a vis this problem--- the presence of hand written corrections is particularly important. it was common in the 18th/19th century to correct these lists over rte year based on supplements published by Lloyd's as well as news articles. as cursive OCR is still a largely unsolved problem the lines that have been hand edited remain the kingdom of human eyes.
* *[to eventually do]*- use NLP processing to identify differing spellings of names etc in order to eventually-  log all of these ships into a database against which they can be logged. this would in theory allow the visualization of the lives of ships, the geospatial visualization of their routes, and given enough digitized logs, even the careers of ships masters. 

#### necessary packages (so far): 
numpy
opencv python
pytesseract

this git directory initiated april 2021
enjoy!
