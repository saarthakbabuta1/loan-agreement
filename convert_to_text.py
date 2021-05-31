# import the necessary packages
from time import time
from typing import Text
import pytesseract
import cv2
import pdf2image
import time
import re


def image_to_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we should apply thresholding to preprocess the
        # image
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # make a check to see if median blurring should be done to remove
        # noise
        #gray = cv2.medianBlur(gray, 3)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        return(e)

def pdf_to_text(image):
    try:
        # convert pdf to image
        pages = pdf2image.convert_from_path(pdf_path=image,dpi=200, size=(1654,2340))
        text = ""
        # parse through all the images
        start_time = time.time()
        for i in range(len(pages)):
            
            print("page number: {}".format(i))
            #pages[i].save(str(i) + '.png')
            text = text + " "+ pytesseract.image_to_string(pages[i])
        print("PDF to text completed in  %s seconds" % (time.time() - start_time))
        return text

    except Exception as e:
        return(e)

def get_paragraphs(text):
    try:
        par=[]
        # Splitting the data into paragraphs
        p = text.split('\n\n')
        for i in p:
        # Strip whitespace.
            i = i.strip()
            i = re.sub("\s\s+" , " ", i)
        # Removing the paragraphs with length less than 3 as they might be of no use
            if len(i)>3:
                par.append(i) 
        return par,len(par)
    except Exception as e:
        return(e)

