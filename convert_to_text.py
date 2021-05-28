# import the necessary packages
from typing import Text
import pytesseract
import cv2
import pdf2image


def image_to_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    #gray = cv2.medianBlur(gray, 3)
    text = pytesseract.image_to_string(gray)
    return text

def pdf_to_text(image):
    # convert pdf to image
    pages = pdf2image.convert_from_path(pdf_path=image,dpi=200, size=(1654,2340))
    text = ""
    # parse through all the images
    for i in range(len(pages)):
        #pages[i].save(str(i) + '.png')
        text = text + " "+ pytesseract.image_to_string(pages[i])
    return text

def get_paragraphs(text):
    par=[]
    # Splitting the data into paragraphs
    p = text.split('\n\n')
    # Removing the paragraphs with length less than 3 as they might be of no use
    for i in p:
        if len(i)>3:
            par.append(i) 
    return par,len(par)

