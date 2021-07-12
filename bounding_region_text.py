from typing import Text
import cv2
import numpy as np
import pytesseract
import re
import pdf2image
import docx
from helper import regex,is_heading

def get_contours(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=7)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts

def pdf_to_text(file):
    par = []
    body = ""
    try:
        pages = pdf2image.convert_from_path(pdf_path=file,dpi=200, size=(1654,2340))
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        print("Converted PDF to Pages")
        for i in range(len(pages)):
            data = []
            filename = file+str(i) + '.png'
            pages[i].save(filename)
            image = cv2.imread(filename)
            cnts = get_contours(img=image)
                
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if w>50:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                    cropped = image[y:y + h, x:x + w]
                    text = pytesseract.image_to_string(cropped)
                    data = [text] + data
                    body = text + " " + body

            
            cv2.imwrite(filename,image)
            par.append(data)
            print("Done {}".format(filename))
    except Exception as e:
        print(e)
        return({"Error:",e})

    return {"body":body,"paragraph":par}

def image_to_text(file):
    try:
        image = cv2.imread(file)
        cnts = get_contours(img=image)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            cropped = image[y:y + h, x:x + w]
            cv2.imshow()
            body = pytesseract.image_to_string(cropped)
            par = [body] + par
    
    except Exception as e:
        print(e)
        return({"Error:",e})
    
    return {"body":body,"paragraph":par}

def word_to_text(file):
    try:
        par = []
        doc = docx.Document(file)
        for i in doc.paragraphs:
            text = regex(i.text)
            if len(text)>0:
                par.append(text)
        return par,len(par)
    except Exception as e:
        return(e)

def get_paragraphs(par):
    print("Paragraph process started")
    try:
        paragraph = []
        heading = []
        for i in par:
            if(len(i)>1):
                for j in i:
                    text = regex(j)
                    if len(text) > 3 and text.lower() != 'page':
                        paragraph.append(text)
                        heading.append(is_heading(text))
            else:
                text = regex(i)
                if len(text) > 3 and text.lower() != 'page':
                    paragraph.append(text)
                    heading.append(is_heading(text))
        print("Paragraph process complete")
        return {"paragraph":paragraph,"heading":heading}
    except Exception as e:
        print("Error Here",e)
        return e


