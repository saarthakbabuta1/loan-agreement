import cv2
import numpy as np
import pytesseract
import re
import pdf2image

file = "GHPL_Loan_Agreement-edited.pdf"

#file = "image.png"

def get_contours(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts

def pdf_to_text(file):
    par = []
    body = ""
    if file.rsplit('.', 1)[1].lower() == 'pdf':
        pages = pdf2image.convert_from_path(pdf_path=file,dpi=200, size=(1654,2340))
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        for i in range(len(pages)):
            data = []
            filename = file+str(i) + '.png'
            pages[i].save(filename)
            image = cv2.imread(filename)
            cnts = get_contours(img=image)
                
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                cropped = image[y:y + h, x:x + w]
                text = pytesseract.image_to_string(cropped)
                data = [text] + data
                body = text + " " + body 
            par.append(data)

    return body,par

def image_to_text(file):
    image = cv2.imread(file)
    cnts = get_contours(img=image)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        cropped = image[y:y + h, x:x + w]
        body = pytesseract.image_to_string(cropped)
        par = [body] + par
    
    return body,par



def plot_boundary(file):
    par = []
    body = ""
    if file.rsplit('.', 1)[1].lower() == 'pdf':
        pages = pdf2image.convert_from_path(pdf_path=file,dpi=200, size=(1654,2340))
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        for i in range(len(pages)):
            data = []
            filename = file+str(i) + '.png'
            pages[i].save(filename)
            image = cv2.imread(filename)
            cnts = get_contours(img=image)
                
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                cropped = image[y:y + h, x:x + w]
                text = pytesseract.image_to_string(cropped)
                data = [text] + data
                body = text + " " + body 
            par.append(data)
    else:

        image = cv2.imread(file)
        cnts = get_contours(img=image)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            cropped = image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            par = [text] + par
    paragraph = []
    for i in par:
        print("length",len(i))
        if(len(i)>1):
            for j in i:
                j = re.sub("\s\s+", " ", j)
                j = re.sub("\n"," ",j)
                j = j.strip()
                if len(j) > 3:
                    text = text + j
                    paragraph.append(j)
        else:
            i = re.sub("\s\s+", " ", i)
            i = re.sub("\n"," ",i)
            i = i.strip()
            if len(i)>3:
                text = text + i
                paragraph.append(i)

    return {'text':body,'paragraphs':paragraph}

print(plot_boundary(file)['paragraphs'])